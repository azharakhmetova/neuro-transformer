import os
import torch
import warnings
import torchinfo
import typing as t
from torch import nn
import torch.distributed
from torch.utils.data import DataLoader
import numpy as np
from functools import partial

from v1t.models.core import get_core
from v1t.models.readout import Readouts
from v1t.models.neuron_processing import NeuronIDTokenizer
from v1t.models.neuron_processing import SampleNeuronIDs
from v1t.utils.tensorboard import Summary
from v1t.models.core_shifter import CoreShifters
from v1t.models.image_cropper import ImageCropper
from v1t.models.utils import ELU1, load_pretrain_core


def get_model_info(
    model: nn.Module,
    input_data: t.Union[torch.Tensor, t.Sequence[t.Any], t.Mapping[str, t.Any], t.Any],#, torch.Tensor],#, torch.Tensor],
    mouse_id: str = None,
    filename: str = None,
    summary: Summary = None,
    device: torch.device = "cpu",
    tag: str = "model/trainable_parameters",
):
    args = {
        "model": model,
        "input_data": input_data, 
        "depth": 5,
        "device": device,
        "verbose": 0,
    }
    if mouse_id is not None:
        args["mouse_id"] = mouse_id

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model_info = torchinfo.summary(**args)

    if filename is not None:
        with open(filename, "w") as file:
            file.write(str(model_info))
    if summary is not None:
        summary.scalar(tag, model_info.trainable_params)
    return model_info


class Model(nn.Module):
    """
    shift mode:
        0 - disable shifter
        1 - shift input to core module
        2 - shift input to readout module
        3 - shift input to both core and readout module
        4 - shift_mode=3 and provide both behavior and pupil center to cropper
    """

    def __init__(self, args: t.Any, ds: t.Dict[str, DataLoader], name: str = "Model"):
        super(Model, self).__init__()
        assert isinstance(
            args.output_shapes, dict
        ), "output_shapes must be a dictionary of mouse_id and output_shape"
        self.name = name
        self.input_shape = args.input_shape
        self.output_shapes = args.output_shapes
        self.shift_mode = args.shift_mode
        self.readout_type = args.readout
        self.core_type = args.core
        self.tokenize_neurons = args.tokenize_neurons
        self.emb_dim_tokenizer = args.emb_dim_tokenizer
        self.frac_input_neurons = args.frac_input_neurons
        self.micro_batch_size = args.micro_batch_size
        self.register_buffer("reg_scale", torch.tensor(args.core_reg_scale))
        print("tokenize neurons: ", self.tokenize_neurons)
        if self.tokenize_neurons == 1:
            self.neuron_id_tokenizer = NeuronIDTokenizer(num_neurons=list(self.output_shapes.items())[0][1][0], emb_dim=args.emb_dim_tokenizer, device=args.device)
            print(self.neuron_id_tokenizer)
            self.neuron_id_sampler = SampleNeuronIDs(num_neurons=list(self.output_shapes.items())[0][1][0], frac_input_neurons=args.frac_input_neurons, device=args.device)

        self.add_module(
            "image_cropper",
            module=ImageCropper(args, ds=ds),
        )
        print("core: ", self.core_type)
        if self.core_type == "multimodalvit":
            print("here")
            self.add_module(
                name="core",
                module=get_core(args)(
                    args,
                    input_shape=self.image_cropper.output_shape,
                    num_neurons=list(self.output_shapes.items())[0][1][0],
                    num_samples_per_neuron=1,
                ),
            )
            print("here 2")
        else:
            self.add_module(
                name="core",
                module=get_core(args)(
                    args,
                    input_shape=self.image_cropper.output_shape,
                ),
            )
        if self.shift_mode in (2, 3, 4):
            self.add_module(
                "core_shifter",
                module=CoreShifters(
                    args,
                    mouse_ids=list(ds.keys()),
                    input_channels=2,
                    hidden_features=5,
                    num_layers=3,
                ),
            )
        else:
            self.core_shifter = None
        self.add_module(
            name="readouts",
            module=Readouts(
                args,
                model=args.readout,
                input_shape=self.core.output_shape,
                output_shapes=self.output_shapes,
                ds=ds,
            ),
        )

        self.elu1 = ELU1()

    @property
    def device(self) -> torch.device:
        """return the device that the model parameters is on"""
        return next(self.parameters()).device

    def get_parameters(self, core_lr: float):

        # separate learning rate for core module from the rest
        params = []
        if not self.core.frozen:
            params.append(
                {
                    "params": self.core.parameters(),
                    "lr": core_lr,
                    "name": "core",
                }
            )
        params.append({"params": self.readouts.parameters(), "name": "readouts"})
        if self.image_cropper.image_shifter is not None:
            params.append(
                {
                    "params": self.image_cropper.parameters(),
                    "name": "image_cropper",
                }
            )
        if self.core_shifter is not None:
            params.append(
                {
                    "params": self.core_shifter.parameters(),
                    "name": "core_shifter",
                }
            )
        return params
    
    def tokenizer_l1(self, reduction: str = "sum"):
        l1 = self.neuron_id_tokenizer.embedding.weight.abs()
        return l1.sum() if reduction == "sum" else l1.mean()

    def regularizer(self, mouse_id: str, reduction: str = "sum"):
        reg = 0
        reg += self.reg_scale * self.tokenizer_l1(reduction=reduction)
        if not self.core.frozen:
            reg += self.core.regularizer()
        reg += self.readouts.regularizer(mouse_id=mouse_id)
        reg += self.image_cropper.regularizer(mouse_id=mouse_id)
        if self.core_shifter is not None:
            reg += self.core_shifter.regularizer(mouse_id=mouse_id)
        return reg

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
        neuron_inputs: torch.Tensor = None,
        input_neuron_ids: torch.Tensor = None,
        query_neuron_ids: torch.Tensor = None,
        # dummy: bool = False,
        activate: bool = True,
    ):
        images, image_grids = self.image_cropper(
            inputs,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )
        # if not dummy and self.tokenize_neurons:
        #     input_neuron_ids, query_neuron_ids = self.neuron_id_sampler()
        #     print("input neuron ids: ", input_neuron_ids.shape)
        #     print("input_neuron_ids dtype:", input_neuron_ids.dtype)
        #     print("input_neuron_ids device:", input_neuron_ids.device)
        #     print("query neuron ids: ", query_neuron_ids.shape)
        # elif dummy and self.tokenize_neurons:
        #     print("dummy forward pass")
        #     input_neuron_ids = torch.arange(int(self.frac_input_neurons * list(self.output_shapes.items())[0][1][0])).view(-1)
        #     query_neuron_ids = torch.arange(int(self.frac_input_neurons * list(self.output_shapes.items())[0][1][0]), list(self.output_shapes.items())[0][1][0]).view(-1)
        #     # self.neuron_id_tokenizer = lambda neuron_ids: torch.zeros(neuron_ids.size(0), self.emb_dim_tokenizer, device="cpu")
        # else:
        #     input_neuron_ids = None
        #     query_neuron_ids = None
        

        if self.core_type == "multimodalvit" and neuron_inputs is not None:
            if neuron_inputs.dim() != 3:
                neuron_inputs = neuron_inputs.unsqueeze(-1)
            print("neuron inputs: ", neuron_inputs.shape)
        outputs = self.core(
            inputs=images,
            neuron_inputs=neuron_inputs,
            input_neuron_ids=input_neuron_ids,
            neuron_id_tokenizer=self.neuron_id_tokenizer,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )    
        print("outputs after core: ", outputs.shape)
        shifts = None
        if self.core_shifter is not None:
            shifts = self.core_shifter(pupil_centers, mouse_id=mouse_id)
        if self.readout_type == "attention":
            outputs = self.readouts(outputs, mouse_id=mouse_id, neuron_ids=query_neuron_ids.to(torch.long), neuron_id_tokenizer=self.neuron_id_tokenizer, shifts=shifts)
            print("outputs after attention readout: ", outputs.shape)
        else:
            outputs = self.readouts(outputs, mouse_id=mouse_id, shifts=shifts)
            print("outputs after gaussian readout: ", outputs.shape)
        if activate:
            outputs = self.elu1(outputs)
        return outputs, images, image_grids


def get_model(args, ds: t.Dict[str, DataLoader], summary: Summary = None) -> Model:
    print("get_model called", flush=True)
    model = Model(args, ds=ds)
    # model.to(args.device)
    print("Model device: ", model.device)
    for name, param in model.named_parameters():
        if param.device.type == 'cpu':
            print(f"{name} is on {param.device}")

    if hasattr(args, "pretrain_core") and args.pretrain_core:
        load_pretrain_core(args, model=model, device=args.device)
        model.core.freeze()

    # get model info
    mouse_id = args.mouse_ids[0]
    batch_size = args.micro_batch_size
    random_input = lambda size: torch.rand(*size)
    
    model.core.forward = partial(model.core.forward, neuron_id_tokenizer=model.neuron_id_tokenizer)
    for key, readout in model.readouts.items():
        readout.forward = partial(readout.forward, neuron_id_tokenizer=model.neuron_id_tokenizer)
    N = list(model.output_shapes.items())[0][1][0]
    input_data = {
        "inputs": random_input((batch_size, *model.input_shape)),
        "behaviors": random_input((batch_size, 3)),
        "pupil_centers": random_input((batch_size, 2)),
        "neuron_inputs": random_input((batch_size, N, 1)),
        # "dummy": True,
        }
    if args.tokenize_neurons:
        K = int(args.frac_input_neurons * N)
        input_data["input_neuron_ids"] = torch.arange(K, dtype=torch.long, device="cpu")#.view(-1)
        input_data["query_neuron_ids"] = torch.arange(K, N, dtype=torch.long, device="cpu")#.view(-1)
    else:
        input_data["input_neuron_ids"] = None
        input_data["query_neuron_ids"] = None
    model_info = get_model_info(
        model=model,
        input_data=input_data,
        mouse_id=mouse_id,
        filename=os.path.join(args.output_dir, "model.txt"),
        summary=summary,
    )
    args.trainable_params = model_info.trainable_params
    if args.verbose > 2:
        print(str(model_info))

    # get core info
    get_model_info(
        model=model.core,
        input_data={
            "inputs": random_input((batch_size, *model.core.input_shape)),
            "behaviors": random_input((batch_size, 3)),
            "pupil_centers": random_input((batch_size, 2)),
            "neuron_inputs": random_input((batch_size, N, 1)),
            "input_neuron_ids": torch.arange(K, dtype=torch.long, device="cpu").view(-1), #.repeat(batch_size),
            # "neuron_id_tokenizer": lambda neuron_ids: torch.zeros(neuron_ids.size(0), args.emb_dim_tokenizer, device="cpu"),
        },
        mouse_id=mouse_id,
        filename=os.path.join(args.output_dir, "model_core.txt"),
        summary=summary,
        tag="model/trainable_parameters/core",
    )
    # get readout summary
    get_model_info(
        model=model.readouts[mouse_id],
        input_data={
            "inputs": random_input((batch_size, *model.core.output_shape)),
            },
        filename=os.path.join(args.output_dir, "model_readout.txt"),
        summary=summary,
        tag=f"model/trainable_parameters/Mouse{mouse_id}Readout",
    )

    model.to(args.device)
    print("Model device after to(args.device): ", model.device)
    return model
