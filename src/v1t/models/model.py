import os
import torch
import warnings
import torchinfo
import typing as t
from torch import nn
import torch.distributed
from torch.utils.data import DataLoader


from v1t.models.core import get_core
from v1t.models.readout import Readouts
from v1t.models.neuron_processing import NeuronIDTokenizer
from v1t.models.neuron_processing import SimpleResponsesTokenizer 
from v1t.models.image_processing import Image2Patches
from v1t.models.layers import PositionalEncoding
from v1t.utils.tensorboard import Summary
from v1t.models.core_shifter import CoreShifters
from v1t.models.image_cropper import ImageCropper
from v1t.models.utils import ELU1, load_pretrain_core


def get_model_info(
    model: nn.Module,
    input_data: t.Union[torch.Tensor, t.Sequence[t.Any], t.Mapping[str, t.Any]],
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
    def __init__(self, args: t.Any, ds: t.Dict[str, DataLoader], name: str = "Model"):
        super(Model, self).__init__()
        assert isinstance(
            args.num_output_neurons, dict
        ), "output_shapes must be a dictionary of mouse_id and output_shape"
        self.name = name
        self.input_shape = args.input_shape
        self.num_output_neurons = args.num_output_neurons
        print("num_output_neurons", self.num_output_neurons)
        self.shift_mode = args.shift_mode
        self.readout_type = args.readout
        self.use_pe_after_core = args.use_pe_after_core
        self.pe_after_core = args.pe_after_core
        self.use_input_neuron_pe = args.use_input_neuron_pe
        self.use_query_neuron_pe = args.use_query_neuron_pe
        self.neuron_pe_mode  = args.neuron_pe_mode
        self.tokenize_neurons = args.tokenize_neurons
        self.frac_input_neurons = args.frac_input_neurons
        if self.tokenize_neurons == 1:
            self.session_tokenizer = nn.Embedding(len(self.num_output_neurons), args.emb_dim_n_id)
            self.sessions_enc = {}
            for i, s in enumerate(self.num_output_neurons.keys()):
                self.sessions_enc[s] = torch.Tensor([i]).long().to(args.device)

            self.neuron_id_tokenizer = nn.ModuleDict({})
            for key, num_neurons in self.num_output_neurons.items():
                self.neuron_id_tokenizer[key] = NeuronIDTokenizer(num_neurons=num_neurons[0], emb_dim=args.emb_dim_n_id)#, device=args.device)

        self.add_module(
            "image_cropper",
            module=ImageCropper(args, ds=ds),
        )
        self.patch_embedding = Image2Patches(
            image_shape=self.image_cropper.output_shape,
            patch_mode=args.patch_mode,
            patch_size=args.patch_size,
            stride=args.patch_stride,
            emb_dim=args.emb_dim_image,
            dropout=args.p_dropout,
            learned = args.learned_pe_before_core,
            pe_mode=args.pe_before_core,
        )
        self.input_neuron_embedding = SimpleResponsesTokenizer(
            args,
            num_output_neurons=args.num_output_neurons,
            num_samples_per_neuron=1,
            frac_input_neurons=args.frac_input_neurons,
            num_samples_per_token=args.num_samples_per_token,
            emb_dim=args.emb_dim_n_response,
            device=args.device,
            use_masking=None,
        )

        if self.neuron_pe_mode in ("1d", "both"):
            self.neuron_pe = PositionalEncoding(
                d_model=args.emb_dim_n_response,
                mode="1d",
                )
        if self.neuron_pe_mode in ("coord", "both"):
            self.neuron_coord_pe = nn.Linear(3, args.emb_dim_n_response)

        self.project_query_pe = args.emb_dim_n_response != args.emb_dim_n_id
        if self.project_query_pe:
            self.projection_query_pe = nn.Linear(args.emb_dim_n_response, args.emb_dim_n_id)
            
        self.add_module(
            name="core",
            module=get_core(args)(
                args,
                # input_shape=self.image_cropper.output_shape,
                num_image_patches=self.patch_embedding.num_patches,
                num_neuron_tokens=self.input_neuron_embedding.num_input_tokens,
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
                output_shapes=self.num_output_neurons,
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

        if self.tokenize_neurons:
            params.append(
                {
                    "params": self.session_tokenizer.parameters(),
                    "name": "session_tokenizer",
                }
            )
            for key, tokenizer in self.neuron_id_tokenizer.items():
                params.append(
                    {
                        "params": tokenizer.parameters(),
                        "name": f"neuron_id_tokenizer_{key}",
                        "weight_decay": 0.0,
                    }
                )
        
            params.append(
                {
                    "params": self.input_neuron_embedding.parameters(),
                    "name": "input_neuron_embeddings",
                }
            )
            if self.neuron_pe_mode in ("1d", "both"):
                params.append(
                    {
                        "params": self.neuron_pe.parameters(),
                        "name": "neuron_fixed_positional_embeddings",
                    }
                )
            if self.neuron_pe_mode in ("coord", "both"):
                params.append(
                    {
                        "params": self.neuron_coord_pe.parameters(),
                        "name": "neuron_coordinate_positional_embeddings",
                    }
                )
            
        params.append(
            {
                "params": self.patch_embedding.parameters(),
                "lr": core_lr,
                "name": "patch_embedding",
            }
        )

        if not self.core.frozen:
            params.append(
                {
                    "params": self.core.parameters(),
                    "lr": core_lr,
                    "name": "core",
                }
            )
        params.append(
                {
                    "params": self.readouts.parameters(), 
                    "name": "readouts"
                }
            )
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
    
    def id_tokenizer_l2(self, reduction: str = "sum"):
        all_weights = torch.cat(
            [tok.embedding.weight.view(-1) for tok in self.neuron_id_tokenizer.values()]
        )
        l2 = all_weights.pow(2)
        return l2.sum() if reduction == "sum" else l2.mean()
    
    def id_tokenizer_l1(self, reduction: str = "sum"):
        all_weights = torch.cat(
            [tok.embedding.weight.view(-1) for tok in self.neuron_id_tokenizer.values()]
        )
        l1 = all_weights.abs()
        return l1.sum() if reduction == "sum" else l1.mean()

    def regularizer(self, mouse_id: str):
        reg = 0
        reg += self.id_tokenizer_l1(reduction="sum") * 0.0076
        if not self.core.frozen:
            reg += self.core.regularizer()
        # reg += self.readouts.regularizer(mouse_id=mouse_id)
        reg += self.image_cropper.regularizer(mouse_id=mouse_id)
        if self.core_shifter is not None:
            reg += self.core_shifter.regularizer(mouse_id=mouse_id)
        return reg

    def forward(
        self,
        images: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
        responses: torch.Tensor = None,
        input_neuron_ids: torch.Tensor = None,
        query_neuron_ids: torch.Tensor = None,
        neuron_coords: torch.Tensor = None,
        activate: bool = True,
    ):
        images, image_grids = self.image_cropper(
            images,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )

        image_tokens = self.patch_embedding(images) 

        if self.frac_input_neurons > 0:
            # add time dimension for image responses
            if responses.dim() != 3:
                responses = responses.unsqueeze(-1)
            input_neuron_id_tokens = self.neuron_id_tokenizer[mouse_id](input_neuron_ids.to(torch.long)) + self.session_tokenizer(self.sessions_enc[mouse_id].to(self.device))  # (B, K, emb_dim_n_id)
            input_neuron_tokens = self.input_neuron_embedding(
                responses=responses[:, input_neuron_ids, :],
                mouse_id=mouse_id,
                neuron_id_tokens=input_neuron_id_tokens,
            )
            if self.use_input_neuron_pe:
                if self.neuron_pe_mode == "1d":
                    # print(self.neuron_pe(responses).shape)
                    # print(self.neuron_pe(responses)[:, input_neuron_ids, :].shape)
                    input_neuron_tokens += self.neuron_pe(responses)[:, input_neuron_ids, :]
                elif self.neuron_pe_mode == "coord":
                    input_coords = neuron_coords[:, input_neuron_ids, :]    # (B, K, 3)
                    input_neuron_tokens += self.neuron_coord_pe(input_coords) #[:, input_neuron_ids.to(torch.long), :]
                elif self.neuron_pe_mode == "both":
                    input_coords = neuron_coords[:, input_neuron_ids, :]  
                    input_neuron_tokens += (self.neuron_pe(responses)[:, input_neuron_ids, :] + self.neuron_coord_pe(input_coords))
        else:
            input_neuron_tokens = None

        outputs = self.core(    # (B, num_tokens, num_channels)
            image_tokens=image_tokens,
            neuron_tokens=input_neuron_tokens,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )
        # CHANGE LEARNABLE CASE and remove it from attention readout
        # add positional encoding after the core 
        if self.use_pe_after_core:
            if self.pe_after_core == "2d":
                temp = outputs.reshape(outputs.shape[0], self.patch_embedding.height, self.patch_embedding.width, outputs.shape[-1])  # (B, h, w, num_channels)
                temp += self.patch_embedding.pos_embedding(temp)
                outputs = temp.reshape(outputs.shape[0], -1, outputs.shape[-1])  # (B, num_tokens, num_channels)
            elif self.pe_after_core == "1d":
                outputs += self.patch_embedding.pos_embedding(outputs)

        shifts = None
        if self.core_shifter is not None:
            shifts = self.core_shifter(pupil_centers, mouse_id=mouse_id)
        
        if self.readout_type == "attention":
            query_neurons = self.session_tokenizer(self.sessions_enc[mouse_id].to(self.device)) + self.neuron_id_tokenizer[mouse_id](neuron_ids=query_neuron_ids.to(torch.long).unsqueeze(0).expand(input_neuron_tokens.shape[0], -1)) # (B, N-K, emb_dim_n_id)
            if self.use_query_neuron_pe:   
                if self.neuron_pe_mode == "1d":
                    if self.project_query_pe:
                        query_neurons += self.projection_query_pe(self.neuron_pe(responses)[:, query_neuron_ids, :])
                    else:
                        query_neurons += self.neuron_pe(responses)[:, query_neuron_ids, :]
                elif self.neuron_pe_mode == "coord":
                    query_coords = neuron_coords[:, query_neuron_ids, :]
                    if self.project_query_pe:
                        query_neurons += self.projection_query_pe(self.neuron_coord_pe(query_coords))
                    else:
                        query_neurons += self.neuron_coord_pe(query_coords)
                elif self.neuron_pe_mode == "both":
                    query_coords = neuron_coords[:, query_neuron_ids, :]
                    if self.project_query_pe:
                        query_neurons += (self.projection_query_pe(self.neuron_coord_pe(query_coords)) + self.projection_query_pe(self.neuron_pe(responses)[:, query_neuron_ids, :]))
                    else:
                        query_neurons += (self.neuron_coord_pe(query_coords) + self.neuron_pe(responses)[:, query_neuron_ids, :])
        outputs = self.readouts(outputs, mouse_id=mouse_id, query_neurons=query_neurons, shifts=shifts) # (B, num_neurons)
        # print("model readout output shape: ", outputs.shape)
        if activate:
            outputs = self.elu1(outputs)
        return outputs, images, image_grids


def get_model(args, ds: t.Dict[str, DataLoader], summary: Summary = None) -> Model:
    model = Model(args, ds=ds)
    print("get model")
    if hasattr(args, "pretrain_core") and args.pretrain_core:
        load_pretrain_core(args, model=model, device=args.device)
        model.core.freeze()

    # get model info
    mouse_id = args.mouse_ids[0]
    batch_size = args.micro_batch_size
    random_input = lambda size: torch.rand(*size)
    N = list(model.num_output_neurons.items())[0][1][0]
    print("N neurons", N)
    input_data={
        "images": random_input((batch_size, *model.input_shape)),
        "responses": random_input((batch_size, N, 1)),
        "neuron_coords": random_input((batch_size, N, 3)),
        "behaviors": random_input((batch_size, 3)),
        "pupil_centers": random_input((batch_size, 2)),
        # "neuron_coords": random_input((batch_size, N, 3)),
    }
    # if args.tokenize_neurons and args.frac_input_neurons == 0.0:
    #     input_data["query_neuron_ids"] = torch.arange(N, dtype=torch.long, device="cpu")#.view(-1)
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
            "image_tokens": random_input((batch_size, *model.patch_embedding.output_shape)),
            "neuron_tokens": random_input((batch_size, *model.input_neuron_embedding.output_shapes[mouse_id])),
            "behaviors": random_input((batch_size, 3)),
            "pupil_centers": random_input((batch_size, 2)),
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
            "query_neurons": random_input((batch_size, K, args.emb_dim_n_id)),
            },
        filename=os.path.join(args.output_dir, "model_readout.txt"),
        summary=summary,
        tag=f"model/trainable_parameters/Mouse{mouse_id}Readout",
    )
    print("exit get_model")
    model.to(args.device)
    return model
