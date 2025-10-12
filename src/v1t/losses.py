import torch
import numpy as np
import typing as t
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
import math

from v1t.models.utils import BufferDict

_CRITERION = dict()


def register(name):
    def add_to_dict(fn):
        global _CRITERION
        _CRITERION[name] = fn
        return fn

    return add_to_dict


REDUCTION = t.Literal["sum", "mean"]
EPS = torch.finfo(torch.float32).eps


def msse(y_true: torch.Tensor, y_pred: torch.Tensor, reduction: REDUCTION = "sum"):
    """Mean sum squared error"""
    loss = torch.square(y_true - y_pred)
    loss = torch.sum(loss, dim=-1)  # sum over neurons
    return torch.sum(loss) if reduction == "sum" else torch.mean(loss)


def poisson_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    eps: t.Union[float, torch.Tensor] = 1e-12,
    reduction: REDUCTION = "sum",
):
    loss = y_pred - y_true * torch.log(y_pred + eps)
    loss = torch.sum(loss, dim=-1)  # sum over neurons
    return torch.sum(loss) if reduction == "sum" else torch.mean(loss)

# code adapted from https://github.com/neurallatents/nlb_tools/blob/1ddc15f45b56388ff093d1396b7b87b36fa32a68/nlb_tools/evaluation.py#L252
def bits_per_spike(
    y_true: torch.Tensor, # true spikes
    y_pred: torch.Tensor, # predicted rates
):
    """Computes bits per spike of rate predictions given true spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = poisson_loss(y_true, y_pred)
    mean_per_neuron = torch.nanmean(y_true, dim=0, keepdim=True)  # (1, N)
    null_rates = mean_per_neuron.expand_as(y_true)                # (B, N)
    nll_null = poisson_loss(y_true, null_rates)
    
    total_spikes = torch.nansum(y_true)
    if not torch.is_nonzero(total_spikes):
        raise ValueError(
            "bits_per_spike: total_spikes is zero (no observed spikes). "
            "Metric is undefined; ensure your evaluation window contains spikes."
        )

    return (nll_null - nll_model) / total_spikes / math.log(2.0)

def _t_correlation(
    y1: torch.Tensor,
    y2: torch.Tensor,
    dim: t.Union[None, int, t.Tuple[int]] = -1,
    eps: float = 1e-8,
):
    if dim is None:
        dim = tuple(range(y1.dim()))
    y1 = (y1 - y1.mean(dim=dim, keepdim=True)) / (
        y1.std(dim=dim, unbiased=False, keepdim=True) + eps
    )
    y2 = (y2 - y2.mean(dim=dim, keepdim=True)) / (
        y2.std(dim=dim, unbiased=False, keepdim=True) + eps
    )
    corr = (y1 * y2).mean(dim=dim)
    return corr


def _np_correlation(
    y1: np.ndarray,
    y2: np.ndarray,
    axis: t.Union[None, int, t.Tuple[int]] = -1,
    eps: float = 1e-8,
    **kwargs,
):
    y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / (
        y1.std(axis=axis, ddof=0, keepdims=True) + eps
    )
    y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / (
        y2.std(axis=axis, ddof=0, keepdims=True) + eps
    )
    corr = (y1 * y2).mean(axis=axis, **kwargs)
    return corr


def correlation(
    y1: t.Union[torch.Tensor, np.ndarray],
    y2: t.Union[torch.Tensor, np.ndarray],
    dim: t.Union[None, int, t.Tuple[int]] = -1,
    eps: t.Union[torch.Tensor, float] = 1e-8,
    **kwargs,
):
    return (
        _t_correlation(y1=y1, y2=y2, dim=dim, eps=eps)
        if isinstance(y1, torch.Tensor)
        else _np_correlation(y1=y1, y2=y2, axis=dim, eps=eps, **kwargs)
    )


class Loss(_Loss):
    """Basic Criterion class"""

    def __init__(
        self,
        args,
        ds: t.Dict[str, DataLoader],
        size_average: bool = None,
        reduce: bool = None,
        reduction: REDUCTION = "sum",
    ):
        super(Loss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction
        )
        self.ds_scale = args.ds_scale
        self.ds_sizes = BufferDict(
            buffers={
                mouse_id: torch.tensor(len(mouse_ds.dataset), dtype=torch.float32)
                for mouse_id, mouse_ds in ds.items()
            }
        )

    def scale_ds(self, loss: torch.Tensor, mouse_id: str, batch_size: int):
        """Scale loss based on the size of the dataset"""
        if self.ds_scale:
            scale = torch.sqrt(self.ds_sizes[mouse_id] / batch_size)
            loss = scale * loss
        return loss


@register("msse")
class MSSE(Loss):
    """mean sum squared error"""

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mouse_id: str,
        reduction: REDUCTION = "sum",
        batch_size: int = None,
    ):
        if batch_size is None:
            batch_size = y_true.size(0)
        loss = msse(y_true=y_true, y_pred=y_pred, reduction=reduction)
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=batch_size)
        return loss


@register("poisson")
class PoissonLoss(Loss):
    def __init__(
        self,
        args,
        ds: t.Dict[str, DataLoader],
        eps: float = EPS,
        reduction: REDUCTION = "sum",
    ):
        super(PoissonLoss, self).__init__(args, ds=ds, reduction=reduction)
        self.register_buffer("eps", torch.tensor(eps))

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mouse_id: str,
        batch_size: int = None,
    ):
        if batch_size is None:
            batch_size = y_true.size(0)
        # add eps to targets and predictions to avoid numeric instability
        y_true, y_pred = y_true + self.eps, y_pred + self.eps
        loss = torch.sum(y_pred - y_true * torch.log(y_pred))
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=batch_size)
        return loss


@register("correlation")
class Correlation(Loss):
    """single trial correlation"""

    def __init__(self, args, ds: t.Dict[str, DataLoader], eps: float = EPS):
        super(Correlation, self).__init__(args, ds=ds)
        self.register_buffer("eps", torch.tensor(eps))

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mouse_id: str,
        batch_size: int = None,
    ):
        if batch_size is None:
            batch_size = y_true.size(0)
        num_neurons = y_true.size(1)
        corr = correlation(y1=y_true, y2=y_pred, dim=0, eps=self.eps)
        loss = num_neurons - torch.sum(corr)
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=batch_size)
        return loss


def get_criterion(args, ds: t.Dict[str, DataLoader]):
    assert args.criterion in _CRITERION, f"Criterion {args.criterion} not found."
    criterion = _CRITERION[args.criterion](args, ds=ds)
    criterion.to(args.device)
    return criterion
