import torch
import typing as t
import numpy as np
from copy import deepcopy
from scipy.stats import pearsonr
from scipy.special import gammaln
from torch.utils.data import DataLoader


from v1t import losses


class Metrics:
    """
    Metric class to compute metrics used in the Sensorium challenge

    Code reference: https://github.com/sinzlab/sensorium/blob/e5017df2ff89c60a4d0a7687c4bde67774de346b/sensorium/utility/metrics.py
    """

    def __init__(self, ds: DataLoader, results: t.Dict[str, torch.Tensor]):
        """
        Computes performance metrics of neural response predictions.
        """
        self.repeat_image = ds.dataset.tier == "test"
        self.hashed = ds.dataset.hashed
        self.targets = results["targets"].numpy()
        self.predictions = results["predictions"].numpy()
        self.image_ids = results["image_ids"].numpy()
        self.neuron_ids = deepcopy(ds.dataset.neuron_ids)
        self.trial_ids = results["trial_ids"]
        if not self.hashed:
            self.trial_ids = self.trial_ids.numpy()
            self.order()

    def order(self):
        """Re-order the responses based on trial IDs and neuron IDs."""
        trial_ids = np.argsort(self.trial_ids)
        neuron_ids = np.argsort(self.neuron_ids)

        self.targets = self.targets[trial_ids, :][:, neuron_ids]
        self.predictions = self.predictions[trial_ids, :][:, neuron_ids]
        self.image_ids = self.image_ids[trial_ids]
        self.neuron_ids = self.neuron_ids[neuron_ids]
        self.trial_ids = trial_ids

    def split_responses(
        self,
    ) -> t.Tuple[t.List[np.ndarray], t.List[np.ndarray]]:
        """
        Split the responses (or predictions) array based on image ids.
        Each element of the list contains the responses to repeated
        presentations of a single image.
        Returns:
            targets: t.List[np.ndarray]: a list of array where each tensor
                is the target responses from repeated images.
            predictions: t.List[np.ndarray]: a list of array where each tensor
                is the predicted responses from repeated images.
        """
        repeat_targets, repeat_predictions = [], []
        for image_id in np.unique(self.image_ids):
            indexes = self.image_ids == image_id
            repeat_targets.append(self.targets[indexes])
            repeat_predictions.append(self.predictions[indexes])
        return repeat_targets, repeat_predictions

    def single_trial_correlation(self, per_neuron: bool = False):
        """
        Compute single-trial correlation.
        Returns:
            corr: t.Union[float, np.ndarray], single trial correlation
        """
        corr = losses.correlation(y1=self.predictions, y2=self.targets, dim=0)
        return corr if per_neuron else corr.mean()

    def correlation_to_average(self, per_neuron: bool = False):
        """
        Compute correlation to average response across repeats.
        Returns:
            np.array or float: Correlation (average across repeats) between responses and predictions
        """
        if not self.repeat_image or self.hashed:
            return None
        mean_responses, mean_predictions = [], []
        for repeat_responses, repeat_predictions in zip(*self.split_responses()):
            mean_responses.append(repeat_responses.mean(axis=0, keepdims=True))
            mean_predictions.append(repeat_predictions.mean(axis=0, keepdims=True))
        mean_responses = np.vstack(mean_responses)
        mean_predictions = np.vstack(mean_predictions)
        corr = losses.correlation(y1=mean_responses, y2=mean_predictions, dim=0)
        return corr if per_neuron else corr.mean()

    def _fev(
        self,
        targets: t.List[np.ndarray],
        predictions: t.List[np.ndarray],
        return_exp_var: bool = False,
    ):
        """
        Compute the fraction of explainable variance explained per neuron
        Args:
            targets (array-like): Neuronal neuron responses (ground truth) to
                image repeats. Dimensions: [num_images] np.array(num_repeats, num_neurons)
            outputs (array-like): Model predictions to the repeated images,
                with an identical shape as the targets
            return_exp_var (bool): returns the fraction of explainable
                variance per neuron if set to True
        Returns:
            FEVe (np.array): the fraction of explainable variance explained per neuron
            --- optional: FEV (np.array): the fraction
        """
        img_var = []
        pred_var = []
        for target, prediction in zip(targets, predictions):
            pred_var.append((target - prediction) ** 2)
            img_var.append(np.var(target, axis=0, ddof=1))
        pred_var = np.vstack(pred_var)
        img_var = np.vstack(img_var)

        total_var = np.var(np.vstack(targets), axis=0, ddof=1)
        noise_var = np.mean(img_var, axis=0)
        fev = (total_var - noise_var) / total_var

        pred_var = np.mean(pred_var, axis=0)
        fev_e = 1 - (pred_var - noise_var) / (total_var - noise_var)
        return [fev, fev_e] if return_exp_var else fev_e

    def feve(self, per_neuron: bool = False, fev_threshold: float = 0.15):
        """
        Compute fraction of explainable variance explained
        Returns:
            fevl_val: t.Union[float, np.ndarray], FEVE value
        """
        if not self.repeat_image or self.hashed:
            return None
        repeat_targets, repeat_predictions = self.split_responses()
        fev_val, feve_val = self._fev(
            targets=repeat_targets,
            predictions=repeat_predictions,
            return_exp_var=True,
        )
        # ignore neurons below FEV threshold
        feve_val = feve_val[fev_val >= fev_threshold]
        return feve_val if per_neuron else feve_val.mean()

    def normalized_correlation(self):
        """Normalized correlation

        Reference:
        - https://www.frontiersin.org/articles/10.3389/fncom.2016.00010/full
        """
        if not self.repeat_image or self.hashed:
            return None
        cc_norm = []
        for repeated_response, repeated_prediction in zip(*self.split_responses()):
            mean_response = np.mean(repeated_response, axis=0)
            mean_prediction = np.mean(repeated_prediction, axis=0)
            cc_abs, _ = pearsonr(mean_response, mean_prediction)
            n = len(repeated_response)
            cc_max = np.sqrt(
                (
                    n * np.var(mean_response, ddof=1)
                    - np.mean(np.var(repeated_response, axis=0, ddof=1))
                )
                / ((n - 1) * np.var(mean_response, ddof=1))
            )
            cc_norm.append(cc_abs / cc_max)
        return np.mean(cc_norm)
    


    def neg_log_likelihood(self, zero_warning=True):
        """Calculates Poisson negative log likelihood given rates and spikes.
        formula: -log(e^(-r) / n! * r^n)
            = r - n*log(r) + log(n!)

        Args:
            zero_warning : bool, optional: Whether to print out warning about 0 rate predictions or not

        Returns:
            float: Total negative log-likelihood of the data
        """
        rates, spikes = self.predictions, self.targets # rate predictions, true spike counts

        assert (
            spikes.shape == rates.shape
        ), f"neg_log_likelihood: Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

        if np.any(np.isnan(spikes)):
            mask = np.isnan(spikes)
            rates = rates[~mask]
            spikes = spikes[~mask]

        assert not np.any(np.isnan(rates)), "neg_log_likelihood: NaN rate predictions found"

        assert np.all(rates >= 0), "neg_log_likelihood: Negative rate predictions found"
        if np.any(rates == 0):
            # if zero_warning:
            #     logger.warning(
            #         "neg_log_likelihood: Zero rate predictions found. Replacing zeros with 1e-9"
            #     )
            rates[rates == 0] = 1e-9

        result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
        return np.sum(result)


    def bits_per_spike(self):
        """Co-smoothing metric 

        Code reference: https://github.com/neurallatents/nlb_tools/blob/1ddc15f45b56388ff093d1396b7b87b36fa32a68/nlb_tools/evaluation.py#L252
        
        Computes bits per spike of rate predictions given spikes.
        Bits per spike is equal to the difference between the log-likelihoods (in base 2)
        of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
        divided by the total number of spikes.

        Returns:
            float: Bits per spike of rate predictions
        """
        rates, spikes = self.predictions, self.targets # rate predictions, true spike counts
        nll_model = self.neg_log_likelihood(rates, spikes)
        null_rates = np.tile(
            np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
            spikes.shape[:-1] + (1,),
        )
        nll_null = self.neg_log_likelihood(null_rates, spikes, zero_warning=False)
        return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)
