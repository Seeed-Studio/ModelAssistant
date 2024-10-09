from abc import ABCMeta, abstractmethod
import torch


class BaseInfer(metaclass=ABCMeta):
    """Base class for inference.

    Args:
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(
        self,
        weights="sscma.pt",
        device=torch.device("cpu"),
    ):
        self.weights = weights
        self.device = device

    @abstractmethod
    def infer(self, input_data):
        """Forward function for inference.

        Args:
            input_data: The input data for the forward function.

        Returns:
            ForwardResults: The output of the forward function.
        """
        pass

    @abstractmethod
    def load_weights(self):
        """Load weights for inference."""
        pass
