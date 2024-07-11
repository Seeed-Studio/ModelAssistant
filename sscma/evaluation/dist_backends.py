# Copyright (c) OpenMMLab. All rights reserved.

import pickle
from abc import ABCMeta, abstractmethod
from typing import Any, List, Tuple, TypeVar, Union, TYPE_CHECKING
import torch
import torch.distributed as torch_dist

Tensor = TypeVar('Tensor')

class BaseDistBackend(metaclass=ABCMeta):
    """The base backend of distributed communication used by mmeval Metric."""

    @abstractmethod
    def is_initialized(self) -> bool:
        """Returns True if the distributed environment has been initialized.

        Returns:
            bool: Returns True if the distributed environment has been
            initialized, otherwise returns False.
        """

    @abstractmethod
    def rank(self) -> int:
        """Returns the rank index of the current process group.

        Returns:
            int: The rank index of the current process group.
        """

    @abstractmethod
    def world_size(self) -> int:
        """Returns the world size of the current process group.

        The `world size` is the size of the communication process group.

        Returns:
           int: The size of the current process group.
        """

    @abstractmethod
    def all_gather_object(self, obj: Any) -> List[Any]:
        """All gather the given object from the current process group and
        returns a list consisting gathered object of each process..

        Args:
            obj (any): Any pickle-able python object for all gather.

        Returns:
            list: A list of the all gathered object.
        """

    @abstractmethod
    def broadcast_object(self, obj: Any, src: int) -> Any:
        """Broadcast the given object from source process to the current
        process group.

        Args:
            obj (any): Any pickle-able python object for broadcast.
            src (int): The source rank index.

        Returns:
            any: The broadcast object.
        """

class NonDist(BaseDistBackend):
    """A dummy distributed communication for non-distributed environment."""

    @property
    def is_initialized(self) -> bool:
        """Returns False directly in a non-distributed environment."""
        return False

    @property
    def rank(self) -> int:
        """Returns 0 as the rank index in a non-distributed environment."""
        return 0

    @property
    def world_size(self) -> int:
        """Returns 1 as the world_size in a non-distributed environment."""
        return 1

    def all_gather_object(self, obj: Any) -> List[Any]:
        """Returns the list with given obj in a non-distributed environment."""
        return [obj]

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Returns the given obj directly in a non-distributed environment."""
        return obj
    


class TensorBaseDistBackend(BaseDistBackend):
    """A base backend of Tensor base distributed communication like PyTorch."""

    @abstractmethod
    def _object_to_tensor(self, obj: Any) -> Tuple[Tensor, Tensor]:
        """Convert the given object to a tensor via `pickle.dumps`.

        Args:
            obj (any): Any pickle-able python object.

        Returns:
            Tuple: A tuple of the tensor converted from the given object and
            the tensor size.
        """

    @abstractmethod
    def _tensor_to_object(self, tensor: Tensor,
                          tensor_size: Union[int, Tensor]) -> Any:
        """Convert the given Tensor to a object via `pickle.loads`.

        Args:
            tenosr (Tensor): A tensor-like data.
            tensor_size (int or Tensor): The tensor size of the given Tensor to
                be convert object.

        Returns:
            Any: The object converted from the given tensor.
        """

    @abstractmethod
    def _pad_tensor(self, tensor: Tensor,
                    max_size: Union[int, Tensor]) -> Tensor:  # yapf: disable
        """Padding the given tensor to the given size with 0.

        Args:
            tensor (Tensor): A tensor-like data to be padded.
            max_size (int or Tensor): The max tensor size that for tensor
                padding.

        Returns:
            Tensor: The padded tensor.
        """

    @abstractmethod
    def _all_gather(self, tensor: Tensor) -> List[Tensor]:
        """All gather the given tensor.

        Args:
            tensor (Tensor): The tensor for all gather.

        Returns:
            list: A list of the gathered tensor.
        """

    @abstractmethod
    def _broadcast(self, tensor: Tensor, src: int) -> Tensor:
        """Broadcast the given object from the source rank.

        Args:
            tensor (Tensor): The tensor for broadcast.
            src (int): The source rank index.

        Returns:
            Tensor: The broadcast tensor.
        """

    def all_gather_object(self, obj: Any) -> List[Any]:
        """All gather the given object from the current process group and
        returns a list consisting gathered object of each process..

        There are 3 steps to all gather a python object using Tensor
        distributed communication:

        1. Serialize picklable python object to tensor.
        2. All gather the tensor size and padding the tensor with
           the same size.
        3. All gather the padded tensor and deserialize tensor to picklable
           python object.

        Args:
            obj (any): Any pickle-able python object for all gather.

        Returns:
            list: A list of the all gathered object.
        """
        obj_tensor, obj_size_tensor = self._object_to_tensor(obj)  # type: ignore # noqa: E501 # yapf: disable

        obj_size_list = self._all_gather(obj_size_tensor)
        max_obj_size = max(obj_size_list)

        padded_obj_tensor = self._pad_tensor(obj_tensor, max_obj_size)
        padded_obj_tensor_list = self._all_gather(padded_obj_tensor)

        obj_list = []
        for padded_obj_tensor, obj_size_tensor in zip(padded_obj_tensor_list,
                                                      obj_size_list):
            obj = self._tensor_to_object(padded_obj_tensor, obj_size_tensor)
            obj_list.append(obj)
        return obj_list

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Broadcast the given object from source process to the current
        process group.

        There are 3 steps to broadcast a python object use Tensor distributed
        communication:

        1. Serialize picklable python object to tensor.
        2. Broadcast the tensor size and padding the tensor with the same size.
        3. Broadcast the padded tensor and deserialize tensor to picklable
        python object.

        Args:
            obj (any): Any pickle-able python object for broadcast.
            src (int): The source rank index.

        Returns:
            any: The broadcast object.
        """
        obj_tensor, obj_size_tensor = self._object_to_tensor(obj)  # type: ignore # noqa: E501 # yapf: disable

        broadcast_obj_size_tensor = self._broadcast(obj_size_tensor, src)

        if self.rank != src:
            obj_tensor = self._pad_tensor(obj_tensor,
                                          broadcast_obj_size_tensor)

        broadcast_obj_tensor = self._broadcast(obj_tensor, src)
        broadcast_obj = self._tensor_to_object(broadcast_obj_tensor,
                                               obj_size_tensor)

        return broadcast_obj
    



Tensor = TypeVar('Tensor', bound='torch.Tensor')
class TorchCPUDist(TensorBaseDistBackend):
    """A cpu distributed communication backend for torch.distributed."""

    def __init__(self) -> None:
        super().__init__()
        if torch is None:
            raise ImportError(f'For availability of {self.__class__.__name__},'
                              ' please install pytorch first.')
        if not torch_dist.is_available():
            raise RuntimeError(
                f'For availability of {self.__class__.__name__},'
                ' make sure torch.distributed is available.')

    @property
    def is_initialized(self) -> bool:
        """Returns True if the distributed environment has been initialized.

        Returns:
            bool: Returns True if the distributed environment has been
            initialized, otherwise returns False.
        """
        return torch_dist.is_initialized()

    @property
    def rank(self) -> int:
        """Returns the rank index of the current process group."""
        return torch_dist.get_rank()

    @property
    def world_size(self) -> int:
        """Returns the world size of the current process group."""
        return torch_dist.get_world_size()

    def _object_to_tensor(self, obj: Any) -> Tuple[Tensor, Tensor]:
        """Convert the given object to a tensor via `pickle.dumps`.

        Args:
            obj (any): Any pickle-able python object.

        Returns:
            Tuple: A tuple of the tensor converted from given object and the
            tensor size.
        """
        buffer = pickle.dumps(obj)
        byte_storage = torch.ByteStorage.from_buffer(buffer)
        obj_tensor = torch.ByteTensor(byte_storage)
        obj_size_tensor = torch.LongTensor([obj_tensor.numel()])
        return obj_tensor, obj_size_tensor

    def _tensor_to_object(self, tensor: Tensor,
                          tensor_size: Union[int, Tensor]) -> Any:
        """Convert the given Tensor to a object via `pickle.loads`.

        Args:
            tenosr (Tensor): A tensor-like data.
            tensor_size (int or Tensor): The tensor size of the given Tensor to
                be convert object.

        Returns:
            Any: The object converted from the given tensor.
        """
        buffer = tensor.numpy().tobytes()[:tensor_size]
        obj = pickle.loads(buffer)
        return obj

    def _pad_tensor(self, tensor: Tensor,
                    max_size: Union[int, Tensor]) -> Tensor:  # yapf: disable
        """Padding the given tensor to the given size.

        Args:
            tensor (Tensor): A tensor-like data to be padded.
            max_size (int or Tensor): The max tensor size that for tensor
                padding.

        Returns:
            Tensor: The padded tensor.
        """
        # We use the `resize_` to pad tensor just like
        # `torch.distributed.all_gather_object`.
        return tensor.resize_(int(max_size))

    def _all_gather(self, tensor: Tensor) -> List[Tensor]:
        """All gather the given tensor.

        Args:
            tensor (Tensor): The tensor for all gather.

        Returns:
            list: A list of the gathered tensor.
        """
        tensor_list = [
            torch.empty_like(tensor).to(tensor.device)
            for _ in range(self.world_size)
        ]
        torch_dist.all_gather(tensor_list, tensor)
        return tensor_list

    def _broadcast(self, tensor: Tensor, src: int = 0) -> Tensor:
        """Broadcast the given object from the source rank.

        Args:
            tensor (Tensor): The tensor for broadcast.
            src (int): The source rank index.

        Returns:
            Tensor: The broadcast tensor.
        """
        torch_dist.broadcast(tensor, src=src)
        return tensor
    


class TorchCUDADist(TorchCPUDist):
    """A cuda distributed communication backend for torch.distributed."""

    def __init__(self) -> None:
        super().__init__()
        if torch is None:
            raise ImportError(f'For availability of {self.__class__.__name__},'
                              ' please install pytorch first.')
        if not torch.distributed.is_nccl_available():
            raise RuntimeError(
                f'For availability of {self.__class__.__name__},'
                ' make sure torch.distributed.is_nccl_available().')

    def _object_to_tensor(self, obj: Any) -> Tuple[Tensor, Tensor]:
        """Convert the given object to a cuda tensor via `pickle.dumps`.

        Args:
            obj (any): Any pickle-able python object.

        Returns:
            tuple: A tuple of the tensor converted from given object and the
            tensor size.
        """
        # Add type annotation make mypy happy
        obj_tensor: Tensor
        obj_size_tensor: Tensor
        obj_tensor, obj_size_tensor = super()._object_to_tensor(obj)
        return obj_tensor.cuda(), obj_size_tensor.cuda()

    def _tensor_to_object(self, tensor: Tensor,
                          tensor_size: Union[int, Tensor]) -> Any:
        """Convert the given cuda tensor to a object via `pickle.loads`.

        Args:
            tenosr (Tensor): A cuda tensor.
            tensor_size (int or Tensor): The tensor size of the given Tensor to
            be convert object.

        Returns:
            Any: The object converted from the given cuda tensor.
        """
        return super()._tensor_to_object(tensor.detach().cpu(), tensor_size)
    
