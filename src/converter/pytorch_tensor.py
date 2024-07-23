from typing import List


class PyTorchTensor:
    """
    Represents a tensor with its associated properties.

    Attributes:
        tensor_data (List[int]): Data of the tensor including tensor_id, storage_id, offset, number of elements, and
            size of each element in bytes.

    Note:
        For more details on the tensor data structure, refer to:
        https://github.com/pytorch/pytorch/blob/7cd48df2dae7e2194438b162968c47d1f05bf20e/torch/csrc/profiler/
        standalone/execution_trace_observer.cpp#L400
    """

    def __init__(self, tensor_data: List[int]) -> None:
        """
        Initialize a PyTorchTensor object with the provided tensor data.

        Args:
            tensor_data (List[int]): Data of the tensor including tensor_id, storage_id, offset, number of elements,
                and size of each element in bytes.
        """
        self.tensor_data = tensor_data

    def is_valid(self) -> bool:
        """
        Check if the tensor data is valid.

        Returns
            bool: True if tensor_data is a list of exactly six integers, False otherwise.
        """
        return (
            isinstance(self.tensor_data, list)
            and len(self.tensor_data) == 6
            and all(isinstance(item, int) for item in self.tensor_data)
        )

    @property
    def tensor_id(self) -> int:
        """
        Return the tensor ID.

        Returns
            int: Tensor ID.
        """
        return self.tensor_data[0]

    @property
    def storage_id(self) -> int:
        """
        Return the storage ID.

        Returns
            int: Storage ID.
        """
        return self.tensor_data[1]

    @property
    def offset(self) -> int:
        """
        Return the offset.

        Returns
            int: Offset value.
        """
        return self.tensor_data[2]

    @property
    def num_elem(self) -> int:
        """
        Return the number of elements in the tensor.

        Returns
            int: Number of elements.
        """
        return self.tensor_data[3]

    @property
    def elem_bytes(self) -> int:
        """
        Return the size of each element in bytes.

        Returns
            int: Size of each element in bytes.
        """
        return self.tensor_data[4]

    def has_valid_storage_id(self) -> bool:
        """
        Check if the tensor has a valid storage ID.

        Returns
            bool: True if the storage ID is greater than 0, False otherwise.
        """
        return self.storage_id > 0


def list_to_pytorch_tensor(tensor_list: List[int]) -> PyTorchTensor:
    """
    Convert a list representation of a tensor into a PyTorchTensor object.

    Args:
        tensor_list (List[int]): Data representing a tensor, including tensor_id, storage_id, offset, num_elem,
            elem_bytes.

    Returns:
        PyTorchTensor: The PyTorchTensor object created from the data.
    """
    return PyTorchTensor(tensor_list)
