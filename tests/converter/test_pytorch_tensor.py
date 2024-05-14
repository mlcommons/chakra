from src.converter.pytorch_tensor import PyTorchTensor, list_to_pytorch_tensor


def test_pytorch_tensor_initialization():
    """Test initialization of PyTorchTensor object."""
    tensor_data = [1, 2, 3, 4, 5, 6]
    tensor = PyTorchTensor(tensor_data)
    assert tensor.tensor_data == tensor_data


def test_pytorch_tensor_is_valid():
    """Test the is_valid method of PyTorchTensor."""
    valid_data = [1, 2, 3, 4, 5, 6]
    invalid_data_1 = [1, 2, 3, 4, 5]  # Less than 6 elements
    invalid_data_2 = [1, 2, 3, 4, 5, 6, 7]  # More than 6 elements
    invalid_data_3 = [1, 2, 3, 4, 5, "a"]  # Non-integer element

    valid_tensor = PyTorchTensor(valid_data)
    invalid_tensor_1 = PyTorchTensor(invalid_data_1)
    invalid_tensor_2 = PyTorchTensor(invalid_data_2)
    invalid_tensor_3 = PyTorchTensor(invalid_data_3)

    assert valid_tensor.is_valid() is True
    assert invalid_tensor_1.is_valid() is False
    assert invalid_tensor_2.is_valid() is False
    assert invalid_tensor_3.is_valid() is False


def test_pytorch_tensor_properties():
    """Test property methods of PyTorchTensor."""
    tensor_data = [1, 2, 3, 4, 5, 6]
    tensor = PyTorchTensor(tensor_data)

    assert tensor.tensor_id == 1
    assert tensor.storage_id == 2
    assert tensor.offset == 3
    assert tensor.num_elem == 4
    assert tensor.elem_bytes == 5


def test_pytorch_tensor_has_valid_storage_id():
    """Test has_valid_storage_id method of PyTorchTensor."""
    valid_storage_id_data = [1, 2, 3, 4, 5, 6]
    invalid_storage_id_data = [1, 0, 3, 4, 5, 6]  # storage_id = 0

    valid_tensor = PyTorchTensor(valid_storage_id_data)
    invalid_tensor = PyTorchTensor(invalid_storage_id_data)

    assert valid_tensor.has_valid_storage_id() is True
    assert invalid_tensor.has_valid_storage_id() is False


def test_list_to_pytorch_tensor():
    """Test list_to_pytorch_tensor function."""
    tensor_data = [1, 2, 3, 4, 5, 6]
    tensor = list_to_pytorch_tensor(tensor_data)

    assert isinstance(tensor, PyTorchTensor)
    assert tensor.tensor_data == tensor_data
