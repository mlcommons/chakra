import pytest

from src.trace_link.unique_id_assigner import UniqueIdAssigner


@pytest.fixture
def assigner():
    """Fixture to create a new UniqueIdAssigner instance for each test."""
    return UniqueIdAssigner()


def test_assign_or_retrieve_id_new(assigner):
    """
    Test that a new unique ID is correctly assigned to a new original ID.
    """
    first_id = assigner.assign_or_retrieve_id(10)
    assert first_id == 0  # Expect the first assigned ID to be 0


def test_assign_or_retrieve_id_existing(assigner):
    """
    Test that the same original ID retrieves the same unique ID upon subsequent calls.
    """
    first_id = assigner.assign_or_retrieve_id(10)
    second_id = assigner.assign_or_retrieve_id(10)
    assert second_id == first_id  # Ensure it retrieves the same ID


def test_assign_or_retrieve_id_distinct(assigner):
    """
    Test that different original IDs receive different unique IDs.
    """
    first_id = assigner.assign_or_retrieve_id(10)
    second_id = assigner.assign_or_retrieve_id(20)
    assert second_id != first_id
    assert second_id == 1  # This should be the next unique ID


def test_generate_new_id_sequence(assigner):
    """
    Test that generate_new_id consistently returns incrementing IDs.
    """
    ids = [assigner.generate_new_id() for _ in range(5)]
    expected_ids = list(range(5))
    assert ids == expected_ids


def test_lookup_new_id_assigned(assigner):
    """
    Test lookup of new IDs, ensuring assigned IDs return the correct new ID.
    """
    original_id = 30
    new_id = assigner.assign_or_retrieve_id(original_id)
    assert assigner.lookup_new_id(original_id) == new_id


def test_lookup_new_id_unassigned(assigner):
    """
    Test lookup for an unassigned ID returns the original ID.
    """
    unassigned_id = 40
    assert assigner.lookup_new_id(unassigned_id) == unassigned_id
