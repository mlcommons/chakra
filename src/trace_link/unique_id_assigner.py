from typing import Dict


class UniqueIdAssigner:
    """
    Assigns unique IDs to items, ensuring each item gets a distinct ID.

    This class is used to maintain a consistent and unique mapping of original identifiers to new unique identifiers.
    It's particularly useful in scenarios where the uniqueness of IDs across different entities or iterations needs to
    be preserved.

    Attributes:
        next_id (int): The next unique ID to be assigned.
        original_to_new_ids (Dict[int, int]): A mapping from original IDs to their corresponding new unique IDs. This
            helps in retrieving already assigned unique IDs and ensures the same original ID always maps to the same
            unique ID.
    """

    def __init__(self) -> None:
        """
        Initializes the UniqueIdAssigner with a starting ID of 0.
        """
        self.next_id: int = 0
        self.original_to_new_ids: Dict[int, int] = {}

    def assign_or_retrieve_id(self, original_id: int) -> int:
        """
        Assigns a new unique ID to the given original ID if it doesn't have one already; otherwise, returns the
        previously assigned unique ID.

        Args:
            original_id (int): The original ID for which a unique ID is needed.

        Returns:
            int: A unique ID corresponding to the original ID.
        """
        if original_id not in self.original_to_new_ids:
            self.original_to_new_ids[original_id] = self.next_id
            self.next_id += 1

        return self.original_to_new_ids[original_id]

    def generate_new_id(self) -> int:
        """
        Generates a new unique ID without needing an original ID.

        This is useful for cases where new entities are created that do not have an existing identifier.

        Returns:
            int: A new unique ID.
        """
        unique_id = self.next_id
        self.next_id += 1
        return unique_id

    def lookup_new_id(self, original_id: int) -> int:
        """
        Retrieves the new unique ID for a given original ID, if it has been assigned.

        This method is useful for checking if a unique ID has already been assigned to an original ID and retrieving it.

        Args:
            original_id (int): The original ID to look up.

        Returns:
            int: The new unique ID if it has been assigned, otherwise returns the original ID.
        """
        return self.original_to_new_ids.get(original_id, original_id)
