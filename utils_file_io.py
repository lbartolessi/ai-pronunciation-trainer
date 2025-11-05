"""
Provides utility functions for file and I/O operations.

This module contains helper functions that are not specific to any particular
domain of the application but provide common, reusable functionality.
"""
import string
import random


def generate_random_string(str_length: int = 20):
    """
    Generates a random string of a given length using lowercase letters.

    Args:
        str_length (int, optional): The desired length of the string. Defaults to 20.

    Returns:
        str: A randomly generated string.
    """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(str_length))
