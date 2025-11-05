import string
import random


def generate_random_string(str_length: int = 20):

    # printing lowercase
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(str_length))
