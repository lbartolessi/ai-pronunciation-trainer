from string import punctuation
from typing import List, Tuple
import numpy as np
from dtwalign import dtw_from_distance_matrix
import word_metrics


def get_word_distance_matrix(words_estimated: list, words_real: list) -> np.ndarray:
    """
    Calculates a distance matrix between two lists of words.

    The distance is computed using the Levenshtein distance (edit distance) between
    the phonemic representations of the words. An extra row is added to handle
    cases where a real word is not matched by any estimated word (a deletion).

    Args:
        words_estimated: A list of words from the ASR transcript.
        words_real: A list of words from the ground truth text.

    Returns:
        A numpy array representing the distance matrix. The shape is
        (len(words_estimated) + 1, len(words_real)).
    """
    number_of_real_words = len(words_real)
    number_of_estimated_words = len(words_estimated)

    word_distance_matrix = np.zeros(
        (number_of_estimated_words + 1, number_of_real_words)
    )
    for idx_estimated in range(number_of_estimated_words):
        for idx_real in range(number_of_real_words):
            word_distance_matrix[idx_estimated, idx_real] = (
                word_metrics.edit_distance_python(
                    words_estimated[idx_estimated], words_real[idx_real]
                )
            )

    # Cost of deleting a real word (matching it with a blank)
    for idx_real in range(number_of_real_words):
        word_distance_matrix[number_of_estimated_words, idx_real] = len(
            words_real[idx_real]
        )

    return word_distance_matrix


def _find_best_match_for_word(
    candidate_indices: np.ndarray, words_estimated: list, real_word: str
) -> Tuple[str, int]:
    """
    Finds the best matching estimated word from a list of candidates.

    If there's only one candidate, it's returned directly. If there are multiple,
    the one with the minimum edit distance to the real word is chosen.

    Args:
        candidate_indices: Indices of candidate words in the `words_estimated` list.
        words_estimated: The full list of words from the ASR transcript.
        real_word: The ground truth word to match against.

    Returns:
        A tuple containing the best matching word and its index.
    """
    if len(candidate_indices) == 1:
        best_idx = candidate_indices[0]
        return words_estimated[best_idx], best_idx

    # If multiple estimated words map to the same real word,
    # choose the one with the minimum edit distance.
    best_idx = -1
    min_error = float("inf")

    for idx in candidate_indices:
        error = word_metrics.edit_distance_python(words_estimated[idx], real_word)
        if error < min_error:
            min_error = error
            best_idx = idx

    return words_estimated[best_idx], best_idx


def get_resulting_string(
    mapped_indices: np.ndarray, words_estimated: list, words_real: list
) -> Tuple[List, List]:
    """
    Constructs the final list of matched words based on the DTW alignment path.

    For each word in the real text, it finds the corresponding word from the
    estimated text based on the alignment. If a real word has no match, a
    placeholder is used.

    Args:
        mapped_indices: The warping path from the DTW alignment.
        words_estimated: The list of words from the ASR transcript.
        words_real: The list of words from the ground truth text.

    Returns:
        A tuple containing:
        - mapped_words: A list of estimated words, aligned to the real words.
        - mapped_words_indices: A list of indices from the estimated words list.
    """
    mapped_words = []
    mapped_words_indices = []
    word_not_found_token = "-"

    for idx_real, real_word in enumerate(words_real):
        # Find all estimated words that mapped to the current real word
        candidate_indices = np.nonzero(mapped_indices == idx_real)[0].astype(int)

        # Filter out indices that point to the "blank" row
        candidate_indices = [
            idx for idx in candidate_indices if idx < len(words_estimated)
        ]

        if not candidate_indices:
            mapped_words.append(word_not_found_token)
            mapped_words_indices.append(-1)
        else:
            best_word, best_idx = _find_best_match_for_word(
                candidate_indices, words_estimated, real_word
            )
            mapped_words.append(best_word)
            mapped_words_indices.append(best_idx)

    return mapped_words, mapped_words_indices


def get_best_word_mapping(words_estimated: list, words_real: list) -> Tuple[List, List]:
    """
    Aligns a list of estimated words to a list of real words using DTW.

    This function computes a distance matrix between the two lists of words and
    then uses Dynamic Time Warping (DTW) to find the optimal alignment. This
    alignment maps each word in the real text to the most likely word in the
    ASR transcript.

    Args:
        words_estimated: A list of words from the ASR transcript.
        words_real: A list of words from the ground truth text.

    Returns:
        A tuple containing:
        - mapped_words: A list of estimated words, aligned to the real words.
                        Unmatched words are represented by a placeholder.
        - mapped_words_indices: A list of indices from the estimated words list
                                corresponding to the alignment.
    """
    word_distance_matrix = get_word_distance_matrix(words_estimated, words_real)

    # We transpose the matrix because dtwalign expects (query, reference)
    # and we want to align the estimated words (query) to the real words (reference).
    alignment = dtw_from_distance_matrix(word_distance_matrix.T)
    mapped_indices = alignment.get_warping_path()

    mapped_words, mapped_words_indices = get_resulting_string(
        mapped_indices, words_estimated, words_real
    )

    return mapped_words, mapped_words_indices


get_best_mapped_words = get_best_word_mapping


def get_which_letters_were_transcribed_correctly(
    real_word: str, transcribed_word: str
) -> List[int]:
    """
    Compares two words letter by letter to see which were transcribed correctly.

    NOTE: This is a naive implementation that assumes a 1-to-1 letter mapping.
    It does not handle insertions or deletions within the word.
    """
    is_letter_correct = [0] * len(real_word)
    for idx, letter in enumerate(real_word):
        if idx < len(transcribed_word) and (
            letter.lower() == transcribed_word[idx].lower() or letter in punctuation
        ):
            is_letter_correct[idx] = 1
    return is_letter_correct
