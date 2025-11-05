from string import punctuation
from typing import List, Tuple
import logging
import time
import torch
import numpy as np
import epitran
import models as mo
import word_metrics
import word_matching as wm
import model_interfaces as mi
import rule_based_models

logger = logging.getLogger(__name__)


def get_trainer(language: str, asr_model: mi.IASRModel = None):
    """
    Factory function to create and configure a PronunciationTrainer instance.

    Args:
        language (str): The language code (e.g., 'de', 'en').

    Raises:
        ValueError: If the requested language is not implemented.

    Returns:
        PronunciationTrainer: A configured instance of the trainer for the
        specified language.
    """
    # If an ASR model is not provided, create a new one.
    # This allows sharing a single model instance to save memory.
    if asr_model is None:
        asr_model = mo.get_asr_model(language, use_whisper=True)

    if language == "de":
        phonem_converter = rule_based_models.EpitranPhonemConverter(
            epitran.Epitran("deu-Latn")
        )
    elif language == "en":
        phonem_converter = rule_based_models.EngPhonemConverter()
    else:
        raise ValueError("Language not implemented")

    trainer = PronunciationTrainer(asr_model, phonem_converter)

    return trainer


class PronunciationTrainer:
    current_transcript: str
    current_ipa: str

    current_recorded_audio: torch.Tensor
    current_recorded_transcript: str
    current_recorded_word_locations: list
    current_recorded_intonations: torch.tensor
    current_words_pronunciation_accuracy = []
    categories_thresholds = np.array([80, 60, 59])

    sampling_rate = 16000

    def __init__(
        self, asr_model: mi.IASRModel, word_to_ipa_coverter: mi.ITextToPhonemModel
    ) -> None:
        """
        Initializes the PronunciationTrainer.

        Args:
            asr_model (mi.IASRModel): An instance of an Automatic Speech
                Recognition model.
            word_to_ipa_coverter (mi.ITextToPhonemModel): An instance of a
                text-to-phonem converter.
        """
        self.asr_model = asr_model
        self.ipa_converter = word_to_ipa_coverter
        self.language = "de" if isinstance(word_to_ipa_coverter, rule_based_models.EpitranPhonemConverter) else "en"

    def get_transcript_and_words_locations(self, audio_length_in_samples: int):
        """
        Retrieves the transcript and word timestamps from the ASR model.

        Args:
            audio_length_in_samples (int): The total length of the audio in samples.

        Returns:
            Tuple[str, list]: A tuple containing:
                - The full text transcript.
                - A list of tuples, where each tuple represents the start and
                  end sample of a word.
        """
        audio_transcript = self.asr_model.get_transcript()
        word_locations_in_samples = self.asr_model.get_word_locations()

        fade_duration_in_samples = 0.05 * self.sampling_rate
        word_locations_in_samples = [
            (
                int(np.maximum(0, word["start_ts"] - fade_duration_in_samples)),
                int(
                    np.minimum(
                        audio_length_in_samples - 1,
                        word["end_ts"] + fade_duration_in_samples,
                    )
                ),
            )
            for word in word_locations_in_samples
        ]

        return audio_transcript, word_locations_in_samples

    def get_words_relative_intonation(self, audio: torch.tensor, word_locations: list):
        """
        Calculates the relative intonation (energy) for each word in the audio.

        It computes the root mean square (RMS) of the audio signal for each
        word's location and normalizes it by the mean intonation of the sentence.

        Args:
            audio (torch.tensor): The audio signal tensor.
            word_locations (list): A list of tuples with the start and end
                samples for each word.

        Returns:
            torch.tensor: A tensor containing the relative intonation for each word.
        """
        intonations = torch.zeros((len(word_locations), 1))
        intonation_fade_samples = 0.3 * self.sampling_rate
        logger.debug("Intonations tensor shape: %s", intonations.shape)
        for idx, location in enumerate(word_locations):
            intonation_start = int(
                np.maximum(0, location[0] - intonation_fade_samples)
            )
            intonation_end = int(
                np.minimum(
                    audio.shape[1] - 1, location[1] + intonation_fade_samples
                )
            )
            intonations[idx] = torch.sqrt(
                torch.mean(audio[0][intonation_start:intonation_end] ** 2)
            )

        intonations = intonations / torch.mean(intonations)
        return intonations

    ##################### ASR Functions ###########################

    def process_audio_for_given_text(
        self, recorded_audio: torch.Tensor = None, real_text=None
    ):
        """
        Main processing pipeline for evaluating a recorded audio against a real text.

        This method orchestrates the entire evaluation process:
        1. Transcribes the audio.
        2. Matches the transcribed words against the real text words.
        3. Calculates pronunciation accuracy based on phonemic distance.
        4. Categorizes the pronunciation of each word.

        Args:
            recorded_audio (torch.Tensor): The recorded audio to evaluate.
            real_text (str): The ground truth text to compare against.

        Returns:
            dict: A dictionary containing detailed results of the analysis,
            including transcripts, IPA conversions, accuracy scores, word
            timings, and pronunciation categories.
        """

        start = time.time()
        recording_transcript, recording_ipa, word_locations = self.get_audio_transcript(
            recorded_audio
        )
        logger.info("Time for NN to transcript audio: %s", str(time.time() - start))

        start = time.time()
        (
            real_and_transcribed_words,
            real_and_transcribed_words_ipa,
            mapped_words_indices,
        ) = self.match_sample_and_recorded_words(real_text, recording_transcript)
        logger.info("Time for matching transcripts: %s", str(time.time() - start))

        start_time, end_time = self.get_word_locations_from_record_in_seconds(
            word_locations, mapped_words_indices
        )

        pronunciation_accuracy, current_words_pronunciation_accuracy = (
            self.get_pronunciation_accuracy(real_and_transcribed_words_ipa)
        )

        pronunciation_categories = self.get_words_pronunciation_category(
            current_words_pronunciation_accuracy
        )

        result = {
            "recording_transcript": recording_transcript,
            "real_and_transcribed_words": real_and_transcribed_words,
            "recording_ipa": recording_ipa,
            "start_time": start_time,
            "end_time": end_time,
            "real_and_transcribed_words_ipa": real_and_transcribed_words_ipa,
            "pronunciation_accuracy": pronunciation_accuracy,
            "pronunciation_categories": pronunciation_categories,
        }

        return result

    def get_audio_transcript(self, recorded_audio: torch.Tensor = None):
        """
        Transcribes an audio tensor to text and phonemes.

        Processes the audio using the configured ASR model to get the text
        transcript and word locations, then converts the transcript to its
        phonemic representation (IPA).

        Args:
            recorded_audio (torch.Tensor): The audio signal to transcribe.

        Returns:
            Tuple[str, str, list]: A tuple containing the text transcript,
            the IPA transcript, and the list of word locations (timestamps).
        """
        current_recorded_audio = recorded_audio

        current_recorded_audio = self.preprocess_audio(current_recorded_audio)

        self.asr_model.processAudio(current_recorded_audio)
        self.asr_model.processAudio(current_recorded_audio, language=self.language)

        current_recorded_transcript, current_recorded_word_locations = (
            self.get_transcript_and_words_locations(current_recorded_audio.shape[1])
        )
        current_recorded_ipa = self.ipa_converter.convert_to_phonem(
            current_recorded_transcript
        )

        return (
            current_recorded_transcript,
            current_recorded_ipa,
            current_recorded_word_locations,
        )

    def get_word_locations_from_record_in_seconds(
        self, word_locations, mapped_words_indices
    ) -> Tuple[str, str]:
        """
        Converts word locations from samples to space-separated strings of seconds.

        Args:
            word_locations (list): A list of tuples with start and end times
                in samples for each word in the original recording.
            mapped_words_indices (list): A list of indices that map the real
                text words to the words found in the recording.

        Returns:
            Tuple[str, str]: A tuple containing two strings:
                - The first string is the space-separated start times.
                - The second string is the space-separated end times.
        """
        start_time = []
        end_time = []
        for mapped_idx in mapped_words_indices:
            start_time.append(float(word_locations[mapped_idx][0]) / self.sampling_rate)
            end_time.append(float(word_locations[mapped_idx][1]) / self.sampling_rate)
        return " ".join([str(time) for time in start_time]), " ".join(
            [str(time) for time in end_time]
        )

    ##################### END ASR Functions ###########################

    ##################### Evaluation Functions ###########################
    def match_sample_and_recorded_words(self, real_text, recorded_transcript):
        """
        Aligns words from a recorded transcript to a real text and converts pairs to IPA.

        Uses a word matching algorithm (DTW) to find the best alignment between
        the words in the ground truth text and the words from the ASR transcript.
        It then creates pairs of (real_word, transcribed_word) in both text and
        IPA format.

        Args:
            real_text (str): The ground truth text.
            recorded_transcript (str): The text transcribed from the audio.

        Returns:
            Tuple[list, list, list]: A tuple containing:
                - A list of (real_word, transcribed_word) tuples.
                - A list of (real_word_ipa, transcribed_word_ipa) tuples.
                - A list of indices mapping real words to the transcribed
                  word list.
        """
        words_estimated = recorded_transcript.split()

        if real_text is None:
            words_real = self.current_transcript[0].split()
        else:
            words_real = real_text.split()

        mapped_words, mapped_words_indices = wm.get_best_mapped_words(
            words_estimated, words_real
        )

        real_and_transcribed_words = []
        real_and_transcribed_words_ipa = []
        for word_idx, real_word in enumerate(words_real):
            mapped_word = mapped_words[word_idx]
            real_and_transcribed_words.append((real_word, mapped_word))
            real_and_transcribed_words_ipa.append(
                (
                    self.ipa_converter.convert_to_phonem(real_word),
                    self.ipa_converter.convert_to_phonem(mapped_word),
                )
            )
        return (
            real_and_transcribed_words,
            real_and_transcribed_words_ipa,
            mapped_words_indices,
        )

    def get_pronunciation_accuracy(
        self, real_and_transcribed_words_ipa
    ) -> Tuple[float, List[float]]:
        """
        Calculates the overall and per-word pronunciation accuracy.

        Accuracy is calculated based on the Levenshtein distance (edit distance)
        between the phonemic representations of the real word and the
        transcribed word.

        Args:
            real_and_transcribed_words_ipa (list): A list of tuples, where each
                tuple contains the IPA representation of the real word and the
                transcribed word.

        Returns:
            Tuple[float, List[float]]: A tuple containing:
                - The overall pronunciation accuracy percentage for the sentence.
                - A list of accuracy percentages for each individual word.
        """
        total_mismatches = 0.0
        number_of_phonemes = 0.0
        current_words_pronunciation_accuracy = []
        for pair in real_and_transcribed_words_ipa:

            real_without_punctuation = self.remove_punctuation(pair[0]).lower()
            number_of_word_mismatches = word_metrics.edit_distance_python(
                real_without_punctuation, self.remove_punctuation(pair[1]).lower()
            )
            total_mismatches += number_of_word_mismatches
            number_of_phonemes_in_word = len(real_without_punctuation)
            number_of_phonemes += number_of_phonemes_in_word

            current_words_pronunciation_accuracy.append(
                float(number_of_phonemes_in_word - number_of_word_mismatches)
                / number_of_phonemes_in_word
                * 100
            )

        percentage_of_correct_pronunciations = (
            (number_of_phonemes - total_mismatches) / number_of_phonemes * 100
        )

        return (
            np.round(percentage_of_correct_pronunciations),
            current_words_pronunciation_accuracy,
        )

    def remove_punctuation(self, word: str) -> str:
        """
        Removes all punctuation characters from a string.

        Args:
            word (str): The input string.

        Returns:
            str: The string with punctuation removed.
        """
        return "".join([char for char in word if char not in punctuation])

    def get_words_pronunciation_category(self, accuracies) -> list:
        """
        Categorizes each word's pronunciation based on its accuracy score.

        Args:
            accuracies (list): A list of pronunciation accuracy scores for each word.

        Returns:
            list: A list of integer categories for each word.
        """
        categories = []

        for accuracy in accuracies:
            categories.append(self.get_pronunciation_category_from_accuracy(accuracy))

        return categories

    def get_pronunciation_category_from_accuracy(self, accuracy) -> int:
        """
        Determines the category for a single accuracy score.

        Compares the accuracy score against predefined thresholds to assign a
        category (e.g., 0 for good, 1 for okay, 2 for bad).

        Args:
            accuracy (float): The pronunciation accuracy score.

        Returns:
            int: The corresponding category index.
        """
        return np.argmin(abs(self.categories_thresholds - accuracy))

    def preprocess_audio(self, audio: torch.tensor) -> torch.tensor:
        """
        Normalizes an audio tensor.

        It centers the audio by subtracting the mean and normalizes its
        amplitude to the range [-1, 1] by dividing by the maximum absolute value.

        Args:
            audio (torch.tensor): The input audio tensor.

        Returns:
            torch.tensor: The preprocessed audio tensor.
        """
        audio = audio - torch.mean(audio)
        audio = audio / torch.max(torch.abs(audio))
        return audio
