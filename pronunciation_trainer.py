from string import punctuation
from typing import List, Tuple
import time
import torch
import numpy as np
import epitran
import models as mo
import word_metrics
import word_matching as wm
import model_interfaces as mi
import rule_based_models


def get_trainer(language: str):

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
        self.asr_model = asr_model
        self.ipa_converter = word_to_ipa_coverter

    def get_transcript_and_words_locations(self, audio_length_in_samples: int):

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
        intonations = torch.zeros((len(word_locations), 1))
        intonation_fade_samples = 0.3 * self.sampling_rate
        print(intonations.shape)
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

        start = time.time()
        recording_transcript, recording_ipa, word_locations = self.get_audio_transcript(
            recorded_audio
        )
        print("Time for NN to transcript audio: ", str(time.time() - start))

        start = time.time()
        (
            real_and_transcribed_words,
            real_and_transcribed_words_ipa,
            mapped_words_indices,
        ) = self.match_sample_and_recorded_words(real_text, recording_transcript)
        print("Time for matching transcripts: ", str(time.time() - start))

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
        current_recorded_audio = recorded_audio

        current_recorded_audio = self.preprocess_audio(current_recorded_audio)

        self.asr_model.processAudio(current_recorded_audio)

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
        return "".join([char for char in word if char not in punctuation])

    def get_words_pronunciation_category(self, accuracies) -> list:
        categories = []

        for accuracy in accuracies:
            categories.append(self.get_pronunciation_category_from_accuracy(accuracy))

        return categories

    def get_pronunciation_category_from_accuracy(self, accuracy) -> int:
        return np.argmin(abs(self.categories_thresholds - accuracy))

    def preprocess_audio(self, audio: torch.tensor) -> torch.tensor:
        audio = audio - torch.mean(audio)
        audio = audio / torch.max(torch.abs(audio))
        return audio
