"""
A wrapper for the OpenAI Whisper ASR model.

This module provides a class that implements the `IASRModel` interface, using
the Hugging Face `transformers` pipeline for the Whisper model. It handles
processing audio and extracting both the full transcript and word-level timestamps.
"""
from typing import Union
import torch
from transformers import pipeline
import numpy as np
from model_interfaces import IASRModel

class WhisperASRModel(IASRModel):
    """
    An ASR model implementation that uses the OpenAI Whisper model.

    This class encapsulates the logic for loading a Whisper model via the
    Hugging Face pipeline and using it to transcribe audio. It conforms to the
    `IASRModel` interface, making it interchangeable with other ASR implementations.
    """
    def __init__(self, model_name="openai/whisper-base"):
        """Initializes the Whisper pipeline using the 'tiny' model for lower memory usage."""
        self.asr = pipeline(
            "automatic-speech-recognition", model="openai/whisper-tiny", return_timestamps="word"
        )
        self._transcript = ""
        self._word_locations = []
        self.sample_rate = 16000

    def processAudio(self, audio: Union[np.ndarray, torch.Tensor]):
        """
        Processes an audio array and generates a transcript with word timestamps.

        The results (transcript and word locations) are stored internally.

        Args:
            audio (Union[np.ndarray, torch.Tensor]): The audio data to process.
                Expected to be a 1D or 2D array/tensor of audio samples.
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        result = self.asr(audio[0])
        self._transcript = result["text"]
        self._word_locations = [
            {
                "word": word_info["text"],
                "start_ts": (
                    word_info["timestamp"][0] * self.sample_rate
                    if word_info["timestamp"][0] is not None
                    else None
                ),
                "end_ts": (
                    word_info["timestamp"][1] * self.sample_rate
                    if word_info["timestamp"][1] is not None
                    else (word_info["timestamp"][0] + 1) * self.sample_rate
                ),
                "tag": "processed",
            }
            for word_info in result["chunks"]
        ]

    def get_transcript(self) -> str:
        """
        Returns the full text transcript of the last processed audio.

        Returns:
            str: The transcript.
        """
        return self._transcript

    def get_word_locations(self) -> list:
        """
        Returns the word-level timestamps of the last processed audio.
        """
        return self._word_locations
