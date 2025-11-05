from typing import Union
import torch
from transformers import pipeline
import numpy as np
from model_interfaces import IASRModel

class WhisperASRModel(IASRModel):
    def __init__(self, model_name="openai/whisper-base"):
        self.asr = pipeline(
            "automatic-speech-recognition", model=model_name, return_timestamps="word"
        )
        self._transcript = ""
        self._word_locations = []
        self.sample_rate = 16000

    def processAudio(self, audio: Union[np.ndarray, torch.Tensor]):
        # 'audio' can be a path to a file or a numpy array of audio samples.
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
        return self._transcript

    def get_word_locations(self) -> list:

        return self._word_locations
