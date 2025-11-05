"""
Provides concrete implementations of model interfaces using neural networks.

This module contains classes that wrap neural network models (like those from
Silero or Hugging Face) to make them conform to the abstract base classes
defined in `model_interfaces`.
"""
import torch
import model_interfaces

class NeuralASR(model_interfaces.IASRModel):
    """
    An ASR model implementation using Silero's STT models.

    This class processes audio tensors and uses a CTC (Connectionist Temporal
    Classification) decoder to produce a text transcript and word timestamps.
    """
    word_locations_in_samples = None
    audio_transcript = None

    def __init__(self, model: torch.nn.Module, decoder) -> None:
        """Initializes the NeuralASR model."""
        super().__init__()
        self.model = model
        self.decoder = decoder  # Decoder from CTC-outputs to transcripts

    def get_transcript(self) -> str:
        """Get the transcripts of the process audio"""
        assert self.audio_transcript is not None, (
            'Can get audio transcripts without having processed the audio')
        return self.audio_transcript

    def get_word_locations(self) -> list:
        """Get the pair of words location from audio"""
        assert self.word_locations_in_samples is not None, (
            'Can get word locations without having processed the audio'
        )
        return self.word_locations_in_samples

    def processAudio(self, audio: torch.Tensor):
        """Process the audio"""
        audio_length_in_samples = audio.shape[1]
        with torch.inference_mode():
            nn_output = self.model(audio)

            self.audio_transcript, self.word_locations_in_samples = self.decoder(
                nn_output[0, :, :].detach(), audio_length_in_samples, word_align=True)


class NeuralTranslator(model_interfaces.ITranslationModel):
    """
    A translation model implementation using models from Hugging Face.

    This class uses a tokenizer and a sequence-to-sequence model (like Helsinki-NLP)
    to translate a given sentence.
    """
    def __init__(self, model: torch.nn.Module, tokenizer) -> None:
        """Initializes the NeuralTranslator model."""
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def translate_sentence(self, sentence: str) -> str:
        """Translates a sentence using the configured model."""
        tokenized_text = self.tokenizer(sentence, return_tensors='pt')
        translation = self.model.generate(**tokenized_text)
        translated_text = self.tokenizer.batch_decode(
            translation, skip_special_tokens=True)[0]

        return translated_text
