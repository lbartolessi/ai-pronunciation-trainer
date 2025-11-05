"""
Defines abstract base classes (interfaces) for various AI model types.

This module provides a set of contracts that concrete model implementations must
adhere to. Using these interfaces allows the application to be decoupled from
specific model libraries (like Whisper, Silero, etc.), making it easier to swap
or add new models in the future.
"""
import abc
import numpy as np


class IASRModel(metaclass=abc.ABCMeta):
    """
    Interface for an Automatic Speech Recognition (ASR) model.
    Defines the contract for processing audio and retrieving transcripts.
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'getTranscript') and
                callable(subclass.getTranscript) and
                hasattr(subclass, 'getWordLocations') and
                callable(subclass.getWordLocations) and
                hasattr(subclass, 'processAudio') and
                callable(subclass.processAudio))

    @abc.abstractmethod
    def get_transcript(self) -> str:
        """Get the transcripts of the process audio"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_word_locations(self) -> list:
        """Get the pair of words location from audio"""
        raise NotImplementedError

    @abc.abstractmethod
    def processAudio(self, audio):
        """Process the audio"""
        raise NotImplementedError


class ITranslationModel(metaclass=abc.ABCMeta):
    """
    Interface for a text translation model.
    Defines the contract for translating a sentence from one language to another.
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'translateSentence') and
                callable(subclass.translateSentence))

    @abc.abstractmethod
    def translate_sentence(self, sentence: str) -> str:
        """Get the translation of the sentence"""
        raise NotImplementedError


class ITextToSpeechModel(metaclass=abc.ABCMeta):
    """
    Interface for a Text-to-Speech (TTS) model.
    Defines the contract for converting a sentence into an audio array.
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'getAudioFromSentence') and
                callable(subclass.getAudioFromSentence))

    @abc.abstractmethod
    def get_audio_from_sentence(self, sentence: str) -> np.array:
        """Get audio from sentence"""
        raise NotImplementedError


class ITextToPhonemModel(metaclass=abc.ABCMeta):
    """
    Interface for a Text-to-Phoneme conversion model.
    Defines the contract for converting a sentence into its IPA representation.
    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'convertToPhonem') and
                callable(subclass.convertToPhonem))

    @abc.abstractmethod
    def convert_to_phonem(self, sentence: str) -> str:
        """Convert sentence to phonemes"""
        raise NotImplementedError
