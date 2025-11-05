import abc
import numpy as np


class IASRModel(metaclass=abc.ABCMeta):
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
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'translateSentence') and
                callable(subclass.translateSentence))

    @abc.abstractmethod
    def translate_sentence(self, sentence: str) -> str:
        """Get the translation of the sentence"""
        raise NotImplementedError


class ITextToSpeechModel(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'getAudioFromSentence') and
                callable(subclass.getAudioFromSentence))

    @abc.abstractmethod
    def get_audio_from_sentence(self, sentence: str) -> np.array:
        """Get audio from sentence"""
        raise NotImplementedError


class ITextToPhonemModel(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'convertToPhonem') and
                callable(subclass.convertToPhonem))

    @abc.abstractmethod
    def convert_to_phonem(self, sentence: str) -> str:
        """Convert sentence to phonemes"""
        raise NotImplementedError
