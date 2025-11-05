"""
Factory functions for creating and loading different AI models.

This module provides a centralized way to instantiate models for Automatic Speech
Recognition (ASR), Text-to-Speech (TTS), and translation. It abstracts away the
details of which specific library or pre-trained model to use for a given language.
"""
import pickle # TODO: This is flagged for removal in the roadmap.
import torch
from torch import nn
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from model_interfaces import IASRModel
from whisper_wrapper import WhisperASRModel
from ai_models import NeuralASR

LANG_NO_IMPLEMENTED = "Language not implemented"
SILERO_MODELS = "snakers4/silero-models"


def get_asr_model(language: str, use_whisper: bool = True) -> IASRModel:
    """
    Gets an instance of an ASR model for the specified language.

    This factory function can return either a Whisper-based model or a Silero-based
    model depending on the parameters.

    Args:
        language (str): The language code (e.g., 'de', 'en').
        use_whisper (bool, optional): If True, returns a Whisper model.
                                      Otherwise, returns a Silero model.
                                      Defaults to True.

    Returns:
        IASRModel: An object conforming to the ASR model interface.
    """
    if use_whisper:
        return WhisperASRModel()

    if language == "de":

        model, decoder, _ = torch.hub.load(
            repo_or_dir=SILERO_MODELS,
            model="silero_stt",
            language="de",
            device=torch.device("cpu"),
        )
        model.eval()
        return NeuralASR(model, decoder)

    elif language == "en":
        model, decoder, _ = torch.hub.load(
            repo_or_dir=SILERO_MODELS,
            model="silero_stt",
            language="en",
            device=torch.device("cpu"),
        )
        model.eval()
        return NeuralASR(model, decoder)
    elif language == "fr":
        model, decoder, _ = torch.hub.load(
            repo_or_dir=SILERO_MODELS,
            model="silero_stt",
            language="fr",
            device=torch.device("cpu"),
        )
        model.eval()
        return NeuralASR(model, decoder)
    else:
        raise ValueError(LANG_NO_IMPLEMENTED)


def get_tts_model(language: str) -> nn.Module:
    """
    Gets an instance of a TTS model for the specified language.

    This factory function loads a pre-trained Silero TTS model based on the
    language.

    Args:
        language (str): The language code (e.g., 'de', 'en').

    Returns:
        nn.Module: A PyTorch module capable of synthesizing speech from text.
                   Note: This should ideally return an ITextToSpeechModel instance.
    """
    if language == "de":

        speaker = "thorsten_v2"  # 16 kHz
        model, _ = torch.hub.load(
            repo_or_dir=SILERO_MODELS,
            model="silero_tts",
            language=language,
            speaker=speaker,
        )

    elif language == "en":
        speaker = "lj_16khz"  # 16 kHz
        model = torch.hub.load(
            repo_or_dir=SILERO_MODELS,
            model="silero_tts",
            language=language,
            speaker=speaker,
        )
    else:
        raise ValueError(LANG_NO_IMPLEMENTED)

    return model


def get_translation_model(language: str) -> nn.Module:
    """
    Gets an instance of a translation model for the specified language.

    This factory function loads a pre-trained sequence-to-sequence model from
    Hugging Face (e.g., Helsinki-NLP) for translation.

    Note:
        The use of pickle to cache models is outdated and planned for removal,
        as libraries like `transformers` handle caching automatically.

    Args:
        language (str): The language code (e.g., 'de').

    Returns:
        Tuple[nn.Module, AutoTokenizer]: A tuple containing the model and its tokenizer.
    """
    if language == "de":
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        # Cache models to avoid Hugging face processing
        with open("translation_model_de.pickle", "wb") as handle:
            pickle.dump(model, handle)
        with open("translation_tokenizer_de.pickle", "wb") as handle:
            pickle.dump(tokenizer, handle)
    else:
        raise ValueError(LANG_NO_IMPLEMENTED)

    return model, tokenizer
