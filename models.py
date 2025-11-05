import pickle
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
