"""
Provides rule-based models for converting text to its phonemic representation (IPA).

This module contains wrapper classes for different text-to-phoneme libraries,
ensuring they conform to the `ITextToPhonemModel` interface defined in
`model_interfaces`. It includes implementations for German (using Epitran) and
English (using eng-to-ipa).
"""
import epitran
import eng_to_ipa
import model_interfaces

def get_phonem_converter(language: str):
    """
    Factory function to get a phoneme converter for a given language.

    Args:
        language (str): The language code ('de' or 'en').

    Raises:
        ValueError: If the language is not implemented.

    Returns:
        ITextToPhonemModel: An instance of a phoneme converter."""
    if language == 'de':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('deu-Latn'))
    elif language == 'en':
        phonem_converter = EngPhonemConverter()
    else:
        raise ValueError('Language not implemented')

    return phonem_converter

class EpitranPhonemConverter(model_interfaces.ITextToPhonemModel):
    """A phoneme converter for German using the Epitran library."""

    def __init__(self, epitran_model) -> None:
        """
        Initializes the converter with a pre-loaded Epitran model.

        Args:
            epitran_model: An instance of an epitran.Epitran model.
        """
        super().__init__()
        self.epitran_model = epitran_model

    def convert_to_phonem(self, sentence: str) -> str:
        """
        Converts a German sentence to its IPA representation.

        Args:
            sentence (str): The input sentence in German.

        Returns:
            str: The phonemic (IPA) representation of the sentence.
        """
        phonem_representation = self.epitran_model.transliterate(sentence)
        return phonem_representation


class EngPhonemConverter(model_interfaces.ITextToPhonemModel):
    """A phoneme converter for English using the eng-to-ipa library."""

    def convert_to_phonem(self, sentence: str) -> str:
        """
        Converts an English sentence to its IPA representation.

        This method also removes asterisk characters that the library sometimes
        adds to denote stress, which are not needed for this project's comparisons.

        Args:
            sentence (str): The input sentence in English.

        Returns:
            str: The phonemic (IPA) representation of the sentence.
        """
        phonem_representation = eng_to_ipa.convert(sentence)
        phonem_representation = phonem_representation.replace('*','')
        return phonem_representation
