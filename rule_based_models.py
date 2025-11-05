import epitran
import eng_to_ipa
import model_interfaces

def get_phonem_converter(language: str):
    if language == 'de':
        phonem_converter = EpitranPhonemConverter(
            epitran.Epitran('deu-Latn'))
    elif language == 'en':
        phonem_converter = EngPhonemConverter()
    else:
        raise ValueError('Language not implemented')

    return phonem_converter

class EpitranPhonemConverter(model_interfaces.ITextToPhonemModel):

    def __init__(self, epitran_model) -> None:
        super().__init__()
        self.epitran_model = epitran_model

    def convert_to_phonem(self, sentence: str) -> str:
        phonem_representation = self.epitran_model.transliterate(sentence)
        return phonem_representation


class EngPhonemConverter(model_interfaces.ITextToPhonemModel):

    def convert_to_phonem(self, sentence: str) -> str:
        phonem_representation = eng_to_ipa.convert(sentence)
        phonem_representation = phonem_representation.replace('*','')
        return phonem_representation
