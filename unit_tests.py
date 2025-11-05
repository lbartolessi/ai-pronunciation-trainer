import unittest
import json
import epitran
import model_interfaces
import lambda_get_sample
import rule_based_models
import pronunciation_trainer


def test_category(category: int, threshold_min: int, threshold_max: int):
    event = {"body": json.dumps({"category": category, "language": "de"})}
    for _ in range(1000):
        response = lambda_get_sample.lambda_handler(event)
        response_dict = json.loads(response)
        number_of_words = len(response_dict["real_transcript"][0].split())
        length_valid = (
            number_of_words > threshold_min and number_of_words <= threshold_max
        )
        if not length_valid:
            print(
                "Category ", category, " had a sentence with length ", number_of_words
            )
            return False
    return True


class TestDataset(unittest.TestCase):

    def test_random_sentences(self):

        self.assertFalse(test_category(0, 0, 8))

    def test_easy_sentences(self):

        self.assertTrue(test_category(1, 0, 8))

    def test_normal_sentences(self):
        self.assertTrue(test_category(2, 8, 20))

    def test_hard_sentences(self):
        self.assertTrue(test_category(3, 20, 10000))


def check_phonem_converter(
    converter: model_interfaces.ITextToPhonemModel,
    input_text: str,
    expected_output: str,
):
    output = converter.convert_to_phonem(input_text)

    is_correct = output == expected_output
    if not is_correct:
        print(
            'Conversion from "',
            input_text,
            '" should be "',
            expected_output,
            '", but was "',
            output,
            '"',
        )
    return is_correct


class TestPhonemConverter(unittest.TestCase):

    def test_english(self):
        phonem_converter = rule_based_models.EngPhonemConverter()
        self.assertTrue(
            check_phonem_converter(
                phonem_converter, "Hello, this is a test", "hɛˈloʊ, ðɪs ɪz ə tɛst"
            )
        )

    def test_german(self):
        phonem_converter = rule_based_models.EpitranPhonemConverter(
            epitran.Epitran("deu-Latn")
        )

        self.assertTrue(
            check_phonem_converter(
                phonem_converter, "Hallo, das ist ein Test", "haloː, dɑːs ɪst ain tɛst"
            )
        )


trainer_SST_lambda = {}
trainer_SST_lambda["de"] = pronunciation_trainer.get_trainer("de")


class TestScore(unittest.TestCase):

    def test_exact_transcription(self):
        words_real = "Ich habe sehr viel glück, am leben und gesund zu sein"

        real_and_transcribed_words, _, _ = trainer_SST_lambda[
            "de"
        ].match_sample_and_recorded_words(words_real, words_real)

        pronunciation_accuracy, _ = trainer_SST_lambda["de"].get_pronunciation_accuracy(
            real_and_transcribed_words
        )

        self.assertEqual(int(pronunciation_accuracy), 100)

    def test_incorrect_transcription(self):
        words_real = "Ich habe sehr viel glück, am leben und gesund zu sein"
        words_transcribed = "Ic hab zeh viel guck am und gesund tu sein"

        real_and_transcribed_words, _, _ = trainer_SST_lambda[
            "de"
        ].match_sample_and_recorded_words(words_real, words_transcribed)

        pronunciation_accuracy, _ = trainer_SST_lambda["de"].get_pronunciation_accuracy(
            real_and_transcribed_words
        )

        self.assertEqual(int(pronunciation_accuracy), 71)


if __name__ == "__main__":
    unittest.main()
