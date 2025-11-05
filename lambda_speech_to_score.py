"""AWS Lambda function for processing recorded audio and evaluating pronunciation.

This module provides an AWS Lambda function that serves as the backend for an
API endpoint designed to handle audio recordings. The primary purpose of this
function is to process base64 encoded audio data, transcribe the speech to text,
and compare it against a provided reference text. Following the comparison, it
generates a detailed analysis of the pronunciation, including accuracy scores and
word-level feedback.

The core logic for pronunciation assessment is handled by the `pronunciation_trainer`
module, which this Lambda function utilizes. Additionally, it integrates with the
`word_matching` module to ensure proper alignment between the transcribed text and
the reference text. This integration is crucial for delivering precise and
contextually accurate feedback on pronunciation.

The function is designed to be triggered by an API Gateway event, and it returns
a JSON response containing the full analysis. This response can then be used by a
front-end application to display the results to the user, helping them improve
their pronunciation skills.
"""

import base64
import time
import logging
import os
import json
import tempfile
import torch
import audioread
import numpy as np
from torchaudio.transforms import Resample
import word_matching as wm
import pronunciation_trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

trainer_SST_lambda = {}
trainer_SST_lambda["de"] = pronunciation_trainer.get_trainer("de")
trainer_SST_lambda["en"] = pronunciation_trainer.get_trainer("en")

transform = Resample(orig_freq=48000, new_freq=16000)


def lambda_handler(event):
    """Main handler for the AWS Lambda function.

    This function serves as the primary entry point for the AWS Lambda service. It is
    designed to receive an event from an API Gateway, which contains the necessary
    data for pronunciation analysis. The core functionality includes decoding the
    audio data, processing it to evaluate pronunciation against a reference text,
    and returning a detailed JSON response with the analysis results.

    The function expects the event to contain a `body` with a JSON string that
    includes the reference text (`title`), the base64 encoded audio (`base64Audio`),
    and the language of the text (`language`).

    Args:
        event (dict): The event object provided by AWS Lambda, which is expected to
                    contain the following structure:
                    - "body" (str): A JSON string with the following keys:
                        - "title" (str): The reference text (real text) against
                          which the audio is compared.
                        - "base64Audio" (str): The base64 encoded audio data.
                        - "language" (str): The language of the text (e.g., "en", "de").

    Returns:
        str: A JSON string containing the results of the pronunciation analysis.
             The JSON object includes the following keys:
             - "real_transcript" (str): The transcript of the recorded audio.
             - "ipa_transcript" (str): The IPA transcript of the recorded audio.
             - "pronunciation_accuracy" (str): The overall pronunciation accuracy score.
             - "real_transcripts" (str): The reference text, with words separated by spaces.
             - "matched_transcripts" (str): The transcribed words that match the reference.
             - "real_transcripts_ipa" (str): The IPA representation of the reference text.
             - "matched_transcripts_ipa" (str): The IPA representation of the matched words.
             - "pair_accuracy_category" (str): A string of categories for each word pair.
             - "start_time" (float): The start time of the audio processing.
             - "end_time" (float): The end time of the audio processing.
             - "is_letter_correct_all_words" (str): A string indicating which letters
               were transcribed correctly for each word.

             If the `real_text` is empty, the function returns an empty body with a
             status code of 200, effectively terminating the process early.
    """

    data = json.loads(event["body"])

    real_text = data["title"]
    file_bytes = base64.b64decode(data["base64Audio"][22:].encode("utf-8"))
    language = data["language"]

    if len(real_text) == 0:
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
            },
            "body": "",
        }

    tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
    tmp_name = tmp.name

    try:
        tmp.write(file_bytes)
        tmp.flush()

        tmp.close()

        signal, _ = audioread_load(tmp_name)

    finally:

        os.remove(tmp_name)

    signal = transform(torch.Tensor(signal)).unsqueeze(0)

    result = trainer_SST_lambda[language].process_audio_for_given_text(
        signal, real_text
    )

    start = time.time()
    real_transcripts_ipa = " ".join(
        [word[0] for word in result["real_and_transcribed_words_ipa"]]
    )
    matched_transcripts_ipa = " ".join(
        [word[1] for word in result["real_and_transcribed_words_ipa"]]
    )

    real_transcripts = " ".join(
        [word[0] for word in result["real_and_transcribed_words"]]
    )
    matched_transcripts = " ".join(
        [word[1] for word in result["real_and_transcribed_words"]]
    )

    words_real = real_transcripts.lower().split()
    mapped_words = matched_transcripts.split()

    is_letter_correct_all_words = ""
    for idx, word_real in enumerate(words_real):

        mapped_letters, _ = wm.get_best_mapped_words(mapped_words[idx], word_real)

        is_letter_correct = wm.get_which_letters_were_transcribed_correctly(
            word_real, mapped_letters
        )

        is_letter_correct_all_words += (
            "".join([str(is_correct) for is_correct in is_letter_correct]) + " "
        )

    pair_accuracy_category = " ".join(
        [str(category) for category in result["pronunciation_categories"]]
    )
    logger.info("Time to post-process results: %s", str(time.time() - start))

    res = {
        "real_transcript": result["recording_transcript"],
        "ipa_transcript": result["recording_ipa"],
        "pronunciation_accuracy": str(int(result["pronunciation_accuracy"])),
        "real_transcripts": real_transcripts,
        "matched_transcripts": matched_transcripts,
        "real_transcripts_ipa": real_transcripts_ipa,
        "matched_transcripts_ipa": matched_transcripts_ipa,
        "pair_accuracy_category": pair_accuracy_category,
        "start_time": result["start_time"],
        "end_time": result["end_time"],
        "is_letter_correct_all_words": is_letter_correct_all_words,
    }

    return json.dumps(res)


def audioread_load(path, offset=0.0, duration=None, dtype=np.float32):
    """Loads an audio buffer from a file using the `audioread` library.

    This function is designed to efficiently load audio data from a given file path.
    It reads the audio in chunks, processes them, and then concatenates them to form
    a complete audio signal. This approach is memory-efficient, especially for large
    audio files.

    The function allows for specifying an offset and duration, enabling partial
    loading of the audio. It also supports specifying the data type for the output
    array, providing flexibility for different use cases.

    Args:
        path (str): The file path of the audio file to be loaded.
        offset (float, optional): The time (in seconds) from the beginning of the
                                  audio to start reading. Defaults to 0.0.
        duration (float, optional): The maximum duration (in seconds) of the audio
                                    to load. If None, the entire file is loaded.
                                    Defaults to None.
        dtype (np.dtype, optional): The desired NumPy data type for the output array.
                                    Defaults to np.float32.

    Returns:
        Tuple[np.ndarray, int]: A tuple where the first element is a NumPy array
                                containing the loaded audio samples, and the second
                                element is the native sample rate of the audio file.
    """

    y = []
    with audioread.audio_open(path) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration)) * n_channels)

        n = 0

        for frame in input_file:
            frame = buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[: s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev) :]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
    else:
        y = np.empty(0, dtype=dtype)

    return y, sr_native


# From Librosa


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Converts an integer data buffer to floating-point values.

    This function is essential for handling audio data that is stored in an
    integer format, such as in WAV files. It takes an integer buffer and scales
    it to a floating-point representation, which is a common requirement for
    audio processing and analysis in NumPy.

    The conversion is performed by first calculating a scaling factor based on the
    number of bytes per sample. This factor is then applied to the data, which is
    re-interpreted from an integer format to the specified floating-point type.

    Args:
        x (np.ndarray): The integer-valued data buffer to be converted.
        n_bytes (int, optional): The number of bytes per sample in the input buffer `x`.
                                 Common values are 1, 2, or 4. Defaults to 2.
        dtype (numeric type, optional): The target NumPy data type for the output.
                                        Defaults to `np.float32`.

    Returns:
        np.ndarray: A NumPy array containing the input data, but cast to floating-point
                    values. This array is scaled and ready for further processing.
    """

    # Invert the scale of the data
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = f"<i{n_bytes}"

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)
