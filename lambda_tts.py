"""
AWS Lambda function for Text-to-Speech (TTS) conversion.

This module uses the Silero TTS model to convert a given text string into
speech audio and returns it as a base64 encoded WAV file.
"""
import json
import os
import base64
import soundfile as sf
import models
import utils_file_io

SAMPLING_RATE = 16000
model_TTS_lambda = models.get_tts_model('de')


def lambda_handler(event):
    """
    Main handler for the AWS Lambda function.

    Receives a text string, generates the corresponding audio using a TTS model,
    and returns the audio as a base64 encoded string.

    Args:
        event (dict): The Lambda event object, expected to contain:
            - "body" (str): A JSON string with a "value" key holding the
                            text to be converted to speech.

    Returns:
        dict: A dictionary suitable for an API Gateway response, containing:
              - "statusCode": 200
              - "headers": CORS headers.
              - "body": A JSON string with the key "wavBase64" containing the
                        base64 encoded WAV audio.
    """

    body = json.loads(event['body'])

    text_string = body['value']

    linear_factor = 0.2
    audio = model_TTS_lambda.apply_tts(texts=[text_string],
                                       sample_rate=SAMPLING_RATE)[0].detach().numpy()*linear_factor
    random_file_name = utils_file_io.generate_random_string(20)+'.wav'

    sf.write('./'+random_file_name, audio, 16000)

    with open(random_file_name, "rb") as f:
        audio_byte_array = f.read()

    os.remove(random_file_name)


    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Headers': '*',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        },
        'body': json.dumps(
            {
                "wavBase64": str(base64.b64encode(audio_byte_array))[2:-1],
            },
        )
    }
