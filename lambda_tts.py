import json
import os
import base64
import soundfile as sf
import models
import utils_file_io

SAMPLING_RATE = 16000
model_TTS_lambda = models.get_tts_model('de')


def lambda_handler(event):

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
