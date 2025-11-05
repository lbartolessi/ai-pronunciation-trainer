import os
import json
import webbrowser
from flask import Flask, render_template, request
from flask_cors import CORS

import lambda_tts
import lambda_speech_to_score
import lambda_get_sample

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = '*'

ROOTPATH = ''


@app.route(ROOTPATH+'/')
def main():
    return render_template('main.html')


@app.route(ROOTPATH+'/getAudioFromText', methods=['POST'])
def get_audio_from_text():
    event = {'body': json.dumps(request.get_json(force=True))}
    return lambda_tts.lambda_handler(event)


@app.route(ROOTPATH+'/getSample', methods=['POST'])
def get_next():
    event = {'body':  json.dumps(request.get_json(force=True))}
    return lambda_get_sample.lambda_handler(event)


@app.route(ROOTPATH+'/GetAccuracyFromRecordedAudio', methods=['POST'])
def get_accuracy_from_recorded_audio():

    try:
        event = {'body': json.dumps(request.get_json(force=True))}
        lambda_correct_output = lambda_speech_to_score.lambda_handler(event)
    except Exception as e:
        print('Error: ', str(e))
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Credentials': "true",
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': ''
        }

    return lambda_correct_output


if __name__ == "__main__":
    LANGUAGE = 'de'
    print(os.system('pwd'))
    webbrowser.open_new('http://127.0.0.1:3000/')
    app.run(host="0.0.0.0", port=3000)
