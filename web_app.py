"""
A simple Flask web server for local development and testing.

This module sets up a Flask application that serves the main HTML page and
provides API endpoints that mimic the behavior of the AWS Lambda functions.
It allows for running the entire application locally without needing to deploy
to AWS.
"""
import os
import logging
import json
import webbrowser
from flask import Flask, render_template, request
from flask_cors import CORS

import lambda_tts
import lambda_speech_to_score
import lambda_get_sample

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = '*'

ROOTPATH = ''


@app.route(ROOTPATH+'/')
def main():
    """Renders the main HTML page of the application."""
    return render_template('main.html')


@app.route(ROOTPATH+'/getAudioFromText', methods=['POST'])
def get_audio_from_text():
    """
    Handles POST requests to generate audio from text.

    This endpoint wraps the `lambda_tts.lambda_handler`, allowing the frontend
    to get synthesized speech for a given text string.
    """
    event = {'body': json.dumps(request.get_json(force=True))}
    return lambda_tts.lambda_handler(event)


@app.route(ROOTPATH+'/getSample', methods=['POST'])
def get_next():
    """
    Handles POST requests to get a new sample sentence.

    This endpoint wraps the `lambda_get_sample.lambda_handler`, providing the
    frontend with a new sentence to practice.
    """
    event = {'body':  json.dumps(request.get_json(force=True))}
    return lambda_get_sample.lambda_handler(event)


@app.route(ROOTPATH+'/GetAccuracyFromRecordedAudio', methods=['POST'])
def get_accuracy_from_recorded_audio():
    """
    Handles POST requests to evaluate the pronunciation of recorded audio.

    This endpoint wraps the `lambda_speech_to_score.lambda_handler`. It includes
    a try-except block to catch any errors during processing and log them,
    returning a safe response to the client.
    """
    try:
        event = {'body': json.dumps(request.get_json(force=True))}
        lambda_correct_output = lambda_speech_to_score.lambda_handler(event)
    except Exception:  # pylint: disable=broad-except
        logger.exception("An error occurred while processing the audio.")
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
