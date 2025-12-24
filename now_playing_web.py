import logging
from threading import Thread
from wsgiref import validate

import webview
from flask import Flask, jsonify, render_template
from flask_pydantic import validate

from viewmodels import NowPlayingViewModel, RecognitionState

try:
    from npsettings_local import DEBUG
except ImportError:
    from npsettings import DEBUG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

npapi = Flask(__name__, template_folder="www")


npapi.config["state"] = NowPlayingViewModel(
    art_url=None, artist="", album="", track="", state="stopped"
)
npapi.config["recognition_state"] = RecognitionState()


@npapi.route("/update-now-playing", methods=["POST"])
@validate()
def update_now_playing(body: NowPlayingViewModel):
    """API endpoint for updating the now playing information on the display."""
    npapi.config["state"] = body

    return jsonify({"status": "success"})


@npapi.route("/recognition-state", methods=["GET"])
def get_recognition_state():
    """API endpoint to get the current recognition state."""
    return npapi.config["recognition_state"].model_dump(mode="json")


@npapi.route("/recognition-state", methods=["PUT"])
@validate()
def update_recognition_state(body: RecognitionState):
    """API endpoint to update the recognition state."""
    npapi.config["recognition_state"] = body
    return jsonify({"status": "success"})


@npapi.route("/")
def index():
    return render_template("index.html", data=npapi.config["state"])


@npapi.route("/now-playing-data", methods=["GET"])
def now_playing_data():
    return npapi.config["state"].model_dump(mode="json")


def start_api():
    """Start the Flask API to accept requests to update the now playing information."""
    flask_log = logging.getLogger("werkzeug")
    flask_log.setLevel(logging.ERROR)
    npapi.run(host="0.0.0.0", port=5432, threaded=True)


if __name__ == "__main__":
    if DEBUG:
        logger.setLevel(logging.DEBUG)

    logger.info("Starting API...")
    Thread(target=start_api, daemon=True).start()
