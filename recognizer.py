import json

import requests
import sounddevice as sd
from scipy.io.wavfile import write


class SongRecognizer:
    def __init__(
        self, api_token, duration=10, filename="recording.wav", samplerate=44100
    ):
        self.api_token = api_token
        self.duration = duration
        self.filename = filename
        self.samplerate = samplerate

    def record_audio(self, device=None):
        print("Recording...")
        recording = sd.rec(
            int(self.duration * self.samplerate),
            samplerate=self.samplerate,
            channels=1,
            dtype="int16",
            device=device,  # Optional: specify input device
        )
        sd.wait()
        write(self.filename, self.samplerate, recording)
        print("Saved:", self.filename)

    def recognize_song(self):
        print("Sending to AudD...")
        with open(self.filename, "rb") as f:
            files = {"file": f}
            data = {
                "api_token": self.api_token,
                "return": "apple_music,spotify",
            }
            response = requests.post("https://api.audd.io/", data=data, files=files)
            return response.json()
