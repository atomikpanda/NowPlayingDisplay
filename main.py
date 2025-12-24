import asyncio
import os

import httpx
from dotenv import load_dotenv
from shazamio import Serialize, Shazam
from shazamio.schemas.models import (
    SongSection,
)

from audio_capture import MicrophoneCapture
from music_dectector import MusicDetector
from shazam_models import ShazamResponse
from viewmodels import NowPlayingViewModel

load_dotenv()


async def get_more_info(shazam: Shazam, track_id: int) -> dict | None:
    track_data = await shazam.track_about(track_id)
    track_info = Serialize.track(track_data)
    album_info = None
    for section in track_info.sections if track_info.sections else []:
        if isinstance(section, SongSection):
            for meta in section.metadata:
                if meta.title == "Album":
                    album_info = {
                        "title": meta.text,
                    }
                    break
    return album_info


async def update_now_playing(shazam_response: ShazamResponse, album_info: dict | None):
    async with httpx.AsyncClient() as client:
        await client.post(
            "http://localhost:5432/update-now-playing",
            json=NowPlayingViewModel(
                art_url=shazam_response.track.images.coverart
                if shazam_response.track
                else None,
                artist=shazam_response.track.subtitle if shazam_response.track else "",
                album=album_info["title"] if album_info else "",
                track=shazam_response.track.title if shazam_response.track else "",
                state="playing",
            ).model_dump(mode="json"),
        )


async def send_to_shazam(shazam: Shazam, file_path: str):
    try:
        out = await shazam.recognize(file_path)
        response = ShazamResponse(**out)
        print(response)
        album = None
        if response.track:
            if response.matches and len(response.matches) > 0:
                first_match = response.matches[0]
                album = await get_more_info(shazam, int(first_match.id))

            print("Album info:")
            print(album)
            await update_now_playing(response, album)
    except Exception as e:
        print("Error recognizing track:", e)
    finally:
        os.remove(file_path)


async def set_is_music_detecting(is_detecting: bool):
    async with httpx.AsyncClient() as client:
        state = "listening" if is_detecting else "idle"
        await client.put(
            "http://localhost:5432/recognition-state",
            json={"state": state},
        )


async def main():
    shazam = Shazam(endpoint_country="US")

    def on_recording_complete(result: str) -> None:
        asyncio.create_task(send_to_shazam(shazam, result))

    def on_is_recording_changed(is_recording: bool) -> None:
        asyncio.create_task(set_is_music_detecting(is_recording))

    mic = MicrophoneCapture(debug=False, chunk_size=4096, sample_rate=16000)
    detector = MusicDetector(
        similarity_threshold=0.75,
        on_recording_complete=on_recording_complete,
        microphone=mic,
        debug=False,
        on_is_recording_changed=on_is_recording_changed,
    )
    task = detector.start_detection()
    if task:
        await task


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
