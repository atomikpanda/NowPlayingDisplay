from typing import Literal

from pydantic import BaseModel


class NowPlayingViewModel(BaseModel):
    art_url: str | None = None
    artist: str = ""
    album: str = ""
    track: str = ""
    state: Literal["stopped", "playing", "paused"] = "stopped"


class RecognitionState(BaseModel):
    state: Literal["idle", "listening"] = "idle"
