from typing import List, Literal, Optional

from pydantic import BaseModel


class Match(BaseModel):
    id: str
    offset: float
    timeskew: float
    frequencyskew: float


class Location(BaseModel):
    accuracy: float


class ImageSet(BaseModel):
    background: Optional[str] = None
    coverart: Optional[str] = None
    overflow: Optional[str] = None
    default: Optional[str] = None


class Share(BaseModel):
    subject: str | None = None
    text: str | None = None
    href: str | None = None
    image: str | None = None
    twitter: str | None = None
    html: str | None = None
    avatar: str | None = None
    snapchat: str | None = None


class Action(BaseModel):
    name: str
    type: str
    id: Optional[str] = None
    uri: Optional[str] = None


class Provider(BaseModel):
    caption: str
    images: ImageSet
    actions: List[Action] | None = None
    type: str


class Hub(BaseModel):
    type: str
    image: str
    actions: List[Action] | None = None
    providers: List[Provider] | None = None
    explicit: bool
    displayname: str


class Genres(BaseModel):
    primary: str


class Track(BaseModel):
    layout: str
    type: Literal["MUSIC"]
    key: str
    title: str
    subtitle: str
    images: ImageSet
    share: Share
    coverarthq: str | None = None
    joecolor: str | None = None
    hub: Hub
    url: str
    genres: Genres
    albumadamid: str | None = None
    isrc: str | None = None


class ShazamResponse(BaseModel):
    matches: List[Match] | None = None
    location: Location | None = None
    timestamp: int | None = None
    timezone: str | None = None
    track: Track | None = None
    tagid: str | None = None
