#!/bin/bash

uv run now_playing_web.py &
uv run main.py &

wait