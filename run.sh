#!/bin/bash
export DISPLAY=:0
export XAUTHORITY=/home/pi/.Xauthority
# Optional: Add a delay to ensure the desktop environment is ready
sleep 10 

uv run now_playing_web.py &
uv run main.py &

wait