#!/bin/bash
export DISPLAY=:0
export XAUTHORITY=/home/pi/.Xauthority
# Optional: Add a delay to ensure the desktop environment is ready

cd /home/pi/NowPlayingDisplay || exit 1
/home/pi/.local/bin/uv run now_playing_web.py &
sleep 10
/home/pi/.local/bin/uv run main.py &

wait