#!/bin/bash
export DISPLAY=:0
export XAUTHORITY=/home/pi/.Xauthority
# Optional: Add a delay to ensure the desktop environment is ready
sleep 10 
cd /home/pi/NowPlayingDisplay || exit 1
/home/pi/.local/bin/uv run now_playing_web.py &
/home/pi/.local/bin/uv run main.py &

wait