#!/bin/bash
export DISPLAY=:0
export XAUTHORITY=/home/pi/.Xauthority
# Optional: Add a delay to ensure the desktop environment is ready

cd /home/pi/NowPlayingDisplay || exit 1
/home/pi/.local/bin/uv run now_playing_web.py &
/home/pi/.local/bin/uv run gunicorn --bind 0.0.0.0:5432 now_playing_web:npapi &
/home/pi/.local/bin/uv run kiosk.py &

wait