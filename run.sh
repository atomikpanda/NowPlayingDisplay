#!/bin/bash
export DISPLAY=:0
export XAUTHORITY=/home/pi/.Xauthority
# Optional: Add a delay to ensure the desktop environment is ready

cd /home/pi/NowPlayingDisplay || exit 1
/home/pi/.local/bin/uv run gunicorn --bind 0.0.0.0:5432 now_playing_web:npapi &
/home/pi/.local/bin/uv run main.py &
chromium \
  --kiosk \
  --no-memcheck \
  --noerrdialogs \
  --disable-infobars \
  --disable-session-crashed-bubble \
  --disable-features=TranslateUI \
  --disable-pinch \
  --overscroll-history-navigation=0 \
  --disable-component-update \
  --disable-background-networking \
  --disable-sync \
  --disable-default-apps \
  --disable-extensions \
  --disable-popup-blocking \
  --disable-translate \
  --disable-features=MediaRouter \
  --disable-features=VizDisplayCompositor \
  --disable-gpu \
  --disable-software-rasterizer \
  --autoplay-policy=no-user-gesture-required \
  "http://localhost:5432" &

wait