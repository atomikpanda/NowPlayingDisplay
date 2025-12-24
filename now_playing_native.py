#!/usr/bin/env python3
"""
Now Playing full-bleed UI for Raspberry Pi (Zero 2 friendly)
- Left half: album art "cover" fill (cropped as needed)
- Right half: track + Artist/Album rows + status pill + updated time
- Polls:
    /now-playing-data (default: http://localhost:5000/now-playing-data)
    recognition state (default: http://localhost:5432/recognition-state)
"""

import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import pygame
import requests

# ----------------------------- Config -----------------------------

NOW_PLAYING_URL = os.getenv("NOW_PLAYING_URL", "http://localhost:5432/now-playing-data")
RECOGNITION_STATE_URL = os.getenv(
    "RECOGNITION_STATE_URL", "http://localhost:5432/recognition-state"
)

# Poll slower by default on Pi Zero W (still configurable via env)
REFRESH_MS = int(os.getenv("REFRESH_MS", "2500"))
STATE_REFRESH_MS = int(os.getenv("STATE_REFRESH_MS", "1200"))

# If you have a local placeholder image, set FALLBACK_ART_PATH.
FALLBACK_ART_PATH = os.getenv(
    "FALLBACK_ART_PATH", ""
)  # e.g. "/home/pi/NowPlayingDisplay/images/missing_art.png"

# For kiosk: set to 1 to hide mouse cursor
HIDE_CURSOR = os.getenv("HIDE_CURSOR", "1") == "1"

# If you want a fixed size (e.g. 1024x600), set WIDTH/HEIGHT env vars
FIXED_WIDTH = os.getenv("WIDTH")
FIXED_HEIGHT = os.getenv("HEIGHT")

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "1.0"))

# UI refresh cap (used only when there *are* changes / events)
UI_FPS = int(os.getenv("UI_FPS", "8"))

# ----------------------------- Data -----------------------------


@dataclass
class NowPlaying:
    art_url: str = ""
    track: str = ""
    artist: str = ""
    album: str = ""


@dataclass
class RecState:
    state: str = "idle"  # "listening" | "idle" etc.


# ----------------------------- Helpers -----------------------------


def clamp(n, a, b):
    return max(a, min(b, n))


def safe_text(s: Optional[str]) -> str:
    if not s:
        return "—"
    s2 = str(s).strip()
    return s2 if s2 else "—"


def fit_cover(src_w: int, src_h: int, dst_w: int, dst_h: int) -> pygame.Rect:
    """
    Return a source-rect (crop) that will "cover" dst (like CSS object-fit: cover),
    assuming we scale the cropped region to dst size.
    """
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return pygame.Rect(0, 0, src_w, src_h)

    src_ratio = src_w / src_h
    dst_ratio = dst_w / dst_h

    if src_ratio > dst_ratio:
        # source is wider -> crop left/right
        new_w = int(src_h * dst_ratio)
        x = (src_w - new_w) // 2
        return pygame.Rect(x, 0, new_w, src_h)
    else:
        # source is taller -> crop top/bottom
        new_h = int(src_w / dst_ratio)
        y = (src_h - new_h) // 2
        return pygame.Rect(0, y, src_w, new_h)


def draw_vertical_gradient(
    surface: pygame.Surface, rect: pygame.Rect, top_rgba, bot_rgba
):
    """Simple vertical gradient fill."""
    x, y, w, h = rect
    if h <= 0:
        return
    for i in range(h):
        t = i / max(1, h - 1)
        r = int(top_rgba[0] + (bot_rgba[0] - top_rgba[0]) * t)
        g = int(top_rgba[1] + (bot_rgba[1] - top_rgba[1]) * t)
        b = int(top_rgba[2] + (bot_rgba[2] - top_rgba[2]) * t)
        a = int(top_rgba[3] + (bot_rgba[3] - top_rgba[3]) * t)
        pygame.draw.line(surface, (r, g, b, a), (x, y + i), (x + w - 1, y + i))


def render_wrapped_ellipsis(
    font: pygame.font.Font, text: str, max_width: int, max_lines: int
) -> Tuple[list, bool]:
    """
    Return list of lines (<= max_lines) wrapped to max_width.
    If truncated, last line ends with ellipsis.
    """
    words = text.split()
    if not words:
        return ["—"], False

    lines = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if font.size(trial)[0] <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
            if len(lines) >= max_lines:
                break

    if len(lines) < max_lines and cur:
        lines.append(cur)

    truncated = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True

    if len(lines) == max_lines:
        # Check if we consumed all words
        consumed = " ".join(lines).split()
        if len(consumed) < len(words):
            truncated = True

    if truncated and lines:
        # add ellipsis to last line
        last = lines[-1]
        ell = "…"
        while last and font.size(last + ell)[0] > max_width:
            last = last[:-1].rstrip()
        lines[-1] = (last + ell) if last else ell

    return lines, truncated


# ----------------------------- Network Poller -----------------------------


class Poller:
    def __init__(self):
        self._lock = threading.Lock()
        self.now_playing = NowPlaying()
        self.rec_state = RecState()
        self.last_update_str = "server render"
        self.online = True

        self._art_surface: Optional[pygame.Surface] = None
        self._art_key: str = ""

        # Cache for already-cropped+scaled art (big CPU saver vs doing it every draw)
        self._scaled_art: Optional[pygame.Surface] = None
        self._scaled_art_key: Tuple[str, int, int] = ("", 0, 0)

        self._stop = threading.Event()
        self._changed = threading.Event()
        self._changed.set()  # force initial draw

        # Reuse sessions (keep-alive) to reduce overhead
        self._np_session = requests.Session()
        self._st_session = requests.Session()
        self._art_session = requests.Session()

    def stop(self):
        self._stop.set()
        self._changed.set()

    def start(self):
        threading.Thread(target=self._poll_now_playing, daemon=True).start()
        threading.Thread(target=self._poll_state, daemon=True).start()

    def mark_changed(self):
        self._changed.set()

    def wait_changed(self, timeout_s: float) -> bool:
        return self._changed.wait(timeout_s)

    def clear_changed(self):
        self._changed.clear()

    def _poll_now_playing(self):
        prev_key = None
        prev_online = None
        prev_updated = None

        while not self._stop.is_set():
            t0 = time.time()
            try:
                r = self._np_session.get(
                    NOW_PLAYING_URL,
                    timeout=REQUEST_TIMEOUT,
                    headers={"Cache-Control": "no-store"},
                )
                r.raise_for_status()
                d = r.json()

                np = NowPlaying(
                    art_url=safe_text(d.get("art_url", "")) if d.get("art_url") else "",
                    track=safe_text(d.get("track", "")),
                    artist=safe_text(d.get("artist", "")),
                    album=safe_text(d.get("album", "")),
                )

                # Avoid “change every second”: show minutes, not seconds
                updated = time.strftime("%-I:%M %p")

                key = (np.art_url, np.track, np.artist, np.album)
                online = True

                changed = (
                    (key != prev_key)
                    or (online != prev_online)
                    or (updated != prev_updated)
                )

                with self._lock:
                    self.now_playing = np
                    self.online = online
                    self.last_update_str = updated
                    if prev_key is None or (key[0] != prev_key[0]):
                        # Art URL changed -> drop scaled art cache
                        self._scaled_art = None
                        self._scaled_art_key = ("", 0, 0)

                if changed:
                    self._changed.set()

                prev_key, prev_online, prev_updated = key, online, updated

            except Exception:
                online = False
                updated = "offline"
                changed = (online != prev_online) or (updated != prev_updated)

                with self._lock:
                    self.online = online
                    self.last_update_str = updated

                if changed:
                    self._changed.set()

                prev_online, prev_updated = online, updated

            elapsed = (time.time() - t0) * 1000.0
            sleep_ms = max(0, REFRESH_MS - int(elapsed))
            self._stop.wait(sleep_ms / 1000.0)

    def _poll_state(self):
        prev_state = None
        while not self._stop.is_set():
            t0 = time.time()
            try:
                r = self._st_session.get(
                    RECOGNITION_STATE_URL,
                    timeout=REQUEST_TIMEOUT,
                    headers={"Cache-Control": "no-store"},
                )
                r.raise_for_status()
                d = r.json()
                st = str(d.get("state", "idle")).strip().lower()
                if st not in ("listening", "idle"):
                    st = "idle"

                if st != prev_state:
                    with self._lock:
                        self.rec_state = RecState(state=st)
                    self._changed.set()
                    prev_state = st

            except Exception:
                if prev_state != "idle":
                    with self._lock:
                        self.rec_state = RecState(state="idle")
                    self._changed.set()
                    prev_state = "idle"

            elapsed = (time.time() - t0) * 1000.0
            sleep_ms = max(0, STATE_REFRESH_MS - int(elapsed))
            self._stop.wait(sleep_ms / 1000.0)

    def get_snapshot(self) -> Tuple[NowPlaying, RecState, str, bool]:
        with self._lock:
            return self.now_playing, self.rec_state, self.last_update_str, self.online

    def get_art_surface(self, art_url: str) -> Optional[pygame.Surface]:
        """
        Lazy-load art image. Supports:
          - file paths (file:///... or /path)
          - http/https URLs
        Caches by art_url.
        """
        art_url = (art_url or "").strip()

        with self._lock:
            if art_url == self._art_key and self._art_surface is not None:
                return self._art_surface

        surf = None

        # Fallback local file if no art_url
        if not art_url and FALLBACK_ART_PATH:
            try:
                surf = pygame.image.load(FALLBACK_ART_PATH).convert()
            except Exception:
                surf = None

        elif art_url.startswith("file://"):
            path = art_url[7:]
            try:
                surf = pygame.image.load(path).convert()
            except Exception:
                surf = None

        elif art_url.startswith("/") and os.path.exists(art_url):
            try:
                surf = pygame.image.load(art_url).convert()
            except Exception:
                surf = None

        elif art_url.startswith("http://") or art_url.startswith("https://"):
            try:
                r = self._art_session.get(
                    art_url,
                    timeout=REQUEST_TIMEOUT,
                    headers={"Cache-Control": "no-store"},
                )
                r.raise_for_status()
                import io

                bio = io.BytesIO(r.content)
                # convert() is faster to blit; if you need alpha artwork, use convert_alpha()
                surf = pygame.image.load(bio).convert()
            except Exception:
                surf = None

        with self._lock:
            self._art_key = art_url
            self._art_surface = surf

        return surf

    def get_scaled_art(
        self, art_url: str, target_w: int, target_h: int
    ) -> Optional[pygame.Surface]:
        """
        Return cached album art already cropped+scaled to (target_w, target_h).
        """
        art_url = (art_url or "").strip()
        cache_key = (art_url, int(target_w), int(target_h))

        with self._lock:
            if self._scaled_art is not None and cache_key == self._scaled_art_key:
                return self._scaled_art

        art = self.get_art_surface(art_url)
        if not art:
            return None

        sw, sh = art.get_width(), art.get_height()
        src = fit_cover(sw, sh, target_w, target_h)
        cropped = art.subsurface(src)
        scaled = pygame.transform.smoothscale(cropped, (target_w, target_h))

        with self._lock:
            self._scaled_art = scaled
            self._scaled_art_key = cache_key

        return scaled


# ----------------------------- UI -----------------------------


class NowPlayingUI:
    def __init__(self):
        pygame.init()
        pygame.font.init()

        if HIDE_CURSOR:
            pygame.mouse.set_visible(False)

        # Reduce event overhead
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN])

        if FIXED_WIDTH and FIXED_HEIGHT:
            self.w = int(FIXED_WIDTH)
            self.h = int(FIXED_HEIGHT)
            flags = pygame.NOFRAME | pygame.HWSURFACE | pygame.DOUBLEBUF
            self.screen = pygame.display.set_mode((self.w, self.h), flags)
        else:
            info = pygame.display.Info()
            self.w, self.h = info.current_w, info.current_h
            flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
            self.screen = pygame.display.set_mode((self.w, self.h), flags)

        pygame.display.set_caption("Now Playing")

        # Use a separate surface for alpha effects
        self.ui = pygame.Surface((self.w, self.h), pygame.SRCALPHA)

        # Fonts: pygame doesn't do clamp(), so we approximate based on width
        self.font_track = self._make_font(int(clamp(self.w * 0.055, 28, 86)), bold=True)
        self.font_label = self._make_font(int(clamp(self.w * 0.012, 12, 18)), bold=True)
        self.font_value = self._make_font(
            int(clamp(self.w * 0.020, 16, 30)), bold=False
        )
        self.font_small = self._make_font(
            int(clamp(self.w * 0.012, 12, 16)), bold=False
        )
        self.font_pill = self._make_font(int(clamp(self.w * 0.014, 12, 18)), bold=True)

        self.clock = pygame.time.Clock()
        self.poller = Poller()
        self.poller.start()

        self.bg_base = (11, 15, 20)  # #0b0f14
        self.text = (232, 240, 251)
        self.muted = (138, 160, 184)

        self.state_colors = {
            "listening": (34, 197, 94),
            "idle": (100, 116, 139),
        }

        # Pre-render static overlays once (avoid per-frame gradient work + allocations)
        left_w = self.w // 2
        right_w = self.w - left_w

        self._art_overlay = pygame.Surface((left_w, self.h), pygame.SRCALPHA)
        draw_vertical_gradient(
            self._art_overlay,
            self._art_overlay.get_rect(),
            (255, 255, 255, 22),
            (0, 0, 0, 90),
        )

        self._meta_bg = pygame.Surface((right_w, self.h), pygame.SRCALPHA)
        draw_vertical_gradient(
            self._meta_bg, self._meta_bg.get_rect(), (0, 0, 0, 55), (0, 0, 0, 90)
        )

        # Cache constant label renders
        self._label_cache = {
            "ARTIST": self.font_label.render("ARTIST", True, (232, 240, 251, 205)),
            "ALBUM": self.font_label.render("ALBUM", True, (232, 240, 251, 205)),
        }

    def _make_font(self, size: int, bold: bool) -> pygame.font.Font:
        # Default system font
        f = pygame.font.SysFont(None, size)
        f.set_bold(bold)
        return f

    def run(self):
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise SystemExit
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            raise SystemExit
                        if event.key == pygame.K_r:
                            self.poller.mark_changed()

                # Block a bit waiting for changes; don’t spin at full speed
                changed = self.poller.wait_changed(timeout_s=1.0 / max(1, UI_FPS))
                if changed:
                    self.draw()
                    pygame.display.flip()
                    self.poller.clear_changed()

                self.clock.tick(UI_FPS)
        finally:
            self.poller.stop()
            pygame.quit()

    def draw(self):
        np, rs, updated_str, online = self.poller.get_snapshot()

        left_w = self.w // 2
        right_w = self.w - left_w

        art_rect = pygame.Rect(0, 0, left_w, self.h)
        meta_rect = pygame.Rect(left_w, 0, right_w, self.h)

        # Background
        self.screen.fill(self.bg_base)

        # --- Left: art (cached scaled)
        art_scaled = self.poller.get_scaled_art(np.art_url, art_rect.w, art_rect.h)
        if art_scaled:
            self.screen.blit(art_scaled, art_rect.topleft)
        else:
            pygame.draw.rect(self.screen, (16, 24, 38), art_rect)
            t = self.font_value.render("No artwork", True, self.muted)
            self.screen.blit(
                t,
                (
                    art_rect.centerx - t.get_width() // 2,
                    art_rect.bottom - t.get_height() - 24,
                ),
            )

        # Pre-rendered overlays
        self.screen.blit(self._art_overlay, art_rect.topleft)
        self.screen.blit(self._meta_bg, meta_rect.topleft)

        # Divider line
        pygame.draw.line(
            self.screen, (255, 255, 255, 25), (left_w, 0), (left_w, self.h), 1
        )

        # Padding similar to clamp(18px, 3.5vw, 56px)
        pad = int(clamp(self.w * 0.035, 18, 56))
        x0 = meta_rect.x + pad
        y = pad

        # Track: up to 3 lines with ellipsis
        track_text = safe_text(np.track)
        track_max_w = meta_rect.w - 2 * pad
        track_lines, _ = render_wrapped_ellipsis(
            self.font_track, track_text, track_max_w, max_lines=3
        )
        for line in track_lines:
            surf = self.font_track.render(line, True, self.text)
            self.screen.blit(surf, (x0, y))
            y += surf.get_height() + int(clamp(self.h * 0.006, 2, 10))

        y += int(clamp(self.h * 0.02, 10, 26))

        # Info rows
        y = self._draw_row("Artist", safe_text(np.artist), x0, y, meta_rect.w, pad)
        y += int(clamp(self.h * 0.012, 8, 18))
        y = self._draw_row("Album", safe_text(np.album), x0, y, meta_rect.w, pad)

        # Status area at bottom
        status_h = int(clamp(self.h * 0.16, 96, 170))
        status_rect = pygame.Rect(
            meta_rect.x + pad,
            meta_rect.bottom - pad - status_h,
            meta_rect.w - 2 * pad,
            status_h,
        )

        # top border line
        pygame.draw.line(
            self.screen,
            (255, 255, 255, 30),
            (status_rect.x, status_rect.y),
            (status_rect.right, status_rect.y),
            1,
        )

        # Pill
        pill_state = rs.state if rs.state in ("listening", "idle") else "idle"
        dot_color = self.state_colors.get(pill_state, self.state_colors["idle"])
        pill_text = pill_state

        pill_pad_x = 14
        pill_pad_y = 10
        pill_label = self.font_pill.render(pill_text, True, self.text)
        dot_r = 5

        pill_w = pill_pad_x * 2 + (dot_r * 2 + 10) + pill_label.get_width()
        pill_h = max(36, pill_pad_y * 2 + pill_label.get_height())

        pill_rect = pygame.Rect(status_rect.x, status_rect.y + 18, pill_w, pill_h)

        # Rounded pill background
        pill_bg = pygame.Surface((pill_rect.w, pill_rect.h), pygame.SRCALPHA)
        pygame.draw.rect(
            pill_bg, (255, 255, 255, 10), pill_bg.get_rect(), border_radius=999
        )
        pygame.draw.rect(
            pill_bg, (255, 255, 255, 28), pill_bg.get_rect(), width=1, border_radius=999
        )
        self.screen.blit(pill_bg, pill_rect.topleft)

        # Dot + glow
        cx = pill_rect.x + pill_pad_x + dot_r
        cy = pill_rect.y + pill_rect.h // 2
        # glow
        pygame.draw.circle(self.screen, (*dot_color, 45), (cx, cy), dot_r + 4)
        pygame.draw.circle(self.screen, dot_color, (cx, cy), dot_r)

        # Pill text
        tx = cx + dot_r + 10
        ty = pill_rect.y + (pill_rect.h - pill_label.get_height()) // 2
        self.screen.blit(pill_label, (tx, ty))

        # Updated text (right side)
        upd_label = f"Updated: {updated_str}"
        upd_surf = self.font_small.render(upd_label, True, (232, 240, 251, 160))
        self.screen.blit(
            upd_surf, (status_rect.right - upd_surf.get_width(), pill_rect.y + 6)
        )

        # Offline hint
        if not online:
            off = self.font_small.render("offline", True, (239, 68, 68))
            self.screen.blit(
                off,
                (
                    status_rect.right - off.get_width(),
                    pill_rect.y + 6 + upd_surf.get_height() + 4,
                ),
            )

    def _draw_row(
        self, label: str, value: str, x0: int, y: int, meta_w: int, pad: int
    ) -> int:
        label_up = label.upper()
        label_s = self._label_cache.get(label_up)
        if label_s is None:
            label_s = self.font_label.render(label_up, True, (232, 240, 251, 205))
            self._label_cache[label_up] = label_s
        self.screen.blit(label_s, (x0, y))

        # value area starts after label + gap
        gap = int(clamp(self.w * 0.014, 10, 18))
        vx = x0 + label_s.get_width() + gap
        v_max_w = (meta_w - 2 * pad) - (vx - x0)
        v = value

        # Ellipsize value
        ell = "…"
        while v and self.font_value.size(v)[0] > v_max_w:
            v = v[:-1].rstrip()
        if v != value:
            v = (v + ell) if v else ell

        value_s = self.font_value.render(v, True, self.muted)
        self.screen.blit(value_s, (vx, y - 2))
        return y + max(label_s.get_height(), value_s.get_height())


def main():
    ui = NowPlayingUI()
    ui.run()


if __name__ == "__main__":
    main()
