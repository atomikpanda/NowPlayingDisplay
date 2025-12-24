import queue
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd


class MicrophoneCapture:
    """Captures audio from the microphone"""

    def __init__(
        self, sample_rate: int = 44100, chunk_size: int = 2048, debug: bool = False
    ) -> None:
        self.sample_rate: int = sample_rate
        self.chunk_size: int = chunk_size
        self.audio_queue: queue.Queue = queue.Queue(maxsize=10)
        self.stream: Optional[sd.InputStream] = None
        self.debug: bool = debug
        self.chunk_count: int = 0
        self.overflow_count: int = 0

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        """Callback for sounddevice to receive audio data"""
        if status:
            self.overflow_count += 1
            # Don't print in callback - it's too slow
            return

        # Increment counter without printing (printing is slow)
        if self.debug:
            self.chunk_count += 1

        # Try to put data in queue, but don't block if full
        try:
            # Use put_nowait to avoid blocking, and don't copy the array
            # We'll copy it when retrieving from the queue instead
            self.audio_queue.put_nowait(indata[:])
        except queue.Full:
            # Queue is full, drop this chunk to prevent overflow
            pass

    def start(self) -> None:
        """Start capturing from microphone"""
        if self.stream is not None:
            print("Microphone already active")
            return

        if self.debug:
            print(f"Available audio devices:\n{sd.query_devices()}")

        self.stream = sd.InputStream(
            device="hw:1,0",
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16,
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        )
        self.stream.start()
        print(
            f"Microphone started (sample rate: {self.sample_rate} Hz, chunk size: {self.chunk_size})"
        )

    def stop(self) -> None:
        """Stop capturing from microphone"""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("Microphone stopped")

    def get_chunk(self) -> Tuple[Optional[bytes], int]:
        """Get the next audio chunk from the queue (non-blocking)"""
        try:
            # Non-blocking get
            data = self.audio_queue.get_nowait()
            if self.debug and self.chunk_count % 100 == 0:
                print(
                    f"Retrieved chunk {self.chunk_count}, queue: {self.audio_queue.qsize()}, overflows: {self.overflow_count}"
                )
            # Copy here instead of in the callback
            return data.copy().tobytes(), self.sample_rate
        except queue.Empty:
            return None, self.sample_rate

    def __enter__(self):
        """Context manager support"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.stop()
