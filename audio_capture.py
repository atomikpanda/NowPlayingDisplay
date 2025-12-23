import queue
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd


class MicrophoneCapture:
    """Captures audio from the microphone"""

    def __init__(
        self, sample_rate: int = 44100, chunk_size: int = 1024, debug: bool = False
    ) -> None:
        self.sample_rate: int = sample_rate
        self.chunk_size: int = chunk_size
        self.audio_queue: queue.Queue = queue.Queue()
        self.stream: Optional[sd.InputStream] = None
        self.debug: bool = debug
        self.chunk_count: int = 0

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        """Callback for sounddevice to receive audio data"""
        if status:
            print(f"Audio callback status: {status}")

        if self.debug:
            self.chunk_count += 1
            if self.chunk_count % 100 == 0:
                print(
                    f"Audio callback received {self.chunk_count} chunks, queue size: {self.audio_queue.qsize()}"
                )

        # Put audio data into queue
        self.audio_queue.put(indata.copy())

    def start(self) -> None:
        """Start capturing from microphone"""
        if self.stream is not None:
            print("Microphone already active")
            return

        if self.debug:
            print(f"Available audio devices:\n{sd.query_devices()}")

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        )
        self.stream.start()
        print(f"Microphone started (sample rate: {self.sample_rate} Hz)")

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
            if self.debug:
                print(
                    f"Retrieved chunk from queue, shape: {data.shape}, queue remaining: {self.audio_queue.qsize()}"
                )
            return data.tobytes(), self.sample_rate
        except queue.Empty:
            # if self.debug:
            # print("Queue empty - no audio data available")
            return None, self.sample_rate

    def __enter__(self):
        """Context manager support"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.stop()
