import asyncio
import datetime
import hashlib
import os
import wave
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import librosa
import numpy as np

if TYPE_CHECKING:
    from audio_capture import MicrophoneCapture


class AudioFingerprinter:
    """Handles audio fingerprinting and duplicate detection"""

    def __init__(
        self, cooldown_seconds: int = 120, similarity_threshold: float = 0.85
    ) -> None:
        self.cooldown_seconds: int = cooldown_seconds
        self.similarity_threshold: float = similarity_threshold
        self.last_fingerprint: Optional[np.ndarray] = None
        self.last_fingerprint_time: Optional[datetime.datetime] = None
        self._match_cache: dict[str, bool] = {}

    def create_fingerprint(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Create a fingerprint from audio features (returns feature vector, not hash)"""
        # Ensure clean input
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.clip(y, -1.0, 1.0)

        # Extract chroma features (pitch class profiles)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
        # Extract spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)

        # Combine features and take mean across time
        features = np.concatenate(
            [
                np.mean(chroma, axis=1),
                np.mean(contrast, axis=1),
                np.mean(mfcc, axis=1),
            ]
        )

        # Normalize the feature vector
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features = features / norm
        else:
            # Handle zero-norm case
            features = np.zeros_like(features)

        return features

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two feature vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def _fingerprint_hash(self, fingerprint: np.ndarray) -> str:
        """Stable hash for fingerprint arrays."""
        return hashlib.sha1(
            np.asarray(fingerprint, dtype=np.float32).tobytes()
        ).hexdigest()

    def set_has_match(self, fingerprint: np.ndarray, has_match: bool) -> None:
        """Cache whether this fingerprint has an external match."""
        self._match_cache[self._fingerprint_hash(fingerprint)] = has_match

    def set_has_match_by_hash(self, fingerprint_hash: str, has_match: bool) -> None:
        """Cache whether this fingerprint hash has an external match."""
        self._match_cache[fingerprint_hash] = has_match

    def get_has_match(self, fingerprint: np.ndarray) -> bool:
        """Return cached match flag, defaulting to False."""
        return self._match_cache.get(self._fingerprint_hash(fingerprint), False)

    def is_duplicate(self, fingerprint: np.ndarray) -> bool:
        """Check if the fingerprint is similar to the last detected song within cooldown period"""
        if self.last_fingerprint is None:
            return False

        if self.last_fingerprint_time is not None:
            time_since_last = (
                datetime.datetime.now() - self.last_fingerprint_time
            ).total_seconds()
            if time_since_last < self.cooldown_seconds:
                similarity = self._cosine_similarity(fingerprint, self.last_fingerprint)
                print(
                    f"Similarity to last song: {similarity:.4f} (threshold: {self.similarity_threshold})"
                )
                return similarity >= self.similarity_threshold

        return False

    def update(self, fingerprint: np.ndarray) -> None:
        """Update the last fingerprint and timestamp"""
        self.last_fingerprint = fingerprint.copy()
        self.last_fingerprint_time = datetime.datetime.now()

    def reset(self) -> None:
        """Reset fingerprint state"""
        self.last_fingerprint = None
        self.last_fingerprint_time = None
        self._match_cache.clear()


class AudioRecorder:
    """Handles audio recording and saving"""

    def __init__(self, save_dir: str = "./samples") -> None:
        self.save_dir: str = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_sample(self, y: np.ndarray, sr: int) -> str:
        """Save audio sample to WAV file"""
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") + ".wav"
        filepath = os.path.join(self.save_dir, filename)
        with wave.open(filepath, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes((y * 32767).astype(np.int16).tobytes())
        print(f"Saved sample: {filepath}")
        return filepath


class MusicDetector:
    """Detects music in audio stream and triggers recording"""

    def __init__(
        self,
        threshold: float = 0.000813,
        sample_duration: int = 10,
        save_dir: str = "./samples",
        on_recording_complete: Optional[
            Callable[
                [str, str],
                None,
            ]
        ] = None,
        fingerprint_cooldown: int = 30,
        similarity_threshold: float = 0.85,  # Add similarity threshold parameter
        audio_callback: Optional[Callable[[], Tuple[Optional[bytes], int]]] = None,
        microphone: Optional["MicrophoneCapture"] = None,
        debug: bool = False,
        silence_tolerance: float = 2.0,
        on_is_recording_changed: Optional[Callable[[bool], None]] = None,
    ) -> None:
        self.threshold: float = threshold
        self.sample_duration: int = sample_duration
        self.on_recording_complete: Optional[Callable[[str, str], None]] = (
            on_recording_complete
        )
        self.audio_callback: Optional[Callable[[], Tuple[Optional[bytes], int]]] = (
            audio_callback
        )
        self.microphone: Optional["MicrophoneCapture"] = microphone
        self.debug: bool = debug
        self.silence_tolerance: float = silence_tolerance
        self.on_is_recording_changed: Optional[Callable[[bool], None]] = (
            on_is_recording_changed
        )
        # Composition: use specialized classes
        self.fingerprinter: AudioFingerprinter = AudioFingerprinter(
            cooldown_seconds=fingerprint_cooldown,
            similarity_threshold=similarity_threshold,
        )
        self.recorder: AudioRecorder = AudioRecorder(save_dir=save_dir)

        # Detection state
        self.is_recording: bool = False
        self.buffer: list[np.ndarray] = []
        self.buffer_time: float = 0
        self.silence_time: float = 0  # Track silence duration

        # Asyncio
        self.detection_enabled: bool = False
        self.detection_task: Optional[asyncio.Task] = None

        # If microphone is provided, use its callback
        if microphone and not audio_callback:
            self.audio_callback = microphone.get_chunk

    def detect_music(self, y: np.ndarray, sr: int) -> bool:
        """Detect if audio contains music based on RMS threshold"""
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length
        )[0]
        rms_mean = np.mean(rms)

        if self.debug:
            print(
                f"RMS: {rms_mean:.6f}, Threshold: {self.threshold:.6f}, Detected: {rms_mean > self.threshold}"
            )

        return rms_mean > self.threshold

    def _handle_completed_recording(self, combined: np.ndarray, sr: int) -> None:
        """Process a completed recording"""
        print(
            f"_handle_completed_recording called with {len(combined)} samples at {sr} Hz"
        )

        fingerprint = self.fingerprinter.create_fingerprint(combined, sr)

        if not self.fingerprinter.is_duplicate(
            fingerprint
        ) or not self.fingerprinter.get_has_match(fingerprint):
            print("New song detected!")
            filepath = self.recorder.save_sample(combined, sr)

            if self.on_recording_complete:
                self.on_recording_complete(
                    filepath, self.fingerprinter._fingerprint_hash(fingerprint)
                )

            self.fingerprinter.update(fingerprint)
        else:
            print("Same song detected, skipping...")

    def _reset_recording_state(self) -> None:
        """Reset recording buffer and state"""
        was_recording = self.is_recording
        self.is_recording = False
        self.buffer = []
        self.buffer_time = 0
        self.silence_time = 0

        if self.on_is_recording_changed and was_recording:
            self.on_is_recording_changed(False)
        if self.debug and was_recording:
            print("Recording state reset")

    def process_chunk(self, chunk: bytes, sr: int) -> None:
        """Process a single audio chunk"""
        if not self.detection_enabled:
            return

        if self.debug and len(chunk) > 0:
            print(f"Processing chunk: {len(chunk)} bytes, sr={sr}")

        # Convert int16 bytes to float32 properly
        y = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)

        # Normalize int16 range [-32768, 32767] to float32 range [-1.0, 1.0]
        y = y / 32768.0

        # Additional safety clipping
        y = np.clip(y, -1.0, 1.0)

        # Skip if audio is all zeros (silence)
        if np.all(y == 0):
            if self.debug:
                print("Skipping all-zero chunk")
            return

        if self.debug:
            print(
                f"Audio array shape: {y.shape}, min: {y.min():.6f}, max: {y.max():.6f}"
            )

        chunk_duration = len(y) / sr
        is_music = self.detect_music(y, sr)

        if is_music:
            # Reset silence counter when music is detected
            self.silence_time = 0

            if not self.is_recording:
                self.is_recording = True
                self.buffer = []
                self.buffer_time = 0
                if self.debug:
                    print("Started recording")
                if self.on_is_recording_changed:
                    self.on_is_recording_changed(True)

            self.buffer.append(y)
            self.buffer_time += chunk_duration

            if self.debug:
                print(f"Recording... {self.buffer_time:.2f}s / {self.sample_duration}s")

            if self.buffer_time >= self.sample_duration:
                combined = np.concatenate(self.buffer)
                self._handle_completed_recording(combined, sr)
                self._reset_recording_state()
        else:
            # No music detected
            if self.is_recording:
                # We're recording but hit silence - track it
                self.silence_time += chunk_duration

                if self.debug:
                    print(
                        f"Silence during recording: {self.silence_time:.2f}s / {self.silence_tolerance}s tolerance"
                    )

                # Still add the silent chunk to buffer to maintain timing
                self.buffer.append(y)
                self.buffer_time += chunk_duration

                # Check if we've exceeded silence tolerance
                if self.silence_time >= self.silence_tolerance:
                    if self.debug:
                        print("Silence tolerance exceeded - stopping recording")
                    self._reset_recording_state()
                # Check if we reached sample duration despite silence
                elif self.buffer_time >= self.sample_duration:
                    combined = np.concatenate(self.buffer)
                    self._handle_completed_recording(combined, sr)
                    self._reset_recording_state()

    async def _detection_loop(self) -> None:
        """Async loop for continuous detection"""
        loop_count = 0
        while self.detection_enabled:
            if self.audio_callback:
                try:
                    chunk, sr = self.audio_callback()
                    if chunk is not None:
                        self.process_chunk(chunk, sr)
                    else:
                        # No audio available, yield control
                        if self.debug and loop_count % 100 == 0:
                            print("No audio chunk available")
                        await asyncio.sleep(0.01)
                    loop_count += 1
                except Exception as e:
                    print(f"Error in detection loop: {e}")
                    import traceback

                    traceback.print_exc()
                    await asyncio.sleep(0.1)
            else:
                if self.debug and loop_count == 0:
                    print("No audio callback configured!")
                await asyncio.sleep(0.1)
                loop_count += 1

    def start_detection(self) -> asyncio.Task | None:
        """Start music detection as an async task and return it"""
        if self.detection_enabled:
            print("Detection already running")
            return self.detection_task

        # Start microphone if provided
        if self.microphone:
            self.microphone.start()

        self.detection_enabled = True
        self._reset_recording_state()

        self.detection_task = asyncio.create_task(self._detection_loop())
        print("Music detection started")
        return self.detection_task

    async def stop_detection(self) -> None:
        """Stop music detection and clean up"""
        if not self.detection_enabled:
            print("Detection not running")
            return

        self.detection_enabled = False

        if self.detection_task and not self.detection_task.done():
            self.detection_task.cancel()
            try:
                await self.detection_task
            except asyncio.CancelledError:
                pass

        # Stop microphone if we're managing it
        if self.microphone:
            self.microphone.stop()

        self._reset_recording_state()
        print("Music detection stopped")
