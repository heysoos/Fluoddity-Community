import os
import tempfile
from itertools import count

from utilities.vid_saver import VidSaver


class VideoRecorderService:
    """Wraps VidSaver, provides clean interface for Orchestrator."""

    _takes = count()

    def __init__(self):
        self.recorder = VidSaver()
        self._capture = None
        self._record_audio = False

    def configure(self, capture, record_audio: bool) -> None:
        """Push the soundtrack choice, which start() reads.

        Recording begins from the toggle and from the scheduled-start check, so
        the choice is pushed every frame rather than passed at either.
        """
        self._capture = capture
        self._record_audio = bool(record_audio)

    def is_active(self) -> bool:
        """Check if recording is active."""
        return self.recorder.active

    def start(self) -> None:
        """Start recording."""
        if self.recorder.active:
            return
        self.recorder.audio = self._build_audio()
        self.recorder.active = True

    def _build_audio(self):
        """A soundtrack for the take about to start, or None."""
        if not self._record_audio or self._capture is None:
            return None
        from services.record_audio import RecordingAudio
        # Each take gets its own name, or a second one would overwrite the
        # sidecar the first is still being muxed from.
        sidecar = os.path.join(tempfile.gettempdir(),
                               f"fluoddity-take-{next(self._takes)}.pcm")
        return RecordingAudio(self._capture, sidecar)

    def stop(self) -> None:
        """Stop recording and save video."""
        if self.recorder.active:
            self.recorder.finish()
        self.recorder.audio = None

    def take_message(self) -> str:
        """What the last take had to say, consumed once."""
        message, self.recorder.last_message = self.recorder.last_message, ""
        return message

    def toggle(self) -> None:
        """Toggle recording state."""
        if self.recorder.active:
            self.stop()
        else:
            self.start()

    def process_frame(self, ctx, texture, max_frames: int, ssk_w: int,
                      filename_prefix: str = "", view_rect=None) -> None:
        """Process a frame if recording is active.

        Args:
            texture: Already assembled and gamma-corrected texture
            max_frames: Maximum frames to record
            ssk_w: Spatial supersample kernel width
            filename_prefix: Custom filename prefix (empty = use "animation")
            view_rect: The canvas rect recorded with this texture. The frame is
                trimmed to it, so empty space around the world never reaches
                the file. None records the whole texture.
        """
        self.recorder.frame(ctx, texture, max_frames, ssk_w, filename_prefix,
                            view_rect)

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.recorder.active:
            self.recorder.finish()
