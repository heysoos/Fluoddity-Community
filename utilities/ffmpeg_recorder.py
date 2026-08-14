import queue
import subprocess
import threading
from datetime import datetime
import sys
import os
from pathlib import Path
from utilities.paths import get_videos_dir

# The encoder must drain faster than the sim produces, or the queue below only
# postpones the stall. `veryfast` does at 1080p where `slow` does not; see the
# recording caveats in CLAUDE.md.
PRESET = 'veryfast'
CRF = 20

# The queue absorbs BURSTS - a keyframe, a scene change - not a sustained
# deficit, so it is sized by memory rather than by seconds.
_QUEUE_BUDGET_BYTES = 256 << 20
_QUEUE_MIN, _QUEUE_MAX = 4, 32


def find_ffmpeg():
    """
    Find ffmpeg executable, checking bundled location first, then system PATH.

    Returns:
        str: Path to ffmpeg executable

    Raises:
        FileNotFoundError: If ffmpeg cannot be found
    """
    # Check if running as PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Running in a bundle - check for bundled ffmpeg
        bundle_dir = Path(sys._MEIPASS)
        ffmpeg_path = bundle_dir / 'ffmpeg.exe'
        if ffmpeg_path.exists():
            return str(ffmpeg_path)

    # Check system PATH
    import shutil
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path

    # Not found - provide helpful error message
    raise FileNotFoundError(
        "FFmpeg not found!\n\n"
        "Video recording requires FFmpeg. Please either:\n"
        "1. Install FFmpeg and add it to your system PATH, or\n"
        "2. Place ffmpeg.exe in the same folder as this application\n\n"
        "Download FFmpeg from: https://ffmpeg.org/download.html"
    )

# A take shorter than this cannot be judged by percentage alone: the tap and
# the first frame never start on the same instant.
_CLOCK_FLOOR_SECONDS = 0.25
_CLOCK_TOLERANCE = 0.10


def recording_fps(frame_count, audio_seconds, wall_seconds, nominal_fps):
    """(fps, warning) for a finished take.

    The soundtrack is the clock. The tap starts and stops with the recorder, so
    both streams cover the same wall interval and frames divided by audio
    seconds is exactly the rate that makes them the same length - sync falls
    out of the arithmetic rather than depending on a timer, and it rides the
    sound card's clock rather than the frame loop's.

    Wall time is kept only as a cross-check. If a device dies mid-take the
    audio is far too short, and the derived rate would silently speed the video
    up to match it.
    """
    if frame_count <= 0:
        return nominal_fps, ""
    if audio_seconds <= 0.0:
        return nominal_fps, ""          # silent take: unchanged behaviour
    slack = max(_CLOCK_FLOOR_SECONDS, _CLOCK_TOLERANCE * wall_seconds)
    if wall_seconds > 0.0 and abs(audio_seconds - wall_seconds) > slack:
        return (frame_count / wall_seconds,
                f"Audio ran {audio_seconds:.1f}s against {wall_seconds:.1f}s of "
                f"recording, so the soundtrack may drift - the capture device "
                f"probably stopped early.")
    return frame_count / audio_seconds, ""


def mux_command(ffmpeg, video_path, audio_path, out_path, itsscale,
                sample_rate, channels):
    """argv muxing the soundtrack onto a finished take and retiming it.

    -itsscale is an input option and applies to the input it PRECEDES, so its
    position is load-bearing. The video is stream-copied: the pixels are
    already right and a second encode would only cost a generation.
    """
    return [
        ffmpeg, '-y',
        '-itsscale', f'{itsscale:.9f}',
        '-i', str(video_path),
        '-f', 'f32le', '-ar', str(int(sample_rate)), '-ac', str(int(channels)),
        '-i', str(audio_path),
        '-c:v', 'copy',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest',
        str(out_path),
    ]


class FFmpegVideoRecorder:
    """
    Video recorder that pipes frames directly to ffmpeg without intermediate PNG files.
    More efficient than the old PNG-based approach.

    Video Quality Settings:
    - CRF (Constant Rate Factor): Controls quality/file-size tradeoff
      * Range: 0-51 (lower = better quality, larger file)
      * 18 = visually lossless for most content (high quality mode)
      * 23 = default/good balance (realtime mode)
      * 28 = acceptable quality for previews
      * Each +6 roughly doubles file size

    - Preset: Controls encoding speed vs compression efficiency
      * ultrafast, superfast, veryfast, faster, fast (realtime default)
      * medium (default preset)
      * slow (high quality default), slower, veryslow
      * Slower presets = better compression (smaller files) at same quality

    To increase quality: Lower CRF (e.g., 15) and/or use slower preset (e.g., 'veryslow')
    To decrease quality/size: Raise CRF (e.g., 28) and/or use faster preset (e.g., 'veryfast')
    """

    def __init__(self, width, height, fps=50, output_path=None, realtime=True,
                 debug_log=False, preset=PRESET, crf=CRF):
        """
        Initialize ffmpeg video recorder.

        Frames arrive as raw RGBA bytes and are written by a BACKGROUND THREAD:
        `stdin.write` blocks whenever the pipe is full, and on the caller's
        thread that means the encoder's pace sets the caller's.

        Args:
            width: Video width in pixels. Must be even - VidSaver._plan already
                   rounds down, since H.264 will not take odd.
            height: Video height in pixels. Must be even.
            fps: Frames per second. Nominal only when the take has a
                 soundtrack; recording_fps() derives the real rate at the mux.
            output_path: Path for output video. If None, generates timestamped filename
            realtime: Unused, kept so old callers still construct. Quality is
                      `preset`/`crf`.
            debug_log: If True, save ffmpeg output to a log file for debugging
        """
        if width % 2 or height % 2:
            raise ValueError(
                f"H.264 needs even dimensions, got {width}x{height}. VidSaver "
                f"rounds down; nothing should reach here odd.")

        # Store original dimensions
        self.input_width = width
        self.input_height = height
        self.width = width
        self.height = height

        self.fps = fps
        self.frame_bytes = width * height * 4

        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime('%H-%M-%S')
            videos_dir = get_videos_dir()
            videos_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(videos_dir / f"animation-{timestamp}.mp4")

        self.output_path = output_path

        # Find ffmpeg executable (bundled or system PATH)
        ffmpeg_cmd = find_ffmpeg()

        # Start ffmpeg process
        # Optionally capture ffmpeg output to log file for debugging
        if debug_log:
            import os
            self.stderr_log_path = output_path.replace('.mp4', '_ffmpeg.log')
            self.stderr_log = open(self.stderr_log_path, 'w')
            stderr_dest = self.stderr_log
        else:
            self.stderr_log = None
            self.stderr_log_path = None
            stderr_dest = subprocess.DEVNULL

        self.ffmpeg = subprocess.Popen([
            ffmpeg_cmd, '-y',  # Overwrite output file
            '-f', 'rawvideo',
            # rgba, not rgb24: it is what the GPU wrote, so no host pass has
            # to drop the alpha. ffmpeg's own conversion is part of a job it is
            # already doing.
            '-pixel_format', 'rgba',
            '-video_size', f'{self.width}x{self.height}',
            '-framerate', str(fps),
            '-i', '-',  # Read from stdin
            '-c:v', 'libx264',
            '-preset', preset,
            '-crf', str(crf),
            '-pix_fmt', 'yuv420p',  # Compatible with most players
            output_path
        ], stdin=subprocess.PIPE, stderr=stderr_dest, stdout=subprocess.DEVNULL)

        self.frame_count = 0
        depth = max(_QUEUE_MIN,
                    min(_QUEUE_MAX, _QUEUE_BUDGET_BYTES // self.frame_bytes))
        # Bounded and BLOCKING: a full queue slows the sim rather than dropping
        # a frame, so the video is always exactly what the sim produced.
        self._queue = queue.Queue(maxsize=depth)
        self._error = None
        self._writer = threading.Thread(target=self._drain, name="ffmpeg-write",
                                        daemon=True)
        self._writer.start()

    def _drain(self):
        """Pump the queue into ffmpeg. Owns stdin; nothing else may write it."""
        while True:
            frame = self._queue.get()
            try:
                if frame is None:
                    return
                self.ffmpeg.stdin.write(frame)
            except (BrokenPipeError, OSError) as exc:
                # Kept for the next write_frame or for close() to raise on the
                # caller's thread, where it can reach the user.
                self._error = exc
                return
            finally:
                self._queue.task_done()

    def _ffmpeg_died_message(self):
        error_msg = (f"FFmpeg process has terminated unexpectedly with return "
                     f"code {self.ffmpeg.returncode}")
        if self.stderr_log:
            self.stderr_log.flush()
        if self.stderr_log_path:
            try:
                with open(self.stderr_log_path, 'r') as f:
                    log_contents = f.read()
                    if log_contents:
                        error_msg += f"\n\nFFmpeg error log:\n{log_contents}"
            except OSError:
                pass
        else:
            error_msg += ("\n\n(Enable debug_log=True to see detailed ffmpeg "
                          "output)")
        return error_msg

    def _check_writer(self):
        """Raise on the CALLER's thread for anything the writer hit."""
        if self._error is not None:
            raise RuntimeError(
                f"Failed to write frame to ffmpeg: {self._error}. "
                f"The ffmpeg process may have crashed or the pipe is broken."
            ) from self._error
        if not self._writer.is_alive():
            raise RuntimeError(self._ffmpeg_died_message())

    def write_frame(self, frame_bytes):
        """Queue one frame of raw RGBA for the writer thread.

        Blocks only once the queue is full, which means the encoder is behind
        by the whole budget - the sim slows rather than losing a frame.
        """
        if len(frame_bytes) != self.frame_bytes:
            raise ValueError(
                f"Frame is {len(frame_bytes)} bytes, expected "
                f"{self.frame_bytes} for {self.input_width}x"
                f"{self.input_height} RGBA. Did the window size change during "
                f"recording?"
            )
        if self.ffmpeg.poll() is not None:
            raise RuntimeError(self._ffmpeg_died_message())
        self._check_writer()
        self._queue.put(frame_bytes)
        self.frame_count += 1

    def close(self):
        """Close the video file and finish encoding.

        The backlog is drained first: everything already queued belongs in the
        file, and this is the one place it is right to wait for the encoder.
        """
        if self._writer.is_alive():
            self._queue.put(None)
            self._writer.join(timeout=max(15.0, self.frame_count / 20.0))
            if self._writer.is_alive():
                print("Writer thread did not finish; the video may be short.")

        if self.ffmpeg.stdin:
            try:
                # Flush any buffered data before closing
                self.ffmpeg.stdin.flush()
                self.ffmpeg.stdin.close()
            except Exception as e:
                print(f"Error closing ffmpeg stdin: {e}")

        # Wait for ffmpeg to finish with a reasonable timeout
        # Give it more time if we have a lot of frames
        timeout = max(15.0, self.frame_count / 100.0)  # At least 15s, or 1s per 100 frames
        try:
            returncode = self.ffmpeg.wait(timeout=timeout)

            # Close stderr log if it exists
            if self.stderr_log:
                self.stderr_log.close()

            if returncode == 0:
                print(f"Video saved successfully: {self.output_path}")
                print(f"Total frames: {self.frame_count}")
            else:
                print(f"FFmpeg encoding completed with return code {returncode}")
                # Print the log file if there was an error and logging is enabled
                if self.stderr_log_path:
                    try:
                        with open(self.stderr_log_path, 'r') as f:
                            log_contents = f.read()
                            if log_contents:
                                print(f"FFmpeg error log:\n{log_contents}")
                    except:
                        pass

        except subprocess.TimeoutExpired:
            print(f"FFmpeg did not finish within timeout, forcing termination...")
            self.ffmpeg.terminate()
            try:
                self.ffmpeg.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                print(f"Force killing ffmpeg...")
                self.ffmpeg.kill()
                self.ffmpeg.wait()
            print(f"Video may be incomplete: {self.output_path}")

            # Close stderr log if it exists
            if self.stderr_log:
                self.stderr_log.close()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()
