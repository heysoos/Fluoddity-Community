from .save_frame_gpu import save_frame_gpu, reset_gpu_frame_counter
from .ffmpeg_recorder import FFmpegVideoRecorder
from .paths import get_videos_dir
from services.record_crop import record_sizes, world_crop
from services.record_view import RecordView
from datetime import datetime
from time import perf_counter

class VidSaver:
    def __init__(self):
        self.active = False
        self.current_frame = 0
        self.recorder = None
        self.ssk_w = 2
        # Set before a take by whoever wants its soundtrack; see
        # services/record_audio.RecordingAudio.
        self.audio = None
        self.output_path = None
        self.last_message = ""
        self._audio_on = False
        self._video_path = None
        self._started_at = 0.0
        # Built on the first frame of a take and held for its whole length -
        # the encoder cannot take a dimension change, so the size is frozen
        # even though the crop rect is re-derived every frame.
        self.view = None
        self._source_size = None

    def _plan(self, tex, ssk_w, view_rect):
        """(blit target size or None, video size) for a take starting now."""
        if view_rect is None:
            width, height = tex.size
            return None, (width // ssk_w, height // ssk_w)
        return record_sizes(world_crop(view_rect), tex.size, ssk_w)

    def frame(self, ctx, tex, max_frames=-1, ssk_w=2, filename_prefix="",
              view_rect=None):
        if not self.active:
            return

        fbo_size, (output_width, output_height) = self._plan(tex, ssk_w,
                                                             view_rect)

        # Restart on a settings change. A cropped take is immune to a window
        # resize, because its output size no longer tracks the window.
        source_changed = view_rect is None and self._source_size != tex.size
        if self.recorder is None or self.ssk_w != ssk_w or source_changed:
            if self.recorder is not None:
                print(f"WARNING: Recording settings changed mid-recording!")
                print(f"  Old: {self.recorder.input_width}x{self.recorder.input_height}, ssk={self.ssk_w}")
                print(f"  New: {output_width}x{output_height}, ssk={ssk_w}")
                print(f"  Finishing current video and starting new one...")
                self.recorder.close()
                self._release_view()

            # Create timestamped filename in Videos folder
            timestamp = datetime.now().strftime('%H-%M-%S')
            prefix = filename_prefix if filename_prefix else "animation"
            videos_dir = get_videos_dir()
            videos_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(videos_dir / f"{prefix}-{timestamp}.mp4")
            self.output_path = output_path

            # With a soundtrack the final name belongs to the muxed file, so
            # the encoder is sent somewhere else - it must not hold the name
            # the mux is about to write.
            self._audio_on = self.audio is not None and self.audio.start()
            self._video_path = (f"{output_path[:-4]}.video.mp4"
                                if self._audio_on else output_path)
            self._started_at = perf_counter()

            self.recorder = FFmpegVideoRecorder(
                width=output_width,
                height=output_height,
                fps=50,  # Default fps, can be made configurable
                output_path=self._video_path,
                realtime=False
            )
            if fbo_size is not None:
                self.view = RecordView(ctx, fbo_size)
            self.ssk_w = ssk_w
            self._source_size = tex.size
            self.current_frame = 0  # Reset frame counter for new recording

        # Trim to the world before readback, so empty space around the canvas
        # never reaches the file.
        if self.view is not None:
            tex = self.view.crop(tex, view_rect)

        # Process frame with GPU (spatial supersampling only)
        # Temporal accumulation and gamma correction happen in FrameAssembler before this
        # Use return_array=True to get numpy array instead of saving PNG
        frame_array = save_frame_gpu(tex, ctx, supersample_k=ssk_w, return_array=True)

        # frame_array is always returned (no accumulation delay)
        self.recorder.write_frame_from_array(frame_array)
        self.current_frame += 1

        if max_frames > 0 and self.current_frame >= max_frames:
            self.finish()

    def _release_view(self):
        if self.view is not None:
            self.view.release()
            self.view = None

    def finish(self):
        '''Save video and reset everything for another recording'''

        frames = self.current_frame
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None

        # After close(), so the file the mux reads is complete.
        if self._audio_on:
            self.last_message = self.audio.finish(
                self._video_path, self.output_path, frames,
                perf_counter() - self._started_at)
        self._audio_on = False
        self._video_path = None

        self._release_view()
        self._source_size = None
        reset_gpu_frame_counter()
        self.current_frame = 0
        self.active = False
