"""What one recorded frame costs the frame loop, and what ffmpeg can drain.

Two independent limits, and a fix to either alone leaves the other in place.
The LOOP column is what the sim pays per frame; the DRAIN table is the ceiling
the encoder imposes no matter how cheap the loop gets - a queue in between only
postpones a deficit, it cannot absorb one.

Run it after touching the readback format, the shader, the preset or the queue.

    python -m tools.measure_recording
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

import ui  # noqa: F401  - imported first, or services/ hits a circular import

W, H = 1920, 1080
FRAMES = 150

PRESETS = ("slow", "medium", "fast", "veryfast", "ultrafast")


def drain_rate(preset, crf=20, n=90):
    """(frames/s ffmpeg takes off a blocking pipe, worst single write in ms)."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        return None, None
    p = subprocess.Popen([
        exe, "-y", "-f", "rawvideo", "-pixel_format", "rgba",
        "-video_size", f"{W}x{H}", "-framerate", "50", "-i", "-",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-f", "null", "-",
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL)
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (H, W, 4), dtype=np.uint8)
    frames = [np.roll(base, i * 7, axis=1).tobytes() for i in range(4)]
    worst, t0 = 0.0, time.perf_counter()
    for i in range(n):
        t = time.perf_counter()
        p.stdin.write(frames[i % 4])
        worst = max(worst, time.perf_counter() - t)
    dt = time.perf_counter() - t0
    p.stdin.close()
    p.wait()
    return n / dt, worst * 1e3


def loop_cost():
    """Milliseconds VidSaver.frame() takes on the caller's thread."""
    import moderngl

    from utilities import vid_saver as vs_mod
    from utilities.vid_saver import VidSaver

    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        print(f"  no GL 4.3 context ({exc}); skipping the loop column")
        return None, None

    tmp = Path(tempfile.mkdtemp(prefix="fluod_measure_"))
    real_dir, vs_mod.get_videos_dir = vs_mod.get_videos_dir, lambda: tmp
    src = ctx.texture((W, H), 4, dtype="f4")
    fbo = ctx.framebuffer(color_attachments=[src])
    try:
        vs = VidSaver()
        vs.active = True
        per_frame = []
        for i in range(FRAMES):
            fbo.use()
            fbo.clear((i % 30) / 30.0, 0.4, 1.0 - (i % 30) / 30.0, 1.0)
            t = time.perf_counter()
            vs.frame(ctx, src, max_frames=-1, ssk_w=1,
                     filename_prefix="measure")
            per_frame.append((time.perf_counter() - t) * 1e3)
        vs.finish()
        # Frame 0 spawns ffmpeg, which is a one-off and not a per-frame cost.
        return ctx.info["GL_RENDERER"], np.asarray(per_frame[1:])
    finally:
        vs_mod.get_videos_dir = real_dir
        for obj in (fbo, src):
            obj.release()
        ctx.release()
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    from utilities.ffmpeg_recorder import CRF, PRESET

    print(f"{W}x{H}, supersample_k=1 (the default, so this is window size)\n")

    renderer, a = loop_cost()
    if a is not None:
        print(f"=== what VidSaver.frame() costs the frame loop ({renderer}) ===")
        print(f"  mean {a.mean():6.2f}   median {np.median(a):6.2f}   "
              f"p99 {np.percentile(a, 99):6.2f}   worst {a.max():6.2f} ms")
        print(f"  ceiling from this stage alone: {1000 / a.mean():.0f} fps")

    print(f"\n=== what ffmpeg drains off a blocking pipe, rgba in ===")
    print(f"{'preset':>12} | {'frames/s':>9} {'worst write':>13}")
    print("-" * 40)
    for preset in PRESETS:
        fps, worst = drain_rate(preset, CRF)
        if fps is None:
            print("  ffmpeg not on PATH")
            break
        tag = "   <- shipped" if preset == PRESET else ""
        print(f"{preset:>12} | {fps:>9.1f} {worst:>10.1f} ms{tag}")

    print("\nA preset below the sim's framerate is a deficit, and the queue "
          "only\ndelays it. Noise compresses worse than real material, so "
          "these are floors.")


if __name__ == "__main__":
    main()
