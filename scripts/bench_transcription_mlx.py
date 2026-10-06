import sys
import time

import mlx_whisper

audio = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench_audio.wav"
# mlx-community hosts ready-to-use MLX-converted Whisper weights on HF Hub;
# "small" here maps to the same underlying Whisper small model used by
# faster-whisper, just converted to MLX's weight format.
model_repo = "mlx-community/whisper-small-mlx"

t0 = time.time()
result = mlx_whisper.transcribe(
    audio, path_or_hf_repo=model_repo, word_timestamps=True,
)
t1 = time.time()

text = "".join(seg.get("text", "") for seg in result.get("segments", []))
print("TRANSCRIBE_S=%.2f" % (t1 - t0))
print("CHARS=%d" % len(text))
