import sys
import time

from faster_whisper import WhisperModel

threads = int(sys.argv[1]) if len(sys.argv) > 1 else 4
audio = sys.argv[2] if len(sys.argv) > 2 else "/tmp/bench_audio.wav"

t0 = time.time()
model = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=threads)
t1 = time.time()
segments, info = model.transcribe(
    audio, word_timestamps=True, vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 500},
)
text = ""
for seg in segments:
    text += seg.text
t2 = time.time()

print("THREADS=%d" % threads)
print("LOAD_S=%.2f" % (t1 - t0))
print("TRANSCRIBE_S=%.2f" % (t2 - t1))
print("CHARS=%d" % len(text))
