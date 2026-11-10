"""Benchmark transcription engines/settings on ONE audio file, one config per
process (so memory numbers are clean). Prints load time, transcribe time,
realtime factor, word count and peak memory.

  python scripts/bench_transcription.py fw  clip.wav --threads 4
  python scripts/bench_transcription.py fw  clip.wav --threads 4 --beam 1 --temp0 --no-cond
  python scripts/bench_transcription.py fwb clip.wav --batch 8 --beam 1
  python scripts/bench_transcription.py mlx clip.wav
"""
import argparse
import resource
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("engine", choices=["fw", "fwb", "mlx"])
p.add_argument("audio")
p.add_argument("--model", default="small")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--beam", type=int, default=5)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--no-word-ts", action="store_true")
p.add_argument("--temp0", action="store_true", help="disable temperature-fallback re-decoding")
p.add_argument("--no-cond", action="store_true", help="condition_on_previous_text=False")
p.add_argument("--dump", help="write the transcript text to this file")
p.add_argument("--mlx-repo", default="mlx-community/whisper-small-mlx")
a = p.parse_args()

t0 = time.time()
words = 0
texts = []
if a.engine in ("fw", "fwb"):
    from faster_whisper import WhisperModel
    model = WhisperModel(a.model, device="cpu", compute_type="int8", cpu_threads=a.threads)
    kw = dict(beam_size=a.beam, word_timestamps=not a.no_word_ts)
    if a.temp0:
        kw["temperature"] = 0.0
    if a.no_cond:
        kw["condition_on_previous_text"] = False
    t1 = time.time()
    if a.engine == "fwb":
        from faster_whisper import BatchedInferencePipeline
        pipe = BatchedInferencePipeline(model=model)
        segs, info = pipe.transcribe(a.audio, batch_size=a.batch, **kw)
    else:
        segs, info = model.transcribe(a.audio, vad_filter=True,
                                      vad_parameters={"min_silence_duration_ms": 500}, **kw)
    for s in segs:
        texts.append(s.text.strip())
        words += len(s.words) if s.words else len(s.text.split())
    dur = info.duration
else:
    import mlx_whisper
    t1 = time.time()
    kw = {"word_timestamps": not a.no_word_ts, "path_or_hf_repo": a.mlx_repo}
    if a.no_cond:
        kw["condition_on_previous_text"] = False
    if a.temp0:
        kw["temperature"] = 0.0
    r = mlx_whisper.transcribe(a.audio, **kw)
    for s in r["segments"]:
        texts.append(s["text"].strip())
        words += len(s.get("words") or s["text"].split())
    dur = r["segments"][-1]["end"] if r["segments"] else 0
t2 = time.time()
if a.dump:
    open(a.dump, "w").write("\n".join(texts))
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # bytes on macOS
print(f"RESULT engine={a.engine} args={' '.join(sys.argv[3:])}")
print(f"  load={t1-t0:.1f}s transcribe={t2-t1:.1f}s audio={dur:.0f}s realtime_x={dur/max(t2-t1,0.01):.2f} words={words} peak_rss={rss:.0f}MB")
