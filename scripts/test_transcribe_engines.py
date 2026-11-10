"""Offline checks for transcribe.py's engine selection, fast-decoding kwargs and
MLX result mapping, using fake faster_whisper / mlx_whisper modules (no model
downloads, no ffmpeg needed)."""
import sys, types, tempfile
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app.pipeline.transcribe as T


def test_resolve():
    assert T.resolve_engine("faster-whisper") == "faster-whisper"
    assert T.resolve_engine("auto", "cuda") == "faster-whisper"
    T._mlx_available = lambda: True
    assert T.resolve_engine("auto") == "mlx" and T.resolve_engine("mlx") == "mlx"
    T._mlx_available = lambda: False
    assert T.resolve_engine("auto") == "faster-whisper" and T.resolve_engine("mlx") == "faster-whisper"


def fake_fw():
    calls = {}
    seg = NS(start=0.0, end=2.0, text=" hi there", words=[NS(word=" hi", start=0, end=1, probability=.9), NS(word=" there", start=1, end=2, probability=.8)])
    info = NS(language="en", duration=2.0)

    class Model:
        def __init__(self, *a, **k): pass
        def transcribe(self, path, **kw): calls["plain"] = kw; return iter([seg]), info

    class Batched:
        def __init__(self, model): pass
        def transcribe(self, path, **kw): calls["batched"] = kw; return iter([seg]), info

    m = types.ModuleType("faster_whisper"); m.WhisperModel = Model; m.BatchedInferencePipeline = Batched
    sys.modules["faster_whisper"] = m
    return calls


def test_faster_whisper_fast_settings():
    calls = fake_fw()
    T._MODEL_CACHE.clear()
    for batched, key in ((True, "batched"), (False, "plain")):
        tr = T.transcribe_video("x.mp4", engine="faster-whisper", batched=batched)
        kw = calls[key]
        assert kw["beam_size"] == 1 and kw["temperature"] == 0.0 and kw["condition_on_previous_text"] is False and kw["word_timestamps"]
        assert [w.word for w in tr.flat_words()] == ["hi", "there"] and tr.language == "en"
    assert calls["batched"]["batch_size"] == 8


def test_mlx_mapping_and_fallback():
    seen = {}

    def fake_transcribe(path, **kw):
        seen.update(kw)
        return {"language": "hi", "segments": [
            {"start": 0, "end": 1.5, "text": " namaste", "words": [{"word": " namaste", "start": 0, "end": 1.5, "probability": .7}]},
            {"start": 1.5, "end": 2, "text": " ", "words": []},
        ]}

    m = types.ModuleType("mlx_whisper"); m.transcribe = fake_transcribe
    sys.modules["mlx_whisper"] = m
    T._mlx_available = lambda: True
    T._extract_wav = lambda v, o: Path(o).write_bytes(b"")
    tr = T._transcribe_mlx("x.mp4", "small", None, None, None)
    assert seen["path_or_hf_repo"] == "mlx-community/whisper-small-mlx"
    assert seen["condition_on_previous_text"] is False and seen["temperature"] == 0.0
    assert len(tr.segments) == 1 and tr.language == "hi" and tr.flat_words()[0].word == "namaste"

    # MLX blowing up must fall back to faster-whisper, not fail the job.
    def boom(*a, **k): raise OSError("offline")
    m.transcribe = boom
    fake_fw(); T._MODEL_CACHE.clear()
    tr = T.transcribe_video("x.mp4", engine="auto")
    assert [w.word for w in tr.flat_words()] == ["hi", "there"]


def test_cache_roundtrip():
    fake_fw(); T._MODEL_CACHE.clear()
    with tempfile.TemporaryDirectory() as d:
        c = Path(d) / "t.json"
        a = T.transcribe_video("x.mp4", engine="faster-whisper", cache_path=c)
        sys.modules["faster_whisper"].WhisperModel = None  # would crash if re-run
        b = T.transcribe_video("x.mp4", engine="faster-whisper", cache_path=c)
        assert a.to_json() == b.to_json()


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
    print("ALL TRANSCRIBE ENGINE CHECKS PASSED")
