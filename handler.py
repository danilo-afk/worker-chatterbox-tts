"""RunPod serverless worker — Chatterbox Multilingual TTS (clonagem zero-shot).

Contrato de input (todos opcionais exceto text/prompt):
  {
    "prompt" | "text": "texto a falar",
    "language_id": "pt",                # idioma (default pt)
    "audio_prompt": "<base64|data URI>" # amostra de voz p/ clonagem (opcional)
    "exaggeration": 0.5, "cfg_weight": 0.5
  }
Output: {"audio": "data:audio/wav;base64,...", "duration_seconds": float}
"""

import base64
import io
import os
import tempfile

import runpod

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = ChatterboxMultilingualTTS.from_pretrained(device=device)
    return _MODEL


def _decode_audio(ref: str) -> str:
    """base64/data URI → arquivo temporário; retorna o path."""
    b64 = ref.split("base64,", 1)[-1]
    raw = base64.b64decode(b64)
    suffix = ".mp3" if b"ID3" in raw[:16] or raw[:2] == b"\xff\xfb" else ".wav"
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(raw)
    f.close()
    return f.name


def handler(job):
    inp = job.get("input") or {}
    text = (inp.get("prompt") or inp.get("text") or "").strip()
    if not text:
        return {"error": "input.prompt (ou input.text) é obrigatório."}
    language = (inp.get("language_id") or inp.get("language") or "pt").strip()

    kwargs = {}
    ref = inp.get("audio_prompt") or inp.get("reference_audio") or inp.get("audio_url") or ""
    tmp = None
    if ref:
        tmp = _decode_audio(str(ref))
        kwargs["audio_prompt_path"] = tmp
    for k in ("exaggeration", "cfg_weight", "temperature"):
        if inp.get(k) is not None:
            kwargs[k] = float(inp[k])

    try:
        import torchaudio

        model = _model()
        wav = model.generate(text, language_id=language, **kwargs)
        buf = io.BytesIO()
        torchaudio.save(buf, wav, model.sr, format="wav")
        data = buf.getvalue()
        return {
            "audio": "data:audio/wav;base64," + base64.b64encode(data).decode(),
            "duration_seconds": round(wav.shape[-1] / model.sr, 2),
            "language_id": language,
            "cloned": bool(ref),
        }
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


runpod.serverless.start({"handler": handler})
