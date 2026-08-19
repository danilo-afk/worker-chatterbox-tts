# cu128/torch2.7 = suporta Blackwell sm_120 (RunPod aloca PRO6000 MIG no tier 24GB)
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir chatterbox-tts runpod

# Baixa os pesos no BUILD (imagem maior, cold start rápido e sem dependência de rede)
RUN python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; ChatterboxMultilingualTTS.from_pretrained(device='cpu')" || true

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
