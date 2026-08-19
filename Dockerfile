FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN pip install --no-cache-dir chatterbox-tts runpod

# Baixa os pesos no BUILD (imagem maior, cold start rápido e sem dependência de rede)
RUN python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; ChatterboxMultilingualTTS.from_pretrained(device='cpu')" || true

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
