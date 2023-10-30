FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    cython3 \
    python3-pip

ENV PYTHONUNBUFFERED=1
RUN python3 -m pip install pip --upgrade --no-cache-dir \
    && python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

WORKDIR /app
COPY . .

RUN python3 -m pip install -r requirements.txt --no-cache-dir

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]