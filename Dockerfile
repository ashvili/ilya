# База с CUDA и cuDNN, подходящая для PyTorch cu118
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Рабочая директория — КОРЕНЬ проекта (приложение будет стартовать отсюда)
WORKDIR /app

# Сначала только зависимости, чтобы кешировались слои
COPY requirements.txt /app/requirements.txt

# Ставим зависимости из репозитория (как просил)
RUN python3 -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt \
 # На всякий случай — явная установка PyTorch под CUDA 11.8, если не указан в requirements.txt
 && (pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 || true)

# Теперь — ВЕСЬ проект (включая обновлённый config.json из репозитория)
COPY . /app

# По умолчанию стартуем обучение; программа читает config.json из /app
CMD ["python3", "-m", "ml"]
