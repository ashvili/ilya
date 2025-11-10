#!/usr/bin/env bash
set -euo pipefail

echo "== Обновление пакетов =="
sudo apt-get update -y

echo "== Установка утилит (curl, git, jq) =="
sudo apt-get install -y curl git jq

echo "== Установка Docker =="
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
else
  echo "-- Docker уже установлен"
fi
sudo usermod -aG docker "$USER" || true

echo "== NVIDIA Container Toolkit (GPU в Docker) =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Внимание: драйвер NVIDIA на хосте не найден (nvidia-smi недоступен). Убедись, что образ ВМ с предустановленным драйвером."
fi
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update -y
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "== Проверка GPU в Docker =="
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi || {
  echo "Ошибка: Docker не видит GPU. Проверь драйвер/образ ВМ и перезапусти."; exit 1;
}

echo "== Клонирование проекта =="
if [ -d ilya ]; then
  echo "-- Папка ./ilya уже есть, обновляю"
  (cd ilya && git pull --rebase)
else
  git clone https://github.com/ashvili/ilya.git
fi

cd ilya

# Проверим наличие config.json и requirements.txt в репозитории
[ -f config.json ] || { echo "Нет config.json в репо. Добавь в корень и запусти снова."; exit 1; }
[ -f requirements.txt ] || { echo "Нет requirements.txt в репо. Добавь и запусти снова."; exit 1; }

echo "== Обновляю batch_size в config.json для RTX 3080 10GB =="
# Если структура иная — скорректируй путь .ml.batch_size
jq '.ml.batch_size = 8192' config.json > config.tmp.json && mv config.tmp.json config.json

echo "== Сборка Docker-образа из репозитория (requirements.txt из репо) =="
docker build -t ilya-ml:latest .

echo
echo "=== ГОТОВО ==="
echo
echo "1) Пример запуска контейнера (из текущей папки — корня проекта):"
echo "   docker run --rm --gpus all -it -v \$(pwd):/app -w /app ilya-ml:latest bash"
echo
echo "2) Прямой запуск обучения (берёт /app/config.json):"
echo "   docker run --rm --gpus all -it -v \$(pwd):/app -w /app ilya-ml:latest python -m ml"
echo
echo "3) Если словишь OOM, уменьши батч и собери заново:"
echo "   jq '.ml.batch_size = 4096' config.json > config.tmp.json && mv config.tmp.json config.json"
echo "   docker build -t ilya-ml:latest ."
echo "   docker run --rm --gpus all -it -v \$(pwd):/app -w /app ilya-ml:latest python -m ml"
echo
echo "Важно: выйди и зайди в сессию/ssh, чтобы вступило в силу добавление в группу docker (или выполни 'newgrp docker')."
