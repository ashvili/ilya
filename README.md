# neurolithology - ml

## Automatic install in the cloud
(in the UBuntu server with NVIDIA driver)

### 1. Install

```bash
curl -O https://raw.githubusercontent.com/yourpaste/setup.sh && bash setup.sh
# или скопируй сюда setup.sh и Dockerfile как есть, затем:
chmod +x setup.sh
./setup.sh

```

### 2. Run

```bash
docker run --rm --gpus all -it -v $(pwd)/ilya:/app -w /app ilya-ml:latest python -m ml
```

P.S.
THis instuction for use GPU RTX 3080 (10 GB) - checnged batch-size to 8192


