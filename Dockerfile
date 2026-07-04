# CUDA 11.8 build — required because the SageMaker g4dn host driver
# only supports up to CUDA 11.4-era driver APIs. CUDA 12.x torch wheels
# (e.g. 2.6.0-cuda12.4) fail cuda_available and silently fall back to CPU.
# CUDA 11.x runtimes are minor-version compatible with this driver.
# torch must be >= 2.5 because transformers 4.52.x crashes on older torch
# (ALL_PARALLEL_STYLES is None -> "'NoneType' is not iterable" in post_init).
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_MODEL_ID=Qwen/Qwen2-VL-2B-Instruct

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Bake the model into the image (deterministic startup, no runtime download)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-VL-2B-Instruct')"

COPY inference.py .
COPY app.py .

EXPOSE 8080

ENTRYPOINT ["python", "app.py"]