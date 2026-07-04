FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

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