FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_MODEL_ID=Qwen/Qwen2-VL-2B-Instruct

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY inference.py .
COPY app.py .

EXPOSE 8080

# IMPORTANT: explicit python entrypoint
CMD ["python", "app.py"]