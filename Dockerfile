FROM 763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker

ENV PYTHONUNBUFFERED=1
ENV HF_MODEL_ID=Qwen/Qwen2-VL-2B-Instruct

WORKDIR /opt/ml/code

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py .

ENV SAGEMAKER_PROGRAM=inference.py
