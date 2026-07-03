import io
import os
import base64
import logging

import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO)

MODEL_ID = os.getenv(
    "HF_MODEL_ID",
    "Qwen/Qwen2-VL-2B-Instruct"
)

model = None
processor = None


def load_model():
    global model, processor

    if model is None:
        logging.info(f"Loading model: {MODEL_ID}")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)

        model.eval()

        processor = AutoProcessor.from_pretrained(MODEL_ID)

        logging.info(f"Model loaded on {device}")


def decode_images(image_list):
    images = []

    for image_b64 in image_list:
        image = Image.open(
            io.BytesIO(base64.b64decode(image_b64))
        ).convert("RGB")

        images.append(image)

    return images


def predict(payload):
    load_model()

    prompt = payload["prompt"]
    images = decode_images(payload["images"])

    content = []

    for image in images:
        content.append({
            "type": "image",
            "image": image,
        })

    content.append({
        "type": "text",
        "text": prompt,
    })

    messages = [{
        "role": "user",
        "content": content,
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    device = next(model.parameters()).device

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=payload.get("max_new_tokens", 512),
            temperature=payload.get("temperature", 0.2),
            do_sample=payload.get("do_sample", False),
        )

    generated_ids_trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return {
        "generated_text": response
    }