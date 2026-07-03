import io
import os
import json
import base64

import torch

from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info

MODEL_ID = os.getenv(
    "HF_MODEL_ID",
    "Qwen/Qwen2-VL-2B-Instruct"
)


def model_fn(model_dir):

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    return {
        "model": model,
        "processor": processor,
    }


def input_fn(request_body, content_type):

    if content_type != "application/json":
        raise ValueError("Unsupported content type")

    payload = json.loads(request_body)

    return payload


def predict_fn(payload, artifacts):

    model = artifacts["model"]
    processor = artifacts["processor"]

    images = []

    for image_b64 in payload["images"]:

        image = Image.open(
            io.BytesIO(
                base64.b64decode(image_b64)
            )
        ).convert("RGB")

        images.append(image)

    content = []

    for image in images:

        content.append(
            {
                "type": "image",
                "image": image,
            }
        )

    content.append(
        {
            "type": "text",
            "text": payload["prompt"],
        }
    )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )

    inputs = inputs.to(model.device)

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=payload.get(
                "max_new_tokens",
                512,
            ),
            temperature=payload.get(
                "temperature",
                0.2,
            ),
            do_sample=payload.get(
                "do_sample",
                False,
            ),
        )

    generated_ids_trimmed = [
        out[len(inp):]
        for inp, out in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output


def output_fn(prediction, accept):

    return json.dumps(
        {
            "generated_text": prediction
        }
    ), "application/json"