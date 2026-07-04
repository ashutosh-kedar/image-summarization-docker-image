import io
import os
import logging
import threading

import boto3
import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
S3_BUCKET = os.getenv("S3_BUCKET")  # set via create_model Environment

# Hard server-side cap so a single request can't run generation forever
# (InvokeEndpoint has a 60s limit — long generations will time out client-side).
MAX_NEW_TOKENS_CAP = int(os.getenv("MAX_NEW_TOKENS_CAP", "1024"))

ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

model = None
processor = None
s3 = boto3.client("s3")

# The GPU can only run one generation at a time safely on a T4;
# serialize requests instead of letting them OOM each other.
_generate_lock = threading.Lock()


def load_model():
    global model, processor

    if model is None:
        logger.info("Loading model: %s", MODEL_ID)
        logger.info(
            "torch=%s | cuda_available=%s | cuda_version=%s",
            torch.__version__,
            torch.cuda.is_available(),
            torch.version.cuda,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            logger.error(
                "CUDA NOT AVAILABLE — falling back to CPU. "
                "Inference will be ~10x slower and will time out. "
                "Check torch/driver compatibility in the image."
            )

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa",
        ).to(device)

        model.eval()

        # Cap visual tokens per image: without this, a full-res phone photo
        # becomes thousands of tokens and prefill alone can exceed the
        # 60s InvokeEndpoint limit on a T4.
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            min_pixels=int(os.getenv("MIN_PIXELS", 256 * 28 * 28)),
            max_pixels=int(os.getenv("MAX_PIXELS", 1280 * 28 * 28)),
        )

        logger.info("Model loaded on %s, warming up...", device)
        _warmup(device)
        logger.info("Warm-up complete")


def _warmup(device: str):
    """Run one tiny generation so CUDA kernel init / memory allocation
    happens during startup (covered by the health-check timeout) instead
    of on the first real user request."""
    try:
        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": "Hi"},
            ],
        }]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            return_tensors="pt", padding=True,
        ).to(device)
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=8)
    except Exception:
        logger.exception("Warm-up failed (continuing anyway)")


def fetch_images_from_s3(user_id: str, image_names: list) -> list:
    """Download images from s3://S3_BUCKET/<user_id>/<image_name> as PIL Images."""
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET environment variable is not set on the model")

    prefix = user_id.strip("/")
    images = []

    for name in image_names:
        # basename() prevents path traversal out of the user's folder
        clean_name = os.path.basename(str(name).strip())

        if not clean_name.lower().endswith(ALLOWED_EXTENSIONS):
            raise ValueError(f"Unsupported image type: {clean_name}")

        key = f"{prefix}/{clean_name}"
        logger.info("Fetching s3://%s/%s", S3_BUCKET, key)

        try:
            body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        except s3.exceptions.NoSuchKey:
            raise FileNotFoundError(f"Image not found: s3://{S3_BUCKET}/{key}")

        images.append(Image.open(io.BytesIO(body)).convert("RGB"))

    return images


def build_messages(chat_history: list, prompt: str, images: list) -> list:
    """
    Build a proper multi-turn message list for the Qwen chat template.

    chat_history: [{"role": "user"|"assistant", "content": "..."}, ...]
    Images are attached to the current (final) user turn.
    """
    messages = []

    for turn in chat_history or []:
        role = turn.get("role", "user")
        if role not in ("user", "assistant", "system"):
            role = "user"
        content = str(turn.get("content", "")).strip()
        if content:
            messages.append({
                "role": role,
                "content": [{"type": "text", "text": content}],
            })

    current_content = [{"type": "image", "image": img} for img in images]
    current_content.append({"type": "text", "text": prompt})

    messages.append({"role": "user", "content": current_content})
    return messages


def predict(payload: dict) -> dict:
    load_model()

    prompt = payload["prompt"]
    user_id = payload.get("user_id")
    image_names = payload.get("image_names") or []
    chat_history = payload.get("chat_history") or []

    images = []
    if image_names:
        if not user_id:
            raise ValueError("'user_id' is required when 'image_names' is provided")
        images = fetch_images_from_s3(user_id, image_names)

    messages = build_messages(chat_history, prompt, images)

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

    # Clamp generation length to the server-side cap
    max_new_tokens = min(
        int(payload.get("max_new_tokens", 256)),
        MAX_NEW_TOKENS_CAP,
    )

    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if payload.get("do_sample", False):
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(payload.get("temperature", 0.2))

    import time
    start = time.time()
    with _generate_lock:
        with torch.inference_mode():
            output = model.generate(**inputs, **gen_kwargs)
    logger.info(
        "Generation took %.1fs (input tokens=%d, max_new_tokens=%d, images=%d)",
        time.time() - start, inputs.input_ids.shape[1], max_new_tokens, len(images),
    )

    trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, output)
    ]

    result = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
    )[0]

    return {"generated_text": result}