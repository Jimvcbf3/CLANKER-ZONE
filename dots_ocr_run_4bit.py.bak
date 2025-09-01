# C:\ai-auto\dots_ocr_run_4bit.py
# Run the 4-bit dots.ocr using the official chat/processor path (no flash-attn required on Windows).
# Usage:
#   py -u C:\ai-auto\dots_ocr_run_4bit.py --image C:\ai-auto\test.png --prompt "Read all visible text and return as plain text." --max-new 256

import os, argparse
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# Use the tiny helper we just created (no repo install needed)
from qwen_vl_utils import process_vision_info

MODEL_ID = "helizac/dots.ocr-4bit"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new", type=int, default=256, dest="max_new")
    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    return ap.parse_args()

def main():
    args = parse_args()

    # Pick device/dtype
    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    device_map = "auto" if (args.device == "auto" and use_cuda) else None
    torch_dtype = torch.bfloat16 if use_cuda else torch.float32

    print(f"[load] model={MODEL_ID} dtype={torch_dtype} device_map={device_map or args.device}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)

    image_path = os.path.abspath(args.image)
    if not os.path.isfile(image_path):
        raise SystemExit(f"[ERR] image not found: {image_path}")

    # Build the chat messages the way the model expects
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text":  args.prompt},
        ],
    }]

    # Chat template + image tensors
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")

    # Move tensors to the model device
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    print("[gen ] generating…")
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new,
        # Quantized models benefit from anti-loop settings:
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.15,
    )

    # Trim the echoed prompt portion
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], gen_ids)]
    out_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    print("\n===== OCR OUTPUT =====")
    print(out_text.strip())
    print("======================\n")

if __name__ == "__main__":
    main()