#!/usr/bin/env python
import argparse
import os
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Qwen2VLImageProcessor


def load_image(path):
    img = Image.open(path).convert("RGB")
    return img


def downscale_to_max_pixels(img: Image.Image, max_pixels: int = 800_000) -> Image.Image:
    # Maintain aspect ratio, reduce area to <= max_pixels
    w, h = img.size
    area = w * h
    if area <= max_pixels:
        return img
    scale = (max_pixels / float(area)) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC)


def build_messages(prompt: str, pil_image: Image.Image):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to local 4-bit model dir")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new", type=int, default=256, dest="max_new_tokens")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9, dest="top_p")
    parser.add_argument("--rp", type=float, default=1.15, dest="repetition_penalty")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", action="store_true", help="Enable sampling (default: greedy)")
    parser.add_argument("--min-new", type=int, default=32, dest="min_new", help="min_new_tokens (default: 32)")
    parser.add_argument("--no-clean", action="store_true", help="Disable OCR post-processing")
    parser.add_argument("--debug", action="store_true", help="Print debug info about expansion and decoding")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        quantization_config=quant_cfg,
        device_map={"": 0} if device.type == "cuda" else None,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    )
    model.eval()

        # Load and downscale image to keep visual tokens reasonable for 8GB VRAM
    pil = load_image(args.image)
    pil = downscale_to_max_pixels(pil, max_pixels=800_000)

    messages = build_messages(args.prompt, pil)

    # Tokenizer and image processor separately (avoid DotsVLProcessor video dependency)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    image_processor = Qwen2VLImageProcessor.from_pretrained(args.model)
    if args.debug:
        try:
            print('[dbg] ids:', 'eos', tokenizer.eos_token_id, 'pad', tokenizer.pad_token_id)
            for tok in ['<|endofassistant|>', '<|endoftext|>', '<|imgpad|>']:
                try:
                    print(f"[dbg] id[{tok}]:", tokenizer.convert_tokens_to_ids(tok))
                except Exception as e:
                    print(f"[dbg] id[{tok}]: lookup-failed:", e)
        except Exception as e:
            print('[dbg] tokenizer id dbg failed:', e)

    # Build chat text with template
    prompt_txt = args.prompt
    default_prompt = "Read all visible text and return as plain text."
    if prompt_txt.strip() == default_prompt.strip():
        prompt_txt += " Return unique visible text; skip repeating navigation labels. Preserve reading order and line breaks."
    text = "<|user|><|img|><|imgpad|><|endofimg|>" + prompt_txt + "<|endofuser|><|assistant|>"
    tokenized = tokenizer(text, return_tensors="pt")

    # Compute image features for model
    vision = image_processor(images=[pil], return_tensors="pt")

    # Expand image token to match number of vision tokens so img_mask aligns
    image_token_id = getattr(model.config, 'image_token_id', None)
    if image_token_id is None:
        # Fallback: resolve from tokenizer special tokens
        image_token_id = tokenizer.convert_tokens_to_ids('<|imgpad|>')
    input_ids = tokenized['input_ids']
    attn = tokenized.get('attention_mask', None)
    # vision token length after spatial merge
    grid = vision['image_grid_thw'][0]
    t, gh, gw = int(grid[0]), int(grid[1]), int(grid[2])
    merge = getattr(model.config.vision_config, 'spatial_merge_size', 2)
    vision_len = int(t * (gh * gw) // (merge * merge))
    # find the first occurrence of image token id
    where = (input_ids == image_token_id).nonzero()
    if where.numel() == 0:
        raise RuntimeError('Image token id not found in input_ids; check chat template')
    b, pos = int(where[0,0]), int(where[0,1])
    before = input_ids[:, :pos]
    after = input_ids[:, pos+1:]
    repeat = torch.full((1, vision_len), image_token_id, dtype=before.dtype)
    new_input_ids = torch.cat([before, repeat, after], dim=1)
    if attn is not None:
        attn_before = attn[:, :pos]
        attn_after = attn[:, pos+1:]
        attn_repeat = torch.ones((1, vision_len), dtype=attn.dtype)
        new_attn = torch.cat([attn_before, attn_repeat, attn_after], dim=1)
    else:
        new_attn = None
    tokenized['input_ids'] = new_input_ids
    if new_attn is not None:
        tokenized['attention_mask'] = new_attn

    inputs = {
        "input_ids": tokenized["input_ids"].to(device),
        "attention_mask": tokenized.get("attention_mask", None).to(device) if tokenized.get("attention_mask", None) is not None else None,
        "pixel_values": vision["pixel_values"].to(device),
        "image_grid_thw": vision["image_grid_thw"].to(device),
    }
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=4,
        repetition_penalty=1.1,
    )
    if args.sample:
        gen_kwargs.update(dict(do_sample=True, temperature=args.temperature, top_p=args.top_p))
    else:
        gen_kwargs.update(dict(do_sample=False))


    torch.manual_seed(args.seed)



    def _clean_ocr(text: str) -> str:
        import re
        from collections import Counter
        text = text.replace("", "")
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]

        cleaned, last, run = [], None, 0
        for ln in lines:
            if ln == last:
                run += 1
                if run <= 3:
                    cleaned.append(ln)
            else:
                last, run = ln, 1
                cleaned.append(ln)

        freq = Counter(cleaned)
        final, short_count = [], {}
        for ln in cleaned:
            if len(ln.split()) == 1 and len(ln) <= 5 and freq[ln] > 6:
                c = short_count.get(ln, 0) + 1
                short_count[ln] = c
                if c <= 3:
                    final.append(ln)
                continue
            final.append(ln)
        return "
".join(final)

    # Debug: verify expansion and shapes before decoding
    image_token_id = getattr(model.config, "image_token_id", tokenizer.convert_tokens_to_ids("<|imgpad|>"))
    expanded_count = (new_input_ids == image_token_id).sum().item()
    if args.debug:
        print(f"[dbg] expected vision_len: {vision_len}")
        print(f"[dbg] expanded image_token_id count: {expanded_count}")
        if expanded_count != vision_len:
            raise RuntimeError(f"image token expansion mismatch: expanded={expanded_count} expected={vision_len}")

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    if args.debug:
        print(f"[dbg] output_ids.shape: {tuple(outputs.shape)} input_len: {input_len}")
        try:
            prefix_dbg = tokenizer.decode(outputs[0][:input_len], skip_special_tokens=False)
            new_dbg = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)
            print(prefix_dbg.strip())
            print(new_dbg.strip())
        except Exception:
            pass

    new_tokens = outputs[0][input_len:] if outputs.shape[1] > input_len else outputs[0]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not text:
        full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = full.split(args.prompt, 1)[-1].strip() if args.prompt in full else full.strip()
    if not args.no_clean and text:
        text = _clean_ocr(text)
    print(text)


if __name__ == "__main__":
    main()
