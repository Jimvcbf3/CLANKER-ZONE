#!/usr/bin/env python
import argparse
import os
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Qwen2VLImageProcessor


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def downscale_to_max_pixels(img: Image.Image, max_pixels: int = 800_000) -> Image.Image:
    w, h = img.size
    area = w * h
    if area <= max_pixels:
        return img
    scale = (max_pixels / float(area)) ** 0.5
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--max-new', type=int, default=256, dest='max_new')
    ap.add_argument('--min-new', type=int, default=32, dest='min_new')
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--no-clean', action='store_true')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4',
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        quantization_config=quant_cfg,
        device_map={'': 0} if device.type == 'cuda' else None,
        torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
        attn_implementation='sdpa',
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    image_processor = Qwen2VLImageProcessor.from_pretrained(args.model)

    if args.debug:
        try:
            print('[dbg] ids:', 'eos', tokenizer.eos_token_id, 'pad', tokenizer.pad_token_id)
            for tok in ['<|endofassistant|>', '<|endoftext|>', '<|imgpad|>']:
                print(f"[dbg] id[{tok}]:", tokenizer.convert_tokens_to_ids(tok))
        except Exception as e:
            print('[dbg] tokenizer id dbg failed:', e)

    img = downscale_to_max_pixels(load_image(args.image), max_pixels=800_000)

    # Build chat string using template semantics
    prompt_txt = args.prompt
    default_prompt = 'Read all visible text and return as plain text.'
    if prompt_txt.strip() == default_prompt.strip():
        prompt_txt += ' Return unique visible text; skip repeating navigation labels. Preserve reading order and line breaks.'
    chat = '<|user|><|img|><|imgpad|><|endofimg|>' + prompt_txt + '<|endofuser|><|assistant|>'

    toks = tokenizer(chat, return_tensors='pt')
    vision = image_processor(images=[img], return_tensors='pt')

    # Expand single image token to match visual tokens after merge
    image_token_id = getattr(model.config, 'image_token_id', tokenizer.convert_tokens_to_ids('<|imgpad|>'))
    input_ids = toks['input_ids']
    attn = toks.get('attention_mask', None)

    grid = vision['image_grid_thw'][0]
    t, gh, gw = int(grid[0]), int(grid[1]), int(grid[2])
    merge = int(getattr(model.config.vision_config, 'spatial_merge_size', 2))
    vision_len = int(t * (gh * gw) // (merge * merge))

    where = (input_ids == image_token_id).nonzero()
    if where.numel() == 0:
        raise RuntimeError('Image token id not found in input_ids; check chat template')
    pos = int(where[0,1])
    before = input_ids[:, :pos]
    after = input_ids[:, pos+1:]
    repeat = torch.full((1, vision_len), image_token_id, dtype=before.dtype)
    new_input_ids = torch.cat([before, repeat, after], dim=1)

    if attn is not None:
        new_attn = torch.cat([attn[:, :pos], torch.ones((1, vision_len), dtype=attn.dtype), attn[:, pos+1:]], dim=1)
    else:
        new_attn = None

    inputs = {
        'input_ids': new_input_ids.to(device),
        'attention_mask': (new_attn.to(device) if new_attn is not None else None),
        'pixel_values': vision['pixel_values'].to(device),
        'image_grid_thw': vision['image_grid_thw'].to(device),
    }

    if args.debug:
        expanded_count = (new_input_ids == image_token_id).sum().item()
        print(f"[dbg] expected vision_len: {vision_len}")
        print(f"[dbg] expanded image_token_id count: {expanded_count}")
        if expanded_count != vision_len:
            raise RuntimeError(f"image token expansion mismatch: expanded={expanded_count} expected={vision_len}")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_kwargs = dict(
        max_new_tokens=args.max_new,
        min_new_tokens=args.min_new,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=4,
        repetition_penalty=1.1,
        do_sample=False,
    )
    if args.sample:
        gen_kwargs.update(dict(do_sample=True, temperature=0.6, top_p=0.9))

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    input_len = inputs['input_ids'].shape[1]
    if args.debug:
        print(f"[dbg] output_ids.shape: {tuple(outputs.shape)} input_len: {input_len}")
        try:
            print(tokenizer.decode(outputs[0][:input_len], skip_special_tokens=False).strip())
            print(tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False).strip())
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


if __name__ == '__main__':
    main()
