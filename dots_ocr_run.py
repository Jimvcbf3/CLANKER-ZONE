#!/usr/bin/env python
import argparse
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLImageProcessor


def load_image(path):
    return Image.open(path).convert('RGB')


def downscale_to_max_pixels(img: Image.Image, max_pixels: int = 1_200_000) -> Image.Image:
    w, h = img.size
    area = w * h
    if area <= max_pixels:
        return img
    scale = (max_pixels / float(area)) ** 0.5
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--prompt', type=str, required=True)
    ap.add_argument('--max-new', type=int, default=256, dest='max_new_tokens')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map={'': 0} if device.type == 'cuda' else None,
        torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
        attn_implementation='sdpa',
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    image_processor = Qwen2VLImageProcessor.from_pretrained(args.model)

    img = downscale_to_max_pixels(load_image(args.image), max_pixels=800_000)

    text = "<|user|><|img|><|imgpad|><|endofimg|>" + args.prompt + "<|endofuser|><|assistant|>"
    toks = tokenizer(text, return_tensors='pt')
    vision = image_processor(images=[img], return_tensors='pt')

    image_token_id = getattr(model.config, 'image_token_id', tokenizer.convert_tokens_to_ids('<|imgpad|>'))
    input_ids = toks['input_ids']
    attn = toks.get('attention_mask')

    t, gh, gw = map(int, vision['image_grid_thw'][0].tolist())
    merge = getattr(model.config.vision_config, 'spatial_merge_size', 2)
    vision_len = int(t * (gh * gw) // (merge * merge))

    where = (input_ids == image_token_id).nonzero()
    if where.numel() == 0:
        raise RuntimeError('Image token id not found in input_ids')
    pos = int(where[0,1])
    before = input_ids[:, :pos]
    after = input_ids[:, pos+1:]
    repeat = torch.full((1, vision_len), image_token_id, dtype=before.dtype)
    new_input_ids = torch.cat([before, repeat, after], dim=1)
    if attn is not None:
        new_attn = torch.cat([attn[:, :pos], torch.ones((1, vision_len), dtype=attn.dtype), attn[:, pos+1:]], dim=1)
    else:
        new_attn = None

    gen_kwargs = dict(
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.15,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        out = model.generate(
            input_ids=new_input_ids.to(device),
            attention_mask=(new_attn.to(device) if new_attn is not None else None),
            pixel_values=vision['pixel_values'].to(device),
            image_grid_thw=vision['image_grid_thw'].to(device),
            **gen_kwargs,
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    # Remove prompt if echoed
    if args.prompt in decoded:
        decoded = decoded.split(args.prompt, 1)[-1].strip()
    print(decoded.strip())


if __name__ == '__main__':
    main()
