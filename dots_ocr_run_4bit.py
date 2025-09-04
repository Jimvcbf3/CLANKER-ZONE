#!/usr/bin/env python
import argparse
from typing import List, Tuple
import re
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Qwen2VLImageProcessor
from ocr_cleaner import clean as _clean_ocr

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')

def downscale_to_max_pixels(img: Image.Image, max_pixels: int = 800_000) -> Image.Image:
    w, h = img.size
    area = w * h
    if area <= max_pixels:
        return img
    scale = (max_pixels / float(area)) ** 0.5
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)

def _parse_tile_size(spec: str) -> Tuple[int, int]:
    s = (spec or "").strip().lower()
    if not s or s in {"0", "none", "off"}:
        return (0, 0)
    if "x" in s:
        a, b = s.split("x", 1)
        try:
            return (max(0, int(a)), max(0, int(b)))
        except Exception:
            return (0, 0)
    try:
        v = max(0, int(s))
        return (v, v)
    except Exception:
        return (0, 0)


def _tile_image(img: Image.Image, tile_w: int, tile_h: int, overlap: int = 0) -> List[Image.Image]:
    if tile_w <= 0 or tile_h <= 0:
        return [img]
    w, h = img.size
    if tile_w >= w and tile_h >= h:
        return [img]
    tiles: List[Image.Image] = []
    # Special-case: enforce exactly 2 vertical tiles if height fits in one tile
    if tile_h >= h and tile_w < w and tile_w * 2 >= w:
        starts = [0, max(0, w - tile_w)]
        for x in starts:
            x2 = min(w, x + tile_w)
            tiles.append(img.crop((x, 0, x2, h)))
        return tiles

    x = 0
    while x < w:
        y = 0
        x2 = min(w, x + tile_w)
        while y < h:
            y2 = min(h, y + tile_h)
            tiles.append(img.crop((x, y, x2, y2)))
            if y2 >= h:
                break
            y += max(1, tile_h - overlap)
        if x2 >= w:
            break
        x += max(1, tile_w - overlap)
    return tiles


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
    ap.add_argument('--max-pixels', type=int, default=None)
    ap.add_argument('--beams', type=int, default=1)
    # tiling + decoding knobs
    ap.add_argument('--tile', action='store_true')
    ap.add_argument('--tile-size', type=str, default='0')
    ap.add_argument('--tile-overlap', type=int, default=96)
    # extra decoding controls (keep same defaults)
    ap.add_argument('--no-repeat-ngram-size', type=int, default=4, dest='no_repeat_ngram_size')
    ap.add_argument('--repetition-penalty', type=float, default=1.1, dest='repetition_penalty')
    ap.add_argument('--top-k', type=int, default=50)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True

    quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type='nf4')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, quantization_config=quant_cfg,
        device_map={'': 0} if device.type == 'cuda' else None,
        torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
        attn_implementation='sdpa',
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    image_processor = Qwen2VLImageProcessor.from_pretrained(args.model)

    # Print tokenizer ids when --debug
    if args.debug:
        try:
            print('[dbg] ids:', 'eos', tokenizer.eos_token_id, 'pad', tokenizer.pad_token_id)
            for tok in ['<|endofassistant|>', '<|endoftext|>', '<|imgpad|>']:
                print(f"[dbg] id[{tok}]:", tokenizer.convert_tokens_to_ids(tok))
        except Exception as e:
            print('[dbg] tokenizer id dbg failed:', e)

    img = load_image(args.image)
    if args.max_pixels:
        from PIL import Image
        w,h = img.size
        area = w*h
        if area>args.max_pixels:
            scale = (args.max_pixels/float(area))**0.5
            img = img.resize((max(1,int(w*scale)), max(1,int(h*scale))), Image.BICUBIC)

    def run_single(tile_img, prompt: str, *, beams: int, sample: bool, min_new_override: int | None = None):
        # Build chat string per tile
        prompt_txt = prompt
        default_prompt = 'Read all visible text and return as plain text.'
        if prompt_txt.strip() == default_prompt.strip():
            prompt_txt += ' Return unique visible text; skip repeating navigation labels. Preserve reading order and line breaks.'
        chat = '<|user|><|img|><|imgpad|><|endofimg|>' + prompt_txt + '<|endofuser|><|assistant|>'

        toks = tokenizer(chat, return_tensors='pt')
        vision = image_processor(images=[tile_img], return_tensors='pt')

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

        expanded_count = (new_input_ids == image_token_id).sum().item()
        if expanded_count != vision_len:
            raise RuntimeError(f"image token expansion mismatch: expanded={expanded_count} expected={vision_len}")

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        gen_kwargs = dict(
            max_new_tokens=args.max_new,
            min_new_tokens=(min_new_override if min_new_override is not None else args.min_new),
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=max(0, int(args.no_repeat_ngram_size)),
            repetition_penalty=float(args.repetition_penalty),
            do_sample=False,
        )
        if beams and beams > 1:
            gen_kwargs.update(dict(num_beams=beams, early_stopping=True, num_return_sequences=1))
        if sample:
            gen_kwargs.update(dict(do_sample=True, temperature=0.6, top_p=0.9, top_k=int(args.top_k)))

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        input_len = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_len:] if outputs.shape[1] > input_len else outputs[0]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not raw_text:
            full = tokenizer.decode(outputs[0], skip_special_tokens=True)
            raw_text = full.split(prompt, 1)[-1].strip() if prompt in full else full.strip()
        cleaned = _clean_ocr(raw_text) if (not args.no_clean and raw_text) else raw_text

        dbg = {
            'image_hw': (tile_img.size[1], tile_img.size[0]),
            'vision_len': vision_len,
            'expanded': expanded_count,
            'match': (expanded_count == vision_len),
            'prompt_tokens': int(new_input_ids.shape[1]),
            'new_tokens': int(max(0, outputs.shape[1] - input_len)),
        }
        return {'raw': raw_text, 'clean': cleaned, 'dbg': dbg}

    # Tiled path (wrapper around the stable single-image path)
    do_tile = bool(args.tile)
    tw, th = _parse_tile_size(args.tile_size)
    overlap = max(0, int(args.tile_overlap))

    results: List[str] = []
    dbg_lines: List[str] = []
    per_tile_sections: List[str] = []

    # Heuristics and scoring
    re_struct = re.compile(r"(\b\d{2,4}x\d{2,4}\b|\bbox\b|\bcategory\b|https?://\S+|www\.[^\s]+)", re.IGNORECASE)

    def metrics(s: str):
        lines = [ln.strip() for ln in (s or '').splitlines() if ln.strip()]
        n_lines = len(lines)
        alnum = sum(ch.isalnum() for ch in s)
        letters = sum(('a' <= ch.lower() <= 'z') or ('\u4e00' <= ch <= '\u9fff') for ch in s)
        struct_hits = len(re_struct.findall(s))
        # punctuation-only or single-letter lines
        import string
        punc = 0
        for ln in lines:
            if all((c in string.punctuation) for c in ln) or len(ln)==1:
                punc += 1
        punc_ratio = (punc / n_lines) if n_lines else 1.0
        sparse = (n_lines < 10) or (punc_ratio >= 0.6) or (struct_hits > 0)
        return {
            'n_lines': n_lines,
            'alnum': alnum,
            'letters': letters,
            'struct_hits': struct_hits,
            'punc_ratio': punc_ratio,
            'sparse': sparse,
        }

    def better(a: str, b: str):
        ma, mb = metrics(a), metrics(b)
        # Prefer more alnum, more letters, fewer structural hits; tie-break by lines
        key_a = (ma['alnum'], ma['letters'], -ma['struct_hits'], ma['n_lines'])
        key_b = (mb['alnum'], mb['letters'], -mb['struct_hits'], mb['n_lines'])
        return a if key_a >= key_b else b

    if do_tile and (tw > 0 and th > 0):
        tiles = _tile_image(img, tw, th, overlap)
        for i, timg in enumerate(tiles, start=1):
            # primary run: beams=3, min_new=128
            primary = run_single(timg, args.prompt, beams=3, sample=False, min_new_override=max(128, args.min_new))
            pm = metrics(primary['clean'])
            chosen = primary
            mode = 'beams(3)'

            need_fb = pm['sparse']
            fallback = None
            if need_fb:
                # fallback: sampling with min_new=128
                fallback = run_single(timg, args.prompt, beams=1, sample=True, min_new_override=max(128, args.min_new))
                chosen_text = better(primary['clean'], fallback['clean'])
                chosen = fallback if chosen_text == fallback['clean'] else primary
                mode = 'sample' if chosen is fallback else mode

                # second-stage: still sparse? try narrower 800px center crop with sampling
                cm = metrics(chosen['clean'])
                if cm['sparse'] and timg.size[0] > 800:
                    w, h = timg.size
                    new_w = 800
                    left = max(0, (w - new_w)//2)
                    right = min(w, left + new_w)
                    narrow = timg.crop((left, 0, right, h))
                    narrow_try = run_single(narrow, args.prompt, beams=1, sample=True, min_new_override=max(128, args.min_new))
                    chosen_text2 = better(chosen['clean'], narrow_try['clean'])
                    if chosen_text2 == narrow_try['clean']:
                        chosen = narrow_try
                        mode = 'sample-narrow'

            cdbg = chosen['dbg']
            dbg_lines.append(
                f"tile {i}/{len(tiles)} | image(HxW): {cdbg['image_hw'][0]}x{cdbg['image_hw'][1]} | visual_tokens expected={cdbg['vision_len']} expanded={cdbg['expanded']} match={cdbg['match']} | mode: {mode} | prompt_tokens: {cdbg['prompt_tokens']} new_tokens: {cdbg['new_tokens']}"
            )

            # Per-tile sections when --debug
            if args.debug:
                print(f"=== TILE {i} DEBUG ===")
                print(dbg_lines[-1])
                print(f"=== TILE {i} CLEANED ===")
                tlines = (chosen['clean'] or '').splitlines()
                print('\n'.join(tlines[:60]))

            results.append(chosen['clean'])
        final_text = "\n".join(results)
    else:
        # Original single-image path
        single = run_single(img, args.prompt, beams=max(1, int(args.beams)), sample=bool(args.sample))
        cdbg = single['dbg']
        mode = ('sample' if args.sample else (f"beams({args.beams})" if args.beams and args.beams>1 else 'greedy'))
        dbg_lines.append(
            f"image(HxW): {cdbg['image_hw'][0]}x{cdbg['image_hw'][1]} | visual_tokens expected={cdbg['vision_len']} expanded={cdbg['expanded']} match={cdbg['match']} | mode: {mode} | prompt_tokens: {cdbg['prompt_tokens']} new_tokens: {cdbg['new_tokens']}"
        )
        final_text = single['clean']

    # Emit consolidated debug, cleaned head, and VRAM tail
    print('===== DEBUG =====')
    print(f"model: {args.model}")
    print(f"device: {str(device)} | attn_impl: sdpa | load_in_4bit: True")
    for ln in dbg_lines:
        print(ln)
    print('===== END DEBUG =====\n')

    print('=== STITCHED CLEANED OCR ===')
    lines = (final_text or '').splitlines()
    head = '\n'.join(lines[:120])
    print(head)

    # VRAM tail
    import os, subprocess
    cand = [os.path.expanduser('~/nvidia_repo.log'), os.path.expanduser('~/nvidia.log')]
    try:
        print('\n===== VRAM (tail) =====')
        path = next((p for p in cand if os.path.exists(p)), None)
        if path:
            with open(path,'r',encoding='utf-8',errors='ignore') as f:
                tail = f.readlines()[-10:]
            for ln in tail:
                print(ln.rstrip())
        else:
            subprocess.run(['nvidia-smi'], check=False)
        print('===== END VRAM =====')
    except Exception:
        pass

if __name__ == '__main__':
    main()
