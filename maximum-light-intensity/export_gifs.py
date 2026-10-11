#!/usr/bin/env python3
"""Convert finalized scientific diagrams to static GIFs without resizing."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
from PIL import Image


def convert_png_to_gif(png_path, gif_path):
    png_path, gif_path = Path(png_path), Path(gif_path)
    with Image.open(png_path) as source:
        rgba = source.convert('RGBA')
        background = Image.new('RGBA', rgba.size, 'white')
        rgb = Image.alpha_composite(background, rgba).convert('RGB')
        palette = rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                               dither=Image.Dither.NONE)
        palette.save(gif_path, format='GIF', optimize=False)
        return {'file': str(gif_path), 'size': list(source.size)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, default=Path(__file__).resolve().parent/'results')
    args = ap.parse_args()
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(convert_png_to_gif, args.results/f'n{n}'/'diagram.png',
                            args.results/f'n{n}'/'diagram.gif') for n in range(36, 51)]
        for job in jobs:
            print(job.result(), flush=True)

if __name__ == '__main__':
    main()
