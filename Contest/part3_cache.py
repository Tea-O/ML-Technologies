"""Decode all Cats_and_Dogs images once into uint8 arrays (N, S, S, 3) for fast GPU training.

Usage:  python part3_cache.py [--data Cats_and_Dogs] [--sizes 288]
Creates cache/{train,val,test}_{size}.npz with arrays x (uint8 NHWC) and y (0 = cat, 1 = dog).
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ['cats', 'dogs']


def _load(args):
    path, size, preserve_aspect = args
    with Image.open(path) as source:
        source.draft('RGB', (size, size))
        img = source.convert('RGB')
    if preserve_aspect:
        resized = ImageOps.contain(img, (size, size), method=Image.Resampling.BICUBIC)
        img = Image.new('RGB', (size, size), (124, 116, 104))
        img.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    else:
        img = img.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def build(data_dir, out_dir, split, size, workers=8, preserve_aspect=False):
    out_path = os.path.join(out_dir, f'{split}_{size}.npz')
    if os.path.exists(out_path):
        print(f'{out_path} already exists, skip')
        return
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        d = os.path.join(data_dir, split, cls)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(d, f))
                labels.append(label)
    with ProcessPoolExecutor(workers) as ex:
        x = np.stack(list(ex.map(_load, [(p, size, preserve_aspect) for p in paths], chunksize=64)))
    y = np.array(labels, dtype=np.int64)
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = out_path + '.tmp.npz'
    np.savez(tmp_path, x=x, y=y)
    os.replace(tmp_path, out_path)
    print(split, x.shape, 'cats/dogs =', np.bincount(y))


def build_all(data_dir=os.path.join(HERE, 'Cats_and_Dogs'), out_dir=os.path.join(HERE, 'cache'),
              sizes=(288,), workers=8, preserve_aspect=False):
    for split in ['train', 'val', 'test']:
        if not os.path.isdir(os.path.join(data_dir, split)):
            raise FileNotFoundError(f'{os.path.join(data_dir, split)} not found: '
                                    f'data_dir must contain train/ val/ test/ with cats/ and dogs/ inside')
    for size in sizes:
        for split in ['train', 'val', 'test']:
            build(data_dir, out_dir, split, size, workers, preserve_aspect=preserve_aspect)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=os.path.join(HERE, 'Cats_and_Dogs'))
    p.add_argument('--out', default=os.path.join(HERE, 'cache'))
    p.add_argument('--sizes', type=int, nargs='+', default=[288])
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--preserve-aspect', action='store_true', help='pad instead of stretching; use a separate output directory')
    a = p.parse_args()
    build_all(a.data, a.out, a.sizes, a.workers, preserve_aspect=a.preserve_aspect)
