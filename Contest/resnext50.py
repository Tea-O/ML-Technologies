import argparse
import json
from pathlib import Path

import torch

import part3_cache
import part3_train

ROOT = Path(__file__).resolve().parent
ARCH = 'resnext50_32x4d'
SIZE = 288


def load_model(checkpoint, device):
    model = part3_train.build_model(ARCH).to(device, memory_format=part3_train.mem_format(device))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    return model


def train(data, cache, out):
    part3_cache.build_all(data, cache, sizes=(SIZE,))
    settings = json.loads((ROOT / 'training_log.json').read_text(encoding='utf-8'))['args']
    settings.update(cache=str(cache), out=str(out))
    argv = []
    for name, value in settings.items():
        flag = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend((flag, str(value)))
    part3_train.main(argv)


def evaluate(args, model, device):
    scores = {}
    for split in ('val', 'test'):
        part3_cache.build(args.data, args.cache, split, SIZE)
        images, labels = part3_train.load_split(args.cache, split, SIZE)
        probability = part3_train.predict(model, images, device, bs=args.batch, res=SIZE)
        correct = int(((probability > 0.5) == labels.numpy()).sum())
        scores[split] = {'accuracy': correct / len(labels), 'correct': correct,
                         'errors': len(labels) - correct, 'images': len(labels)}
        print(f'{split}: accuracy {scores[split]["accuracy"]:.4f} ({correct}/{len(labels)} correct)')
    if args.report:
        report = {'architecture': ARCH, 'resolution': SIZE, 'weights': args.weights.name,
                  'tta': 'Mean of logits for the original and horizontally flipped image',
                  'split_metrics': scores}
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'train', 'evaluate'):
        sub = commands.add_parser(command)
        sub.add_argument('--data', type=Path, default=ROOT / 'Cats_and_Dogs')
        sub.add_argument('--cache', type=Path, default=ROOT / 'cache')
        if command == 'train':
            sub.add_argument('--out', type=Path, default=ROOT / 'retrained')
        if command == 'evaluate':
            sub.add_argument('--weights', type=Path, default=ROOT / 'model_ema.pt')
            sub.add_argument('--report', type=Path)
            sub.add_argument('--batch', type=int, default=64)
    predict = commands.add_parser('predict')
    predict.add_argument('image', type=Path)
    predict.add_argument('--weights', type=Path, default=ROOT / 'model_ema.pt')
    args = parser.parse_args()
    if args.command == 'prepare':
        part3_cache.build_all(args.data, args.cache, sizes=(SIZE,))
    elif args.command == 'train':
        train(args.data, args.cache, args.out)
    else:
        device = part3_train.get_device()
        model = load_model(args.weights, device)
        if args.command == 'evaluate':
            if args.batch < 1:
                parser.error('--batch must be positive')
            evaluate(args, model, device)
        else:
            pixels = part3_cache._load((args.image, SIZE, False)).copy()
            p_dog = float(part3_train.predict(model, torch.from_numpy(pixels).unsqueeze(0),
                                             device, res=SIZE)[0])
            print(f'{"dog" if p_dog > 0.5 else "cat"} (P(dog)={p_dog:.4f})')


if __name__ == '__main__':
    main()
