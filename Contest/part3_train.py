"""Part 3 train a cats-vs-dogs classifier.

Works on NVIDIA GPUs (CUDA, bf16 autocast, channels_last, optional torch.compile) and on Apple MPS.

Pipeline
* images are pre-decoded to uint8 288x288 (part3_cache.py); kept on the GPU if there is enough VRAM,
  otherwise in (pinned) RAM with a background prefetch thread;
* augmentation on the GPU per batch: RandomResizedCrop + horizontal flip (one affine grid_sample),
  colour jitter, MixUp / CutMix;  progressive resizing of the training crops;
* CNN from torchvision with weights=None, SGD + Nesterov, warmup + cosine LR, label smoothing,
  weight decay, EMA of weights (with warm-up), mixed precision;
* evaluation on full images at a resolution higher than the training crops (FixRes effect) + h-flip TTA;
* checkpoint every few epochs -> an interrupted run continues automatically when started again.

Example:
    python part3_train.py --arch resnet50 --epochs 250 --res-start 160 --res 224 --test-res 288 --out runs/resnet50
"""
import argparse
import copy
import json
import math
import os
import queue
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

HERE = os.path.dirname(os.path.abspath(__file__))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)   # just normalisation constants
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--arch', default='resnet50')
    p.add_argument('--epochs', type=int, default=250)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1, help='lr for batch 128 (scaled linearly with --bs)')
    p.add_argument('--optimizer', choices=['sgd', 'adamw'], default='sgd')
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--res', type=int, default=224, help='final train crop resolution')
    p.add_argument('--res-start', type=int, default=160, help='progressive resizing start (0 = off)')
    p.add_argument('--test-res', type=int, default=288, help='resolution for validation during training')
    p.add_argument('--min-scale', type=float, default=0.25)
    p.add_argument('--smoothing', type=float, default=0.1)
    p.add_argument('--mix-prob', type=float, default=0.5, help='probability to apply mixup/cutmix to a batch')
    p.add_argument('--ema', type=float, default=0.9995)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cache', default=os.path.join(HERE, 'cache'))
    p.add_argument('--size', type=int, default=288, help='resolution of the cached images')
    p.add_argument('--cache-device', default='auto', choices=['auto', 'gpu', 'cpu'],
                   help='where to keep the training images; auto = GPU if VRAM >= 20 GB')
    p.add_argument('--compile', action='store_true', help='torch.compile the model (Linux; Windows needs triton)')
    p.add_argument('--ckpt-every', type=int, default=5)
    p.add_argument('--val-every', type=int, default=1)
    p.add_argument('--limit-train', type=int, default=0, help='use only N training images (quick test)')
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    if args.val_every < 1:
        p.error('--val-every must be positive')
    return args


# ----------------------------------------------------------------------------- model / data
def build_model(arch):
    if arch == 'resnet26':
        from torchvision.models.resnet import ResNet, Bottleneck
        return ResNet(Bottleneck, [2, 2, 2, 2], num_classes=1, zero_init_residual=True)
    if arch.startswith(('resnet', 'resnext', 'wide_resnet')):
        return getattr(models, arch)(weights=None, num_classes=1, zero_init_residual=True)
    return getattr(models, arch)(weights=None, num_classes=1)


def load_split(cache, split, size=288):
    d = np.load(os.path.join(cache, f'{split}_{size}.npz'))
    return torch.from_numpy(d['x']), torch.from_numpy(d['y'])


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def amp_dtype(device):
    # bf16 on NVIDIA (no GradScaler needed), fp16 on Apple MPS
    return torch.bfloat16 if device.type in ('cuda', 'cpu') else torch.float16


def mem_format(device):
    # channels_last is faster on NVIDIA tensor cores; buggy on MPS
    return torch.channels_last if device.type == 'cuda' else torch.contiguous_format


class BatchLoader:
    """Yields (uint8 images on device, float labels on device) for one epoch."""

    def __init__(self, X, Y, bs, device, on_gpu):
        self.bs, self.device, self.on_gpu = bs, device, on_gpu
        self.N = len(X)
        if on_gpu:
            self.X, self.Y = X.to(device), Y.float().to(device)
        else:
            self.X, self.Y = X, Y.float()

    def epoch(self, steps):
        if self.on_gpu:
            perm = torch.randperm(self.N, device=self.device)
            for i in range(steps):
                idx = perm[i * self.bs:(i + 1) * self.bs]
                yield self.X[idx], self.Y[idx]
            return
        perm = torch.randperm(self.N)
        q = queue.Queue(maxsize=4)

        def worker():
            for i in range(steps):
                idx = perm[i * self.bs:(i + 1) * self.bs]
                xb = self.X.index_select(0, idx)
                if self.device.type == 'cuda':
                    xb = xb.pin_memory()
                q.put((xb, self.Y[idx]))
            q.put(None)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            xb, yb = item
            yield xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
        t.join()


# ----------------------------------------------------------------------------- augmentation
def gpu_augment(xb, out_res, min_scale):
    """xb: uint8 (B, H, W, 3) on device -> float (B, 3, out, out) normalised."""
    x = xb.permute(0, 3, 1, 2).contiguous().float().div_(255)  # NB: MPS grid_sample is wrong on channels-last strides
    B, dev = x.size(0), x.device
    area = torch.empty(B, device=dev).uniform_(min_scale, 1.0)
    logr = torch.empty(B, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3))
    ratio = torch.exp(logr)
    w = torch.sqrt(area * ratio).clamp(max=1.0)
    h = torch.sqrt(area / ratio).clamp(max=1.0)
    cx = (torch.rand(B, device=dev) * 2 - 1) * (1 - w)
    cy = (torch.rand(B, device=dev) * 2 - 1) * (1 - h)
    flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = w * flip
    theta[:, 0, 2] = cx
    theta[:, 1, 1] = h
    theta[:, 1, 2] = cy
    grid = F.affine_grid(theta, (B, 3, out_res, out_res), align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)

    # colour jitter: brightness, contrast, saturation
    b = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
    c = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
    s = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.7, 1.3)
    x = x * b
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * c + mean
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = (x - gray) * s + gray
    x = x.clamp_(0, 1)
    return (x - MEAN.to(dev)) / STD.to(dev)


def mix(x, y, prob):
    """MixUp (alpha=0.2) or CutMix (alpha=1.0) on a batch with probability prob. y: float (B,)"""
    if np.random.rand() >= prob:
        return x, y
    perm = torch.randperm(x.size(0), device=x.device)
    if np.random.rand() < 0.5:
        lam = float(np.random.beta(0.2, 0.2))
        x = lam * x + (1 - lam) * x[perm]
    else:
        lam = float(np.random.beta(1.0, 1.0))
        H, W = x.shape[2:]
        rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
        cy, cx = np.random.randint(H), np.random.randint(W)
        y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
        x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return x, lam * y + (1 - lam) * y[perm]


def prep_eval(xb, res=0):
    x = xb.permute(0, 3, 1, 2).contiguous().float().div_(255)
    if res and res != x.shape[-1]:
        x = F.interpolate(x, size=(res, res), mode='bilinear', align_corners=False, antialias=True)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


@torch.no_grad()
def predict(model, X, device, bs=200, tta=True, res=0):
    """Returns P(dog) for every image of uint8 tensor X (N, S, S, 3); TTA = original + h-flip."""
    model.eval()
    probs = []
    for i in range(0, len(X), bs):
        x = prep_eval(X[i:i + bs].to(device), res).contiguous(memory_format=mem_format(device))
        with torch.autocast(device.type, dtype=amp_dtype(device)):
            logit = model(x).float()
            if tta:
                logit = (logit + model(torch.flip(x, dims=[3])).float()) / 2
        probs.append(torch.sigmoid(logit).squeeze(1).cpu())
    return torch.cat(probs).numpy()


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = copy.deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._bind(model)

    def _bind(self, model):
        esd, msd = self.model.state_dict(), model.state_dict()
        fl = [k for k, v in esd.items() if v.dtype.is_floating_point]
        nf = [k for k, v in esd.items() if not v.dtype.is_floating_point]
        self.e_float, self.m_float = [esd[k] for k in fl], [msd[k] for k in fl]
        self.e_int, self.m_int = [esd[k] for k in nf], [msd[k] for k in nf]

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))
        torch._foreach_mul_(self.e_float, d)
        torch._foreach_add_(self.e_float, self.m_float, alpha=1 - d)
        for e, m in zip(self.e_int, self.m_int):
            e.copy_(m)


def build_optimizer(model, args, lr):
    decay, no_decay = [], []
    for param in model.parameters():
        (no_decay if param.ndim <= 1 else decay).append(param)
    groups = [{'params': decay, 'weight_decay': args.wd},
              {'params': no_decay, 'weight_decay': 0.0}]
    if args.optimizer == 'adamw':
        return torch.optim.AdamW(groups, lr=lr, fused=next(model.parameters()).is_cuda)
    return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True)


def probe_batch_size(model, args, device, mf, limit=0.85):
    """CUDA only: largest batch size <= bs (halving) whose train step at the final resolution fits in VRAM."""
    total = torch.cuda.get_device_properties(0).total_memory
    bs, res = args.bs, args.res
    opt = build_optimizer(model, args, lr=0.0)
    while bs >= 16:
        x = out = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(bs, 3, res, res, device=device).contiguous(memory_format=mf)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(x).float().mean()
            out.backward()
            opt.step()
            peak = torch.cuda.max_memory_allocated()
            if peak < limit * total:
                break
            print(f'batch {bs}: peak {peak / 2 ** 30:.1f} GB of {total / 2 ** 30:.1f} GB -> too close, halving', flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f'batch {bs}: CUDA out of memory -> halving', flush=True)
        finally:
            del x, out
            model.zero_grad(set_to_none=True)
        bs //= 2
    del opt
    torch.cuda.empty_cache()
    return max(bs, 16)


def accuracy(p, y):
    return float(((p > 0.5) == y).mean() * 100)


# ----------------------------------------------------------------------------- training
def main(argv=None):
    a = get_args(argv)
    os.makedirs(a.out, exist_ok=True)
    log_path = os.path.join(a.out, 'log.json')
    if os.path.exists(log_path) and 'final_test_acc_tta' in json.load(open(log_path)):
        print(f'{a.out}: already finished, skip (delete the folder to retrain)', flush=True)
        return
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = get_device()
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    mf = mem_format(device)

    Xtr, Ytr = load_split(a.cache, 'train', a.size)
    Xva, Yva = load_split(a.cache, 'val', a.size)
    Xte, Yte = load_split(a.cache, 'test', a.size)
    if a.limit_train:
        sel = torch.randperm(len(Xtr))[:a.limit_train]
        Xtr, Ytr = Xtr[sel], Ytr[sel]
        Xva, Yva, Xte, Yte = Xva[:500], Yva[:500], Xte[:500], Yte[:500]
    yv, yt = Yva.numpy(), Yte.numpy()
    N = len(Xtr)

    if a.cache_device == 'auto':
        on_gpu = device.type == 'mps' or (device.type == 'cuda' and
                                          torch.cuda.get_device_properties(0).total_memory >= 20 * 2 ** 30)
    else:
        on_gpu = a.cache_device == 'gpu'
    loader = BatchLoader(Xtr, Ytr, a.bs, device, on_gpu)

    model = build_model(a.arch).to(device).to(memory_format=mf)
    n_params = sum(p.numel() for p in model.parameters())
    gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else device.type
    print(f'{a.arch}: {n_params / 1e6:.1f}M params | train {N} images | {gpu_name} | '
          f'images kept on {"GPU" if on_gpu else "CPU"}', flush=True)

    resumed = os.path.exists(os.path.join(a.out, 'last.pt'))
    if resumed:   # keep the batch size the run was started with
        a.bs = torch.load(os.path.join(a.out, 'last.pt'), map_location='cpu', weights_only=False)['log']['args']['bs']
    elif device.type == 'cuda':
        bs = probe_batch_size(model, a, device, mf)
        if bs != a.bs:
            print(f'batch size {a.bs} -> {bs} (lr is scaled accordingly)', flush=True)
            a.bs = bs
    loader.bs = a.bs

    base_lr = a.lr * a.bs / 128
    opt = build_optimizer(model, a, lr=base_lr)
    steps_per_epoch = N // a.bs
    total = a.epochs * steps_per_epoch
    warm = a.warmup * steps_per_epoch

    def lr_at(step):
        if step < warm:
            return base_lr * (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return base_lr * 0.5 * (1 + math.cos(math.pi * t))

    def res_at(epoch):
        # progressive resizing: linear from res_start to res over the first 85% of epochs, multiples of 16
        if not a.res_start:
            return a.res
        t = min(1.0, epoch / max(1, 0.85 * a.epochs - 1))
        return int(round((a.res_start + t * (a.res - a.res_start)) / 16) * 16)

    ema = EMA(model, a.ema)
    log = {'args': vars(a), 'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': [],
           'val_acc_ema': [], 'time': []}
    start_epoch, step = 0, 0

    ckpt_path = os.path.join(a.out, 'last.pt')
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        ema.model.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['opt'])
        start_epoch, step, log = ck['epoch'], ck['step'], ck['log']
        torch.set_rng_state(ck['rng_cpu'].cpu())
        if device.type == 'cuda' and 'rng_cuda' in ck:
            torch.cuda.set_rng_state(ck['rng_cuda'].cpu(), device)
        np.random.set_state(ck['rng_np'])
        print(f'resumed from epoch {start_epoch}', flush=True)

    train_model = torch.compile(model) if a.compile else model
    crit = nn.BCEWithLogitsLoss()
    adt = amp_dtype(device)

    for epoch in range(start_epoch, a.epochs):
        t0 = time.time()
        res = res_at(epoch)
        model.train()
        run_loss = torch.zeros((), device=device)
        for xb, y in loader.epoch(steps_per_epoch):
            x = gpu_augment(xb, res, a.min_scale)
            x, y = mix(x, y, a.mix_prob)
            y = y * (1 - a.smoothing) + 0.5 * a.smoothing
            x = x.contiguous(memory_format=mf)
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            with torch.autocast(device.type, dtype=adt):
                logit = train_model(x).squeeze(1)
            loss = crit(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(model, step)
            run_loss += loss.detach()
            step += 1
        train_loss = float(run_loss) / steps_per_epoch
        if device.type == 'mps':
            torch.mps.empty_cache()

        val_loss, val_acc, val_acc_ema = None, None, None
        if (epoch + 1) % a.val_every == 0 or epoch + 1 == a.epochs:
            pv = predict(model, Xva, device, tta=False, res=a.test_res)
            pe = predict(ema.model, Xva, device, tta=False, res=a.test_res)
            val_loss = float(-np.mean(yv * np.log(np.clip(pv, 1e-7, 1)) + (1 - yv) * np.log(np.clip(1 - pv, 1e-7, 1))))
            val_acc, val_acc_ema = accuracy(pv, yv), accuracy(pe, yv)
        vals = [epoch + 1, train_loss, val_loss, val_acc, val_acc_ema, time.time() - t0]
        for k, v in zip(['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_acc_ema', 'time'], vals):
            log[k].append(v)
        validation = (f' val_loss {val_loss:.4f} val_acc {val_acc:.2f} val_acc_ema {val_acc_ema:.2f}'
                      if val_loss is not None else '')
        print(f'{a.arch} epoch {epoch + 1}/{a.epochs} res {res} lr {lr_at(step - 1):.4f} '
              f'train_loss {train_loss:.4f}{validation} ({vals[5]:.0f}s)', flush=True)
        with open(log_path, 'w') as f:
            json.dump(log, f)
        if (epoch + 1) % a.ckpt_every == 0 or epoch + 1 == a.epochs:
            torch.save({'model': model.state_dict(), 'ema': ema.model.state_dict(), 'opt': opt.state_dict(),
                        'epoch': epoch + 1, 'step': step, 'log': log,
                        'rng_cpu': torch.get_rng_state(), 'rng_np': np.random.get_state(),
                        **({'rng_cuda': torch.cuda.get_rng_state(device)} if device.type == 'cuda' else {})},
                       ckpt_path + '.tmp')
            os.replace(ckpt_path + '.tmp', ckpt_path)

    torch.save(ema.model.state_dict(), os.path.join(a.out, 'model_ema.pt'))
    pv = predict(ema.model, Xva, device, res=a.test_res)
    pt = predict(ema.model, Xte, device, res=a.test_res)
    log['final_val_acc_tta'] = accuracy(pv, yv)
    log['final_test_acc_tta'] = accuracy(pt, yt)
    with open(log_path, 'w') as f:
        json.dump(log, f)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"{a.arch} FINAL val acc (EMA+TTA) {log['final_val_acc_tta']:.2f}  "
          f"test acc (EMA+TTA) {log['final_test_acc_tta']:.2f}", flush=True)


if __name__ == '__main__':
    main()
