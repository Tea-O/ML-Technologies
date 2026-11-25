# Part 3: Cat and dog classification

ResNeXt-50 32x4d was trained from scratch for 250 epochs. It classified **2,478 of 2,500** test images correctly, for an accuracy of **0.9912**. The evaluation used 288 x 288 images, EMA weights, and horizontal flips.

## Files

- `PART3_ResNeXt50.ipynb`: lab report with code, learning curves, and a confusion matrix.
- `Cats_and_Dogs/`: original training, validation, and test images.
- `model_ema.pt`: final EMA weights.
- `resnext50.py`: command-line entry point for training, evaluation, and prediction.
- `part3_cache.py` and `part3_train.py`: preprocessing and training code.
- `training_log.json` and `evaluation.json`: training settings, metrics, and final results.

## Run the notebook

Open a terminal in this folder. For an NVIDIA GPU, install the CUDA builds of PyTorch and Torchvision first. The results were produced with Python 3.12 and CUDA 12.8.

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m notebook PART3_ResNeXt50.ipynb
```

Run all cells to evaluate the included weights. Set `RUN_TRAINING = True` to repeat training in `notebook_retrain/`.

## Command-line use

```powershell
python resnext50.py evaluate --report evaluation.json
python resnext50.py predict Cats_and_Dogs/test/cats/cat.10000.jpg
python resnext50.py train
```

Training reads the settings in `training_log.json`, saves checkpoints in `retrained/`, and resumes from the latest checkpoint when restarted. To build the image cache separately, run `python resnext50.py prepare`. The full cache takes about 6 GB of disk space.
