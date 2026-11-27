# Skin Lesion Classifier (HAM10000)

Transfer-learning classifier for 7 skin lesion types (HAM10000), served in a
minimal Streamlit app that accepts a webcam/phone photo or an uploaded image.

> **Disclaimer:** educational project, not a medical device. HAM10000 is a
> dermoscopy dataset; predictions on ordinary phone photos are much less
> reliable. Never use this to make health decisions.

## Project layout

```
skin-ai/
  data/splits/          train/val/test CSVs (image_path, label) — created by prepare_data
  src/config.py         all paths and hyperparameters
  src/prepare_data.py   kagglehub download, metadata parsing, label mapping, group split
  src/dataset.py        Dataset + transforms
  src/model.py          timm model builder, device detection, checkpoint loading, TTA
  src/train.py          training loop
  src/evaluate.py       test metrics + confusion matrix
  src/predict.py        reusable predict(image) -> class name
  app.py                Streamlit app
  models/               best_model.pth, classes.json (git-ignored; auto-downloaded from HF if missing)
  reports/              confusion_matrix.png
  requirements.txt        app-only dependencies (used by Streamlit Cloud)
  requirements-train.txt  + training/evaluation dependencies
```

## Quick start: just run the app

The trained model is hosted at
[huggingface.co/haseebahmed2453/skinai](https://huggingface.co/haseebahmed2453/skinai).
If `models/best_model.pth` or `models/classes.json` is missing, the app downloads
it automatically on first start (~110 MB, once).

```bash
cd skin-ai
python -m venv .venv
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
# macOS / Linux:          source .venv/bin/activate
pip install -r requirements.txt     # CPU-only PyTorch + app dependencies
streamlit run app.py
```

## Deploy free on Streamlit Community Cloud

1. Push this folder to a GitHub repo (`.gitignore` keeps `models/`, `data/`,
   `reports/` and the venv out of git).
2. Go to <https://share.streamlit.io> → **Create app** → pick the repo and branch,
   and set **Main file path** to `app.py`.
3. Under **Advanced settings**, choose **Python 3.12**.
4. Deploy. The first boot installs CPU-only PyTorch from `requirements.txt`
   and downloads the model from Hugging Face. That takes a few minutes; later
   restarts are faster.

## 1. Setup for training

Python 3.11+ is required (tested with 3.14 on Windows).

```bash
cd skin-ai
python -m venv .venv
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
# Windows (Git Bash):     source .venv/Scripts/activate
# macOS / Linux:          source .venv/bin/activate

# CPU (default; requirements.txt pulls CPU-only PyTorch):
pip install -r requirements-train.txt

# NVIDIA GPU: install the CUDA build of PyTorch FIRST, then the rest:
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-train.txt
# (macOS: plain `pip install torch torchvision` gives you MPS support)
```

## 2. Kaggle authentication

The dataset is public, and `kagglehub` can usually download it without logging
in. If you get a 401/403 error, set up credentials:

1. Sign in at <https://www.kaggle.com> → click your avatar → **Settings** → **API** → **Create New Token**.
   This downloads `kaggle.json`.
2. Put it here:
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
   - macOS/Linux: `~/.kaggle/kaggle.json` (then `chmod 600 ~/.kaggle/kaggle.json`)

   Or set environment variables instead:
   ```bash
   export KAGGLE_USERNAME=your_username     # PowerShell: $env:KAGGLE_USERNAME="your_username"
   export KAGGLE_KEY=your_api_key           # PowerShell: $env:KAGGLE_KEY="your_api_key"
   ```

## 3. Run the pipeline (from the `skin-ai/` folder)

```bash
# 1) Download (~2.6 GB, cached after the first run) and build group-aware splits
python src/prepare_data.py

# 2a) Smoke test: 20% of the data, 3 epochs
python src/train.py --quick

# 2b) Full training (early stopping on validation balanced accuracy)
python src/train.py

# 3) Test-set metrics + reports/confusion_matrix.png
python src/evaluate.py

# 4) Single-image prediction from the command line (optional)
python src/predict.py path/to/photo.jpg

# 5) App
streamlit run app.py
```

Each training run writes `models/best_model.pth` and `models/classes.json`,
so a `--quick` run overwrites any earlier full model.

### Training options

```bash
python src/train.py --backbone efficientnetv2_s --img-size 260
python src/train.py --backbone efficientnet_b0 --epochs 10     # much faster on CPU
python src/train.py --workers 4 --batch-size 32
```

Backbones (see `BACKBONES` in `src/config.py`): `convnext_tiny` (default),
`efficientnetv2_s`, `efficientnet_b0`, `mobilenetv3_large`.

### How long will training take?

- **NVIDIA GPU:** about 1–2 min per epoch for convnext_tiny, so a full run takes 15–30 min.
- **CPU only:** measured on an Intel i7-8550U (4 cores, no GPU), full 7,009-image train split:

  | Backbone            | Train speed | Per epoch | Full run (~13 epochs) | CPU inference (with TTA) |
  |---------------------|-------------|-----------|-----------------------|--------------------------|
  | convnext_tiny       | ~2.9 img/s  | ~43 min   | ~9 h (too long)       | ~250 ms                  |
  | efficientnetv2_s    | ~1.9 img/s  | ~60 min   | ~13 h                 | ~350 ms                  |
  | efficientnet_b0     | ~6 img/s    | ~20 min   | ~4.5 h                | ~100 ms                  |
  | mobilenetv3_large   | ~11 img/s   | ~11 min   | ~2.5 h                | ~70 ms                   |

  On a CPU-only machine with a few hours to spare, use
  `python src/train.py --backbone mobilenetv3_large` (or `--epochs 8`
  to shorten further). The `--quick` run with convnext_tiny takes ~18 min.

### Camera on a phone

Browsers allow camera access only on `localhost` or over HTTPS. To use
"Take Photo" from a phone on the same Wi-Fi, either serve over HTTPS, e.g.
`streamlit run app.py --server.sslCertFile cert.pem --server.sslKeyFile key.pem`,
or use a tunnel such as `ngrok http 8501`. The "Upload Image" tab works over plain HTTP.

## Design notes

- **Group split on `lesion_id`** (StratifiedGroupKFold, 70/15/15): images of
  the same lesion never cross splits, so test accuracy isn't inflated.
- **Imbalance:** class-weighted CrossEntropy (inverse frequency) with label
  smoothing 0.1. No weighted sampler.
- **Augmentation** to resemble phone photos: random resized crop, flips,
  rotation, perspective, color jitter, JPEG compression, Gaussian blur.
- **Schedule:** AdamW, lr 3e-4, 1 warmup epoch followed by cosine decay. The backbone is
  frozen for epoch 1. AMP is used on CUDA. Early stopping has patience 4.
- **Inference:** EXIF-aware loading, RGBA→RGB, the same preprocessing as validation, and TTA
  (original + horizontal flip). If the top probability is below 0.45, the app shows
  "Not sure — please retake a clearer, close-up photo".
