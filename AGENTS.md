# PROJECT KNOWLEDGE BASE

**Generated:** 2026-02-28
**Project:** Abnormal-ECG-signal-classification

## OVERVIEW

Deep learning pipeline that classifies 12-lead ECG signals into 5 cardiovascular conditions (NORM, MI, STTC, CD, HYP) using a custom CNN on Recurrence Plot Matrix (RPM) representations. Stack: Python 3.8+, PyTorch, Streamlit, WFDB, scikit-learn, SHAP, Google Generative AI.

## STRUCTURE

```
./
├── model.py              # Training-only script (Jupyter-origin, has !pip install lines)
├── requirements.txt      # Identical to streamlit_app/requirements.txt
├── streamlit_app/        # Complete deployment package — see streamlit_app/AGENTS.md
├── training_curves.png   # Output artifact from model.py training run
├── confusion_matrix.png  # Output artifact
├── gradcam_visualization.png
└── shap_visualization.png
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| ECG preprocessing + RPM creation | `model.py` → `preprocess_signals()`, `create_rpm_representations()` | Shared logic, duplicated into `streamlit_app/model.py` |
| Model architecture | `model.py` → `FeatureExtractionModule`, `ResidualBlock` | Conv2d on 72×72 RPM images |
| All hyperparameters | `model.py` → `Config` class (lines 54–111) | Single source of truth for training |
| Focal Loss + class weights | `model.py` → `FocalLoss`, `FocalLoss.compute_alpha_weights()` | Handles 5-class imbalance |
| Training loop | `model.py` → `main()` (line 889) | AdamW + CosineAnnealingWarmRestarts, early stopping at patience=15 |
| Explainability | `model.py` → `GradCAM`, `SHAPExplainer`, `SHAPCompatibleModel` | Grad-CAM targets `model.conv2` layer |
| Dataset loading | `model.py` → `load_ptbxl_metadata()`, `load_ecg_signals()` | PTB-XL via WFDB format, lead index=1 |
| Data augmentation | `model.py` → `ECGAugmentation` | Gaussian noise + scale only |
| Inference + web UI | `streamlit_app/` | See `streamlit_app/AGENTS.md` |

## DATA PIPELINE

```
PTB-XL (WFDB .dat/.hea files)
  → load_ecg_signals()        # lead_idx=1 (second lead), skips failed records silently
  → adjust_signal_length()    # truncate >1000, pad 800–999, discard <800
  → zscore_normalize()        # per-signal: (x - mean) / std
  → create_rpm_representations()  # downsample→72pts, pairwise |diff|, minmax norm → (N,1,72,72)
  → SMOTE oversampling        # minority classes to 30% of majority count
  → DataLoader (batch=128)
  → FeatureExtractionModule   # returns (features[256-dim], logits[5-dim])
```

## KEY CLASSES & CONFIG

**`Config` class** — all training hyperparameters as class attributes:
- `SIGNAL_LENGTH=1000` (10s @ 100Hz), `RPM_SIZE=72` (72×72 matrix)
- `FOCAL_GAMMA=2.5`, `FOCAL_LABEL_SMOOTHING=0.1`, `CB_BETA=0.9999`
- `OVERSAMPLE_RATIO=0.3`, `DROPOUT=0.4`, `LR=0.0001`, `BATCH_SIZE=128`
- `DEBUG_MODE=False` → set True + `DEBUG_SAMPLES=500` for quick iteration

**`FeatureExtractionModule.forward(x)`** returns `(features, logits)` — **always unpack both**. `features` is the 256-dim intermediate embedding; `logits` are the raw class scores.

**`FocalLoss`** — custom Focal Loss with label smoothing and class-balanced alpha weights. Use `FocalLoss.compute_alpha_weights(samples_per_class, method='effective')` to compute alpha from class counts.

**Class map:** `NORM=0, MI=1, STTC=2, CD=3, HYP=4`

## ANTI-PATTERNS (THIS PROJECT)

- **Never flip or rotate RPM augmentations.** RPM[i,j] encodes temporal distance between time points i and j — flipping destroys this semantic. Only noise + scale augmentations are valid (see `ECGAugmentation`).
- **Don't run root `model.py` as a plain Python script.** Lines 7 and 9 contain `!pip install` commands (Jupyter/Colab syntax). Running outside a notebook will raise `SyntaxError`. Use `streamlit_app/model.py` instead for any non-notebook work, or remove those lines first.
- **Don't use root `model.py` for inference.** It's training-only. The deployment-adapted version is `streamlit_app/model.py`.
- **Don't optimize for accuracy alone.** Dataset is heavily imbalanced. Use macro F1 and balanced accuracy as primary metrics (the training loop already does this for model selection).
- **Don't modify `Config` for inference.** `OVERSAMPLE_RATIO`, `DEBUG_MODE`, training splits are training-only. The Streamlit app uses `Config` only for `SIGNAL_LENGTH`, `RPM_SIZE`, `CLASS_NAMES`, `NUM_CLASSES`.
- **`requirements.txt` is duplicated** (root = `streamlit_app/requirements.txt`). Edit both when adding a dependency, or pick one as canonical and symlink.

## UNIQUE PATTERNS

- **RPM representation:** 1D ECG → downsample to 72 points → pairwise absolute differences → 72×72 image. This converts temporal dynamics to a 2D structure amenable to CNN processing.
- **SHAP wrapper:** `SHAPCompatibleModel` re-implements `FeatureExtractionModule.forward()` manually because SHAP `DeepExplainer` cannot trace through `nn.ModuleList` iteration reliably. If model architecture changes, update both.
- **Silent signal failure:** `load_ecg_signals()` uses bare `except Exception: continue` — failed records are silently skipped. This is intentional for robustness with large datasets.
- **Model selection on macro F1, not val loss.** `best_val_f1` gates checkpoint saving. Early stopping triggers after 15 epochs without F1 improvement.

## COMMANDS

```bash
# Environment setup
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Git LFS (required — best_model.pth stored in LFS)
git lfs install && git lfs pull

# Training (requires Kaggle credentials; downloads PTB-XL automatically)
# First remove !pip install lines from model.py (lines 7, 9), then:
python model.py

# Web application
cd streamlit_app
streamlit run app.py

# Optional: Gemini AI integration
set GEMINI_API_KEY=your_key_here   # Windows CMD
# or create streamlit_app/.streamlit/secrets.toml with GEMINI_API_KEY = "..."
```

## NOTES

- **No tests, no CI/CD, no linting config.** This is a research/demo project. No pytest, no mypy, no black.
- **Git LFS required.** `best_model.pth` (~tracked via `.gitattributes`: `*.pth filter=lfs diff=lfs merge=lfs -text`). Without `git lfs pull`, the app silently loads an uninitialized model.
- **Dataset not in repo.** Training downloads PTB-XL from Kaggle via `kagglehub`. Requires a Kaggle account with API credentials configured.
- **`streamlit_app/data.csv`** — sample/demo ECG data file included for local testing.
- **Deployment target:** Streamlit Cloud. Main file path for deployment: `streamlit_app/app.py`. Add `GEMINI_API_KEY` in Streamlit Cloud Secrets if using AI explanations.
- **`streamlit_app/model.py` is the source of truth** for the model architecture used in production. Root `model.py` may diverge from it during training experiments.
