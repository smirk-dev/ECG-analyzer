# STREAMLIT APP — DEPLOYMENT KNOWLEDGE BASE

**Package:** `streamlit_app/` — complete self-contained deployment unit

## OVERVIEW

Web interface for ECG classification inference. Imports model architecture from local `model.py`, loads `best_model.pth`, and exposes a Streamlit UI. This is the production package — everything deployed to Streamlit Cloud lives here.

## STRUCTURE

```
streamlit_app/
├── app.py            # Entry point — Streamlit UI (778 lines)
├── model.py          # Deployment-adapted copy of root model.py (989 lines)
├── best_model.pth    # Trained weights — tracked via Git LFS (required)
├── data.csv          # Sample ECG data for local testing
├── requirements.txt  # Identical to root requirements.txt
└── .streamlit/
    └── secrets.toml  # GEMINI_API_KEY (gitignored, create manually)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Model loading | `app.py` → `load_model()` (line 233) | `@st.cache_resource`, tries script dir then cwd |
| Inference pipeline | `app.py` → `predict()`, `process_single_signal()` | Signal → RPM → logits → softmax → result dict |
| CSV parsing | `app.py` → `parse_csv()` | Handles Apple (512 Hz), Samsung (500 Hz), generic |
| Demo ECG synthesis | `app.py` → `generate_demo_ecg()` | P-wave + QRS + T-wave, optional ST elevation for abnormal |
| Signal quality check | `app.py` → `assess_signal_quality()` | Scores 0–100; Poor < 60, Fair 60–79, Good ≥ 80 |
| Uncertainty detection | `app.py` → `main()` analysis block (line 610–617) | 3 conditions trigger uncertain flag |
| Heart rate estimation | `app.py` → `estimate_heart_rate()` | scipy `find_peaks`, returns 75.0 as fallback |
| Health report builder | `app.py` → `build_health_report()` | Maps class → risk level, actions, assessment text |
| PDF export | `app.py` → `generate_pdf_report()` | Matplotlib figure saved to BytesIO via PdfPages |
| Gemini AI | `app.py` → `initialize_gemini()`, `analyze_with_gemini()` | `@st.cache_resource`; uses `gemini-2.5-flash` |
| Config + model classes | `model.py` → `Config`, `FeatureExtractionModule`, `create_rpm_representations` | Only these 3 are imported by app.py |

## INFERENCE PIPELINE (app.py)

```
User upload / Demo ECG
  → parse_csv() or generate_demo_ecg()     # raw signal + fs
  → resample_signal()                       # linear interp to 100 Hz if needed
  → process_single_signal()
      → truncate/zero-pad to SIGNAL_LENGTH=1000
      → zscore normalize
      → create_rpm_representations()        # → (1, 1, 72, 72) tensor
  → model(rpm) → (_, logits)               # unpack features + logits; use logits only
  → F.softmax(logits) → probs
  → assess_signal_quality()                 # independent quality score
  → estimate_heart_rate()                   # scipy peak detection
  → uncertainty check                       # 3-condition flag
  → build_health_report()                   # structured result dict
  → Streamlit render + PDF download
```

## UNCERTAINTY DETECTION LOGIC

A result is flagged uncertain when ANY of these hold:
1. `top_probability < 0.60` (low confidence)
2. `prediction_margin < 0.15` (top two classes too close)
3. `quality['grade'] == 'Poor'` (signal quality too low)

When uncertain: Gemini AI summary is **skipped**, risk defaults to Moderate.

## DEVICE-SPECIFIC CSV HANDLING

| Source | Column detection | Sampling rate |
|--------|-----------------|---------------|
| Apple Watch | Looks for `ecg`, `ecg_mv`, `value`, `microvolts`, `uV` | 512 Hz → resampled to 100 Hz |
| Samsung Watch | First numeric column | 500 Hz → resampled to 100 Hz |
| Generic/Research | First numeric column | User-specified (default 100 Hz) |

## ANTI-PATTERNS (THIS PACKAGE)

- **Don't call `model(rpm)` and use only one return value.** `FeatureExtractionModule.forward()` returns `(features, logits)`. Always: `_, logits = model(rpm)`.
- **Don't load `best_model.pth` without `map_location='cpu'`** — the checkpoint was trained on GPU; loading without the flag crashes on CPU-only machines.
- **Don't remove the `@st.cache_resource` decorators** from `load_model()` or `initialize_gemini()` — model re-loads on every interaction without them (catastrophic for performance).
- **Don't modify `model.py` Config values for inference** — `OVERSAMPLE_RATIO`, `DEBUG_MODE`, `NUM_EPOCHS` are training artifacts. App only uses `SIGNAL_LENGTH`, `RPM_SIZE`, `CLASS_NAMES`, `NUM_CLASSES`.
- **Don't add `!pip install` lines** — the commented-out block at the top of `model.py` (lines 4–9) is a Jupyter relic. Keep commented out; real deps go in `requirements.txt`.
- **Don't forget Git LFS on new clones** — `best_model.pth` is an LFS pointer; `load_model()` will silently return an uninitialized model if LFS wasn't pulled.

## UNIQUE PATTERNS

- **Two-path model loading:** `load_model()` checks `os.path.dirname(__file__)` first, then cwd fallback. This handles both `streamlit run app.py` (from `streamlit_app/`) and Streamlit Cloud execution contexts.
- **Gemini prompt engineering:** Temperature=0, top_p=0.2 for deterministic clinical output. Prompt enforces exactly 3 one-sentence sections + confidence/HR line.
- **Padding strategy differs from training:** `process_single_signal()` zero-pads short signals (vs training's last-value-pad in `adjust_signal_length()`). Intentional — zero-pad is safer for unknown sources.

## COMMANDS

```bash
# Run locally
cd streamlit_app
streamlit run app.py

# Deploy to Streamlit Cloud
# Main file: streamlit_app/app.py
# Add GEMINI_API_KEY in Streamlit Cloud Secrets dashboard

# Gemini key (local)
set GEMINI_API_KEY=your_key          # Windows CMD
$env:GEMINI_API_KEY="your_key"       # PowerShell
# or create .streamlit/secrets.toml:
# GEMINI_API_KEY = "your_key"

# Verify LFS model
git lfs ls-files                     # should show best_model.pth
ls -lh best_model.pth                # should be >1MB, not 134 bytes (pointer)
```

## NOTES

- **`data.csv`** — sample ECG included for offline/demo testing; format is single numeric column at 100 Hz.
- **Risk levels by class:** NORM→Low, HYP→Moderate, STTC→High, CD→High, MI→Urgent. Uncertain result overrides to Moderate (or High if signal is Poor).
- **Heart rate normals:** <50 bpm → "Low for resting adults", >100 bpm → "High for resting adults", else normal range. Affects risk when NORM predicted with abnormal HR.
- **Gemini model string:** `'gemini-2.5-flash'` hardcoded in `initialize_gemini()` — update here if switching model versions.
