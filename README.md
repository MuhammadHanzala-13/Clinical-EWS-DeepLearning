# Clinical Early Warning System (EWS) — Deep Learning

This project is a **deep learning Clinical Early Warning System** that tries to predict whether a patient is **likely to deteriorate** (binary classification).

The notebook trains and compares multiple models using:
- **Patient vitals** (tabular/time-series style)
- **Clinical notes** (free-text)

It also emphasizes a key clinical requirement: **minimize false negatives** (missing a deterioration). In other words, **Recall** is treated as the most important metric.

---

## What’s inside

### Models implemented (3 generations)

#### Generation 1 — DNN baseline (vitals)
- Feed-forward neural network with:
  - Batch Normalization
  - Dropout
- Compares optimizers:
  - **SGD** (with momentum)
  - **Adam**

#### Generation 2 — Recurrent models (time-series vitals)
Vials are converted into short sequences (windows) and modeled using:
- **LSTM** (real-time friendly; uses past only)
- **GRU** (similar to LSTM, usually faster)
- **BiLSTM** (better for offline learning, **not** suitable for real-time monitoring)

#### Generation 3 — ClinicalBERT (clinical notes)
Uses **Bio ClinicalBERT** on text notes.
Two training strategies:
- **Frozen base**: cache BERT embeddings once, train only the classification head
- **Full fine-tuning**: train the whole BERT model

Also includes:
- Attention visualization (heatmap) for interpretability

---

## How to run

The main file is:
- **`ClinicalEWS-DeepLearning.ipynb`**

### 1) Open the notebook
Open `ClinicalEWS-DeepLearning.ipynb` in VSCode/Jupyter.

### 2) Install dependencies (inside the notebook)
The notebook installs required packages using `pip install ...`.

### 3) Run cells
Run cells from top to bottom.

> Note: The current notebook uses **synthetic data** (generated inside the notebook). It includes code comments showing where you would replace this with real CSV loading if you have a dataset.

---

## Dataset (synthetic in this notebook)

The notebook creates a dataset with:
- **Features (vitals):**
  - heart_rate, spo2, temperature, sbp, dbp, resp_rate, wbc, lactate, gcs, age, gender
- **Clinical note text:** short positive/negative style notes
- **Label:** `1 = deteriorating`, `0 = stable`

Missing values are added intentionally and then filled using **KNN imputation**.

---

## Preprocessing steps

1. **KNNImputer**: fill missing vital signs
2. **StandardScaler**: normalize vitals
3. Add small **noise** to reduce unrealistically perfect separation
4. Split data into **train / validation / test**

---

## Metrics and why Recall matters

For every model, the notebook prints:
- Accuracy
- Precision
- **Recall**
- F1-score

Clinical reason:
- **False negative (missed deterioration)** can be life-threatening.
- **False positive** triggers an extra review, which is usually less dangerous than missing deterioration.

That’s why the notebook highlights **Recall as critical**.

---

## Output files produced by the notebook

Depending on which cells you run, the notebook may save:
- `optimizer_comparison.png`
- `rnn_loss_curves.png`
- `attention_heatmap.png`
- `confusion_matrices.png`
- `model_comparison_bar.png`

---

## Recommended deployment idea (from this notebook)

For a realistic ICU setup:
- Use **ClinicalBERT (full fine-tuning)** for **clinical notes**
- Use **LSTM/GRU** for **real-time vitals**
- (BiLSTM is better only for offline retrospective analysis)

This can also be combined into an **ensemble**.

---

## Notes / Important limitations

- The notebook uses **synthetic data**, so real performance can differ.
- For real clinical use, you must:
  - replace synthetic generation with actual dataset loading
  - verify labeling correctness
  - perform robust evaluation and calibration
  - follow appropriate clinical validation and ethics review

---

## Repository structure

- `ClinicalEWS-DeepLearning.ipynb` — main training and evaluation notebook
- `README.md` — this file

---

## License

- Made By **Muhammad Hanzala** for DeepLearning Assignment by *Sir Hamza*

