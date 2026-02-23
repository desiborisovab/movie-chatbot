# Movie Chatbot

LSTM + Attention language model trained on Wikipedia movie plots.
Retrieves a matching movie from the database and generates a plot continuation.

---

## Project Structure

```
movie_chatbot/
├── config.py       all hyperparameters and shared constants
├── data.py         data loading and text cleaning
├── tokenizer.py    vectorizer build (training) and restore (inference)
├── model.py        LSTM + attention architecture
├── train.py        training entry point
├── generate.py     sampling and generation loop
├── retrieve.py     keyword-based movie retrieval from Spark table
└── chat.py         inference entry point
```

---

## How the Files Relate

```
TRAINING                        INFERENCE
────────────────────────────────────────────────
config.py   ←── shared by both sides ──→  config.py
data.py         (load + clean movies)
tokenizer.py    (build vectorizer)    →   tokenizer.py  (restore vectorizer)
model.py        (architecture)        →   [loaded from MLflow]
train.py        (entry point)         →   chat.py        (entry point)
                                          generate.py    (generation loop)
                                          retrieve.py    (movie search)
```

---

## Training

Open `train.py` in Databricks. All settings are in `config.py`.

Key things to check before running:
- `MOVIE_TABLE` points to the correct table
- `MLFLOW_EXPERIMENT` matches your Databricks username
- `CHECKPOINT_PATH` is writable (`/dbfs/tmp/` works on most clusters)

After training, copy the printed `run_id` into `chat.py`.

---

## Inference

Open `chat.py` in Databricks. Update `RUN_ID` at the top, then run all cells.

```python
print(chat("dark knight batman gotham"))
print(chat("romantic comedy paris 1990s"))
print(chat("science fiction space war"))
```

Temperature guide:
- `0.5`  — confident, coherent
- `0.7`  — good default
- `1.0`  — more varied
- `1.5+` — creative but risks incoherence

---

## Current Status

| Metric      | Value  | Target  |
|-------------|--------|---------|
| val_loss    | 5.66   | < 3.0   |
| Epochs run  | 10     | ~25-30  |
| Steps/epoch | 500    | 2000    |

The model needs more training. With the updated hyperparameters in `config.py`
(2000 steps/epoch, early stopping, cosine LR decay) it should converge in
roughly 20-25 epochs. Consider switching to DistilGPT2 fine-tuning if LSTM
output quality remains poor after full training.