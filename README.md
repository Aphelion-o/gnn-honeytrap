# GNN Honeytrap Detection

Hackathon project: detecting honeytrap (social-engineering) accounts in a social
network using a Graph Neural Network over user/message interaction graphs,
with a Streamlit demo app.

## Structure

- **`honey-gnn/`** — the main application.
  - `app.py` — Streamlit demo (GNN predictions combined with rule-based risk scoring, network visualization).
  - `model.py` — heterogeneous GNN model (`HoneytrapGNN`) and graph construction, training, saves `honeytrap_model.pth`.
  - `preprocessing.py` — conversation parsing and text embedding.
  - `data.py` — early prototype: random synthetic data generator + simple GNN (not used for the final model).
  - `archive/` (gitignored) — earlier demo iterations and scratch files kept for reference.
- **`data-synthesis/`** — synthetic dataset generation.
  - `persona_synth.py` / `data_synth.py` — Faker-based persona and interaction generation (`personas.csv`, `infrastructure.csv`, `interactions.csv`).
  - Final message-level conversation data was generated with Gemini (web) rather than these scripts.
  - `gemini_prompt_draft.md` — draft of the Gemini prompt (the final prompt was not preserved).

## Running the demo

Each folder is a separate [uv](https://docs.astral.sh/uv/) project:

```bash
cd honey-gnn
uv sync
uv run streamlit run app.py
```

Requires Python 3.13. `torch` is installed from PyPI: on Windows/macOS that's the CPU build, on Linux
it includes CUDA. The model is small, so CPU is fine. The first run downloads the MobileBERT model
(`google/mobilebert-uncased`) from Hugging Face, so it needs internet access.

## Data

The final model (`honeytrap_model.pth`) was trained by `honey-gnn/model.py` on:

- `honey-gnn/data.json` — 789 synthetic chat messages between 58 users, generated with Gemini.
- `honey-gnn/processed/` — `data.json` after `preprocessing.py` (per-user features and
  per-message MobileBERT embeddings). This is what `model.py` loads for training.

To retrain: `uv run python model.py` (or re-run `uv run python preprocessing.py` first to rebuild `processed/`).

`honey-gnn/user_features.json`, `honeytrap_labels.json` and `honeytrap_users.json` come from the
`data.py` prototype's random generator; its ~124 MB `messages.json` is gitignored.
