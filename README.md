# GNN Honeytrap Detection

Hackathon project: detecting honeytrap (social-engineering) accounts in a social
network using a Graph Neural Network over user/message interaction graphs,
with a Streamlit demo app.

## Structure

- **`honey-gnn/`** — the main application.
  - `app.py` — Streamlit demo (GNN predictions combined with rule-based risk scoring, network visualization).
  - `model.py` — heterogeneous GNN model (`HoneytrapGNN`) and graph construction, training, saves `honeytrap_model.pth`.
  - `preprocessing.py` — conversation parsing and text embedding.
  - `data.py` — dataset preparation (`messages.json`, `user_features.json`, `honeytrap_labels.json`).
  - `archive/` (gitignored) — earlier demo iterations and scratch files kept for reference.
- **`data-synthesis/`** — synthetic dataset generation.
  - `persona_synth.py` / `data_synth.py` — Faker-based persona and interaction generation (`personas.csv`, `infrastructure.csv`, `interactions.csv`).
  - Final message-level conversation data was generated with Gemini (web) rather than these scripts.

## Running the demo

Each folder is a separate [uv](https://docs.astral.sh/uv/) project:

```bash
cd honey-gnn
uv sync
uv run streamlit run app.py
```

Note: `honey-gnn/messages.json` (~124 MB raw message data) and `honey-gnn/processed/`
are gitignored; the demo's trained model checkpoint `honeytrap_model.pth` is committed.
