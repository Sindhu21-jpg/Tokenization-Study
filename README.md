# Tokenization Study

A Python project for comparing tokenization methods on social-media and biomedical text. The experiments measure token counts, held-out out-of-vocabulary rates, token lengths, preservation of domain-specific text units, and optional sentiment-classification performance.

## Main files

- `tokenizer_experiments.py` runs the multi-seed experiments and writes results to `my_results_v2/`.
- `app_v2.py` provides a Streamlit dashboard for the generated results and an interactive text analyzer.

## Setup

Python 3.10 or later is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Datasets

The raw datasets are not included in this repository. Provide local copies of:

- Sentiment140, as its headerless CSV file.
- PubMed 20k RCT, using its `train.txt` file.

Check the datasets' original sources and terms before downloading or redistributing them. Run the experiment with paths to your local copies:

```powershell
python tokenizer_experiments.py --sentiment "PATH\TO\training.1600000.processed.noemoticon.csv" --pubmed "PATH\TO\train.txt" --seeds 42 43 44 --out my_results_v2
```

The script also supports `--detok` to detokenize PubMed sentences and `--skip_downstream` to omit the sentiment-classification experiment. Run `python tokenizer_experiments.py --help` for all options. The first run may download tokenizer resources and the `bert-base-uncased` model.

## Dashboard

Generate the experiment results first, then start the dashboard:

```powershell
streamlit run app_v2.py
```

The dashboard reads `my_results_v2/` by default. You can enter a different results folder in its sidebar.

Generated results, raw data, virtual environments, and project documents are excluded from Git by `.gitignore`; regenerate or store them separately as needed. No license is included, so the project remains under its author's copyright unless a license is added.
