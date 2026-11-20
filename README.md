# Promptbook request and evaluation

Workflows for preparing, running, evaluating and revising **codebooks for LLM-based coding** of open survey answers (promptbooks), used for the open questions of the survey *Narratives of the European Energy Crisis* (2025). The LLMs run on Nebula (OpenAI-compatible endpoint; gpt-oss) or the OpenAI API.

| Notebook | Step |
|---|---|
| `01_create_sample_translate.ipynb` | Translate answers to English with an LLM; draw a stratified sample and create the Excel sheet for manual annotation (gold standard) |
| `02_llm_coding.ipynb` | Send answers + codebook to the LLM; one JSON file of code confidences per run |
| `03_llm_evaluation.ipynb` | Krippendorff's α between runs and against the gold standard, confusion matrix, F1 per code, Excel export of false positives/negatives for codebook revision |
| `04_intercoder_validation.ipynb` | Agreement between human coders |
| `05_causal_mapping.ipynb` | Experiment: extraction of causal claims with a causal-mapping codebook |

| File | Content |
|---|---|
| `codebooks/` | The codebooks used (causes, consequences, responses, lessons) and the causal-mapping codebook and prompt |
| `MaxNLP.py` | LLM clients (OpenAI, Nebula), codebook → JSON schema, promptbook coding, evaluation (Krippendorff's α, disagreement sheets), text metrics, causal-claim extraction |
| `MaxStat.py` | Statistics helpers (group comparisons, associations, heatmaps) |

The modules are maintained in one place (`_shared_modules` of the author's working folder) and copied unchanged into each repository that uses them.

## Setup

```bash
pip install -r requirements.txt
cp openAI_key.example.py openAI_key.py   # OpenAI and/or Nebula credentials; gitignored
```

Tested with Python 3.12 and pandas 3.1; the exact versions are pinned in `requirements.txt`.

Instead of a key file in the repository folder you can keep the key files in one central folder outside the repository and add that folder to `PYTHONPATH`; the notebooks load them with `import openAI_key` / `import qualtrics_key` / `import API_key` and find them there.


Cells tagged `api` call the LLM and were **not executed** for this repository.

## Dummy data

`dummy_data/` keeps the original file names, but all content is synthetic:
- 12 citizen and 4 expert rows in the survey's column structure, with random answers and short invented open answers;
- random coding runs and random manual codings (`MANUAL CODING/`);
- an empty causal-mapping annotation sheet.

The original gold standard, coding runs and results are not part of the repository. All numbers shown in the notebooks come from random data.

The analysis of the coded answers (narratives, response explorer) is in `energy-crisis-narratives-pipeline`.

## Changes from the first version of this repository

Notebooks renamed in workflow order and updated to the latest versions. Codebooks added. Dummy data and credential template added. `2026-06-18 LLM data analysis.ipynb` moved to `energy-crisis-narratives-pipeline`.

## Author and license

Maximilian Roßmann. Licensed under [CC BY 4.0](LICENSE).
