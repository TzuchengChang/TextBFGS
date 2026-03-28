# TextBFGS: A Case-Based Reasoning Approach to Code Optimization via Error-Operator Retrieval

Official implementation accompanying *TextBFGS* (ICCBR 2026, LNCS). Canonical repo: [github.com/TzuchengChang/TextBFGS](https://github.com/TzuchengChang/TextBFGS). If you use the full **TextEvolve** workspace locally, the camera-ready LaTeX source lives at `ICCBR_2026_LNCS_Template/main.tex` (sibling directory to `TextBFGS/`).

## Overview

Iterative code generation with LLMs can be viewed as optimization guided by textual feedback. TextBFGS is a **Case-Based Reasoning (CBR)** framework inspired by **Quasi-Newton** methods: instead of stateless first-order search (akin to SGD on text), it maintains a dynamic **Hessian-Proxy Case Base (HPCB)** of historical *error-to-operator* trajectories to approximate **semantic curvature** (an inverse-Hessian role in the discrete space).

- **Retrieve:** query analogous correction patterns by **error/gradient similarity** (not by problem-input similarity), enabling cross-domain transfer of debugging logic.
- **Reuse / Revise (One-Pass):** a single LLM call produces diagnosis (`<GRADIENT>`), abstract rule (`<OPERATOR>`), and updated code (`<IMPROVED>`), fusing “gradient” and “update” steps.
- **Retain:** successful `(gradient, operator)` cases are written back into the case base so the optimizer **self-evolves**.

The implementation uses **ChromaDB** for embedding-based retrieval (same role as the paper’s case base) and follows the **EvalPlus** protocol for HumanEval / MBPP (**base** vs **plus** pass rates).

## Method (aligned with the paper)

| Idea | Role in code |
|------|----------------|
| Textual gradient \(g_t\) as *target problem* | Error feedback from execution / evaluator |
| \(g_{t-1}\) as query (“semantic momentum”) | Retrieval query for the next step |
| Top-\(k\) neighbors in embedding space | `Retrieve(M, query=g_{t-1}, k)` |
| HPCB \(\mathcal{M} = \{(g_i, \mathcal{O}_i)\}\) | Stored trajectories in Chroma (optional code before/after) |
| Baselines in paper | TextGrad, TextGrad-Momentum, TextBFGS (w/o CB), TextBFGS-REMO (input-similarity retrieval) |

**Privacy / deployment:** this tree is suitable for public release—replace API keys, base URLs, and internal identifiers with your own endpoints before running.

## Experimental setup (from the paper)

- **Backbone:** Qwen3-235B-A22B; **embeddings:** Qwen3-Embedding-8B; **case store:** ChromaDB; **max iterations:** 20 per task; reasoning/thinking disabled for the reported setting; decoding hyperparameters as in the paper.
- **Benchmarks:** EvalPlus on **HumanEval** and **MBPP**; **hard** subsets retain tasks where the initial Pass@1 attempt scores 0 (**HumanEval-Hard:** 45 tasks; **MBPP-Hard:** 117 tasks).
- **Metrics:** **Base Pass** (standard tests) and **Plus Pass** (EvalPlus augmented tests).

Bootstrapping the case base (paper) uses **TextBFGS (w/o CB)** over hard subsets (e.g. 3 epochs); stored tuples include pre/post text, gradient, and operator where applicable.

## Repository layout

```text
TextBFGS/
├── requirements.txt
├── README.md
└── textbfgs/
    ├── code/
    │   ├── client.py              # minimal Qwen-compatible chat client
    │   └── optimizer/             # T-BFGS optimizer, prompts, HPCB / “HPKB” memory
    ├── examples/
    │   ├── humaneval_textbfgs.py  # HumanEval / HumanEval+ via EvalPlus
    │   └── mbpp_textbfgs.py       # MBPP / MBPP+
    └── data/
        ├── README.md              # hard subsets & EvalPlus notes
        ├── generate_humaneval_hard.py
        ├── generate_mbpp_hard.py
        ├── humaneval-hard/        # generated JSONL output
        └── mbpp-hard/
```

## Installation

Python 3.10+ and a virtual environment are recommended:

```bash
pip install -r requirements.txt
```

Pulls in **textgrad**, **evalplus**, **chromadb**, and other dependencies listed in `requirements.txt`.

## Configuring the language model

Scripts require (typical flags):

| Parameter | Purpose |
|-----------|---------|
| `--model` | Backbone LLM name |
| `--base-url` | Chat API base URL (often `/chat/completions` appended by client) |
| `--api-key` | Chat API key (e.g. `Bearer …`) |
| `--embedding-api-url` | Embedding endpoint for case retrieval |
| `--embedding-api-key` | Embedding API key |
| `--embedding-api-model` | Embedding model (paper default: `Qwen3-Embedding-8B`) |

The client expects chat APIs with `messages`-style prompts and streaming `data: …` lines exposing `choices[0].delta.content`.

## Running TextBFGS on HumanEval / MBPP

From the **TextBFGS** repository root:

```bash
python textbfgs/examples/humaneval_textbfgs.py \
  --model Qwen3-235B-A22B \
  --base-url "https://api.example.com/v1" \
  --api-key "Bearer YOUR_API_KEY_HERE" \
  --embedding-api-url "https://api.example.com/embeddings" \
  --embedding-api-key "YOUR_EMBEDDING_API_KEY_HERE" \
  --embedding-api-model "Qwen3-Embedding-8B"
```

Behavior:

- Generate an initial solution per task, then iterate with **TextBFGS** (T-BFGS optimizer).
- Write runs under `<model>_runs/humaneval_t_bfgs_*` and EvalPlus-style JSONL under `evalplus_results/humaneval/` for downstream `evalplus.evaluate`.

Adjust `mbpp_textbfgs.py` similarly for MBPP.

## Hessian-Proxy case base (Chroma / “HPKB”)

The case base stores successful trajectories: textual gradients, abstract operators, and optionally code before/after. Retrieval approximates inverse-Hessian structure by reusing **operators** matched to **error dynamics**.

CLI highlights (see `humaneval_textbfgs.py` / `mbpp_textbfgs.py`):

- `--kb-mode`: `use-and-store` | `use-only` | `store-only` | `none`
- `--include-code-examples`: include code before/after in retrieve/store
- Persistence: `<model>_runs/.../t_bfgs_kb_global/`

Point embedding URL, key, and model at your own service; defaults in scripts are placeholders.

## Case study (paper)

On **`HumanEval/127`**, logs show a setting where the initial solution and **TextGrad** remain failing while **TextBFGS** reaches **pass/pass** on base/plus—illustrating recovery from a boundary-sensitive bug (e.g. off-by-one in interval length) via retrieved correction patterns. Full narrative and code excerpts are in the paper’s **Case Study** section (LNCS draft: `ICCBR_2026_LNCS_Template/main.tex` inside the TextEvolve project).

## Citation

If you use this code, please cite the TextBFGS paper (ICCBR 2026 proceedings; BibTeX will match your camera-ready entry). Example:

```bibtex
@inproceedings{zhang2026textbfgs,
  title     = {TextBFGS: A Case-Based Reasoning Approach to Code Optimization via Error-Operator Retrieval},
  author    = {Zhang, Zizheng and others},
  booktitle = {Proceedings of ICCBR},
  year      = {2026},
  note      = {Replace with official publisher metadata when available}
}
```

## Keywords (paper)

Case-Based Reasoning · Large Language Models · Code Optimization · Retrieval Augmented Generation · Code Generation
