TextBFGS Supplementary Material
===============================

This folder contains the anonymized supplementary material for the TextBFGS
experiments described in the paper. All company-specific identifiers, private
URLs, and API keys have been removed or replaced with generic placeholders.

Directory layout
----------------

- `code/`
  - `client.py`: minimal Qwen-compatible client used in the runs. You must
    provide your own `api_key` and `base_url` to actually send requests.
  - `optimizer/`: snapshot of the T-BFGS optimizer and the associated
    Hessian-Proxy Knowledge Base (HPKB) implementation and prompts.
- `examples/`
  - `humaneval_textbfgs.py`: end-to-end script that runs TextBFGS on
    HumanEval/HumanEval+ using EvalPlus.
  - `mbpp_textbfgs.py`: end-to-end script that runs TextBFGS on MBPP/MBPP+.
- `data/`
  - `README.md`: overview of the benchmarks and how to generate hard subsets.
  - `generate_humaneval_hard.py`: script to generate HumanEval-Hard (tasks
    where the initial model solution fails HumanEval base or plus tests).
  - `generate_mbpp_hard.py`: script to generate MBPP-Hard (tasks where the
    initial model solution fails MBPP base or plus tests).
  - `humaneval-hard/`: output directory for HumanEval-Hard JSONL files.
  - `mbpp-hard/`: output directory for MBPP-hard JSONL files.

Installation
------------

We recommend using Python 3.10+ and a fresh virtual environment:

```bash
pip install -r requirements.txt
```

This installs:
- `textgrad` for textual optimization framework,
- `evalplus` for HumanEval/MBPP evaluation,
- `chromadb` for Hessian-Proxy Knowledge Base

Configuring the language model
------------------------------

The scripts require the following parameters (all are mandatory):

- `--model`: backbone LLM name (used for code generation and optimization)
- `--base-url`: chat endpoint base URL (the script will append `/chat/completions`)
- `--api-key`: chat endpoint API key (e.g., `Bearer YOUR_API_KEY_HERE`)
- `--embedding-api-url`: embedding endpoint URL (for HPKB retrieval)
- `--embedding-api-key`: embedding endpoint API key
- `--embedding-api-model`: embedding model name (default: `Qwen3-Embedding-8B`)

You can use any large language model that supports:
- chat-style prompts (`messages = [{\"role\": \"system\"|\"user\", ...}]`),
- streaming responses with `data: ...` JSON lines containing
  `choices[0].delta.content`.

Running TextBFGS on HumanEval/MBPP
------------------------------

For instance, from the repository root:

```bash
python textbfgs/examples/humaneval_textbfgs.py \
  --model Qwen3-235B-A22B \
  --base-url "https://api.example.com/v1" \
  --api-key "Bearer YOUR_API_KEY_HERE" \
  --embedding-api-url "https://api.example.com/embeddings" \
  --embedding-api-key "YOUR_EMBEDDING_API_KEY_HERE" \
  --embedding-api-model "Qwen3-Embedding-8B"
```

This will:
- call the model to produce an initial solution for each HumanEval task,
- iteratively refine the solution using TextBFGS (T-BFGS optimizer),
- write per-task results under `<model>_runs/humaneval_t_bfgs_*`,
- export a JSONL file in `evalplus_results/humaneval/` that can be evaluated
  with `evalplus.evaluate`.

Knowledge base (HPKB)
---------------------

The Hessian-Proxy Knowledge Base stores successful optimization trajectories:
textual gradients, abstract operators, and code before/after the fix (optional). The T-BFGS optimizer retrieves similar trajectories to approximate the inverse Hessian of the semantic landscape.

Key switches (CLI in both `humaneval_textbfgs.py` and `mbpp_textbfgs.py`):
- `--kb-mode` (`use-and-store` | `use-only` | `store-only` | `none`)
  - `use-and-store`: retrieve + persist new traces
  - `use-only`: retrieve only, no new traces
  - `store-only`: no retrieval, but store successful traces
  - `none`: disable HPKB
- `--include-code-examples`: retrieve/store code before/after.
- ChromaDB persists under `<model>_runs/.../t_bfgs_kb_global/`.

To enable HPKB in practice, point the embedding URL/key/model to your own service; the defaults are anonymized placeholders.