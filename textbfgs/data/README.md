Hard subsets for TextBFGS (HumanEval-Hard / MBPP-Hard)
======================================================

This folder provides scripts to *generate* two \"hard\" subsets, which allow you to reproduce the hard
set construction on your own environment.

How \"Hard\" is defined
-----------------------

For each task:
1. Build a plain prompt using the benchmark `prompt` field.
2. Call the model once to generate a candidate solution.
3. Evaluate the candidate using EvalPlus.
4. Mark the task as **hard** if `base_all_passed` is false **OR**
   `plus_all_passed` is false.

Scripts
-------

- `generate_humaneval_hard.py`: generate `humaneval-hard` JSONL
- `generate_mbpp_hard.py`: generate `mbpp-hard` JSONL

Each script writes a JSONL file containing (at minimum):
- `task_id`
- `prompt` (the benchmark prompt)
- `entry_point`
- `initial_solution` (model output after cleaning)
- base/plus pass statistics and booleans

Outputs are written under:
- `textbfgs/data/humaneval-hard/`
- `textbfgs/data/mbpp-hard/`

Preparing EvalPlus datasets
---------------------------

The scripts call EvalPlus loaders (`get_human_eval_plus`, `get_mbpp_plus`) and
use EvalPlus for evaluation. Make sure EvalPlus datasets are available:

```bash
python -m evalplus.humaneval
python -m evalplus.mbpp
```

Notes
-----

- We do not redistribute any benchmark files here to respect original licenses.
- All model endpoints and API keys in the scripts are placeholders and must be
  configured by you.

