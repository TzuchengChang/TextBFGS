mbpp-hard
=========

This directory is the default output location for the MBPP-Hard generation
script:

- `textbfgs/data/generate_mbpp_hard.py`

Hard definition
---------------

Given a model and generation settings, for each MBPP+ task we:
1. Generate an initial solution with a single model call.
2. Evaluate using EvalPlus (base + plus).
3. Mark the task as **hard** if base is not fully passed OR plus is not fully passed.

Generating the JSONL
--------------------

From the repository root:

```bash
python textbfgs/data/generate_mbpp_hard.py \
  --model Qwen3-235B-A22b \
  --api-key "Bearer YOUR_API_KEY_HERE" \
  --base-url "https://api.example.com/v1" \
  --output textbfgs/data/mbpp-hard/mbpp-hard.jsonl
```

Output format
-------------

Each JSON line contains:
- `task_id`
- `entry_point`
- `prompt`
- `initial_solution`
- `metrics` (pass counts and booleans for base/plus)

