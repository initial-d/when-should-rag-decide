# When Should RAG Decide?

Code, benchmark data, trained policies, and cached run outputs for
*When Should RAG Decide? Evidence Acquisition Before Information-Action Selection*.

The study compares RAG controllers that choose among answering, retrieving
further, reporting a conflict, and abstaining. It audits two controlled
evidence-state benchmarks, rebuilds a 400-question confirmation set that
passes structural checks, and compares diagnose-first and retrieve-first
schedules under a hand-set rule, a logistic point policy, and a point+hull
policy, plus confidence-, complexity-, and gold-state adaptive gates.

## Layout

| Path | Contents |
|---|---|
| `oesc_rag/` | Pipeline: retrieval, claim verification, evidence signature, controllers, adaptive gates, metrics |
| `scripts/` | Benchmark construction, runs, and analyses (numbered as in the paper's protocol appendix) |
| `configs/valid_*.yaml` | One config per backbone × schedule × controller |
| `data/benchmark/` | Development (372), ESC-Valid (400), and confirmation (400) sets |
| `data/models/` | Trained point and point+hull policies |
| `data/results/` | Cached run outputs and analysis JSON behind every table |

Backbones are `deepseek41flash` (DeepSeek-V4.1-Flash) and `qwen38flash`
(Qwen3.8-Flash). Result directories follow `confirm_<backbone>[_suffix]`:
no suffix is diagnose-first, `_rf` retrieve-first, `_rf_fixed` retrieve-first
with the diagnose-first policy, `_rf_goldlabel` the refit without
relabelling, `_rf_tuned` the retuned rule, and `_adaptive` the gates.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.9+
pip install -r requirements.txt
```

## Reproducing the tables (no model calls)

Every analysis reads the cached outputs in `data/results/` and finishes in
seconds. Outputs go to `data/results/*.json` by default; pass `--out` to
write elsewhere.

| Paper result | Command |
|---|---|
| Benchmark validity audit | `python scripts/68_benchmark_validity_audit.py` |
| Main confirmation results (one cell) | `python scripts/63_evaluate_end_to_end.py --benchmark data/benchmark/valid_confirm.json --results-dir data/results/confirm_deepseek41flash_rf --methods point` |
| Paired contrasts and schedule × rule interaction | `python scripts/73_order_rule_interaction.py` |
| Observation timing vs. policy adaptation | `python scripts/76_adaptation_audit.py` |
| Adaptive retrieval gates | `python scripts/74_adaptive_gate_comparison.py` |
| Per-question outcome oracle | `python scripts/75_outcome_oracle.py` |
| Retuned retrieve-first rule | `python scripts/77_retune_rf_rule.py` |
| Offline representation audits | `python scripts/72_representation_audit_current.py` |

A single pairwise bootstrap:

```bash
python scripts/64_bootstrap_joint_lift.py --benchmark data/benchmark/valid_confirm.json \
  --a data/results/confirm_deepseek41flash:point \
  --b data/results/confirm_deepseek41flash_rf:point
```

## Retraining policies

```bash
python scripts/66_train_point_policies.py            # diagnose-first
python scripts/71_train_retrieve_first_policies.py   # retrieve-first (relabelled)
python scripts/71_train_retrieve_first_policies.py --gold-labels
```

Both read the development traces in `data/results/`.

## Running new evaluations (model calls)

Runs need an OpenAI-compatible endpoint. The key is read only from
`OPENAI_API_KEY` or a file passed with `--key-file`; never commit it.

```bash
export OPENAI_API_KEY=...
python scripts/62_run_end_to_end_eval.py \
  --config configs/valid_deepseek41flash_rf_point.yaml \
  --benchmark data/benchmark/valid_confirm.json \
  --methods point --out-dir runs/rf_point \
  --base-url https://your-endpoint/v1
```

Add `--dry-run` to print the plan without calling a model. Gateway routing
and model versions may differ from the original runs, so new runs need not
reproduce the cached outputs exactly.

## Data

Benchmarks are derived from HotpotQA (CC BY-SA 4.0); the derived sets in
`data/benchmark/` keep that license. Labels passed automatic structural
checks only and have not been adjudicated by humans.

## License

Code is released under the MIT License (`LICENSE`).
