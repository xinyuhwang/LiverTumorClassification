# E0X · vN — <short name>

| | |
|---|---|
| Status | planned / running / done / abandoned |
| Compared with | vN-1 (or another experiment's version) |
| Change | the single thing that differs |
| Code | git commit `<hash>`; flags `--set …` |
| Run | AICR job id(s); `$HERALD_STORE/results/runs/<pipeline>/<run_name>` |
| Date | YYYY-MM-DD |

## Hypothesis

What we expect to happen, and why.

## How to reproduce

See `run.sh`.

## Results

Paste the output of:

```bash
python common/evaluate.py summary <this run> --by-type --md
python common/evaluate.py compare <previous run> <this run> --md
```

| metric | n | a | b | diff | ci_low | ci_high | p_boot | … |
|---|---|---|---|---|---|---|---|---|

Per-case results are in `results/per_case_test.csv`.

## Observations

What the numbers and failure cases show.

## Decision

Keep / discard / follow up with vN+1, and why.
