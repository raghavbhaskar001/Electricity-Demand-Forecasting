# Contributing Guidelines

This repository documents an academic engineering research project developed in the Department of Electrical Engineering at Thapar Institute of Engineering & Technology, Patiala. Contributions that improve reproducibility, statistical evaluation, documentation clarity, or software reliability are welcome.

## Core Principles

1. **Methodological Integrity:** Any modification to `src/dataset.py`, `src/baselines.py`, `src/evaluation.py`, or `src/statistical_tests.py` must preserve chronological ordering, the `L = 16` boundary purge gap between train/validation/test splits, training-only `StandardScaler` fitting, and physical-unit (`kW`) inverse transformation prior to metric calculation.
2. **No Data Leakage:** Feature scalers and target scalers must never be fit on validation or test partitions.
3. **Empirical Transparency:** All reported numbers in documentation or UI views must correspond directly to saved outputs in `artifacts/metrics/`. Do not extrapolate or manually edit metric tables without regenerating them from the evaluation scripts.

## Running Verification Checks Before Submitting Changes

Before opening a pull request or proposing changes, run the unit test suite from the repository root:

```bash
python -m unittest discover tests -v
```

If your changes affect the dashboard or the interactive prediction simulator, regenerate the static bundles and verify that all 44 tests pass:

```bash
python app/build_dashboard_data.py
python app/build_simulator_bundle.py
python -m unittest discover tests -v
```

## Reporting Issues

When reporting a bug or discrepancy, please include:
- Operating system and Python version (`python --version`)
- Installed PyTorch and NumPy versions
- The exact command executed and full traceback or metric mismatch
