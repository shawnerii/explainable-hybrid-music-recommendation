# Agent Handoff Notes

## What Was Validated
- Database schema, overlap, sparsity, and missingness were profiled.
- Notebook was rechecked for compatibility with this exact dataset.
- Major hardcoded sections were replaced by computed experiment logic in RQ4–RQ7.

## Important Runtime Controls
- `MAX_RAW_INTERACTIONS`, `MAX_USERS`, `MAX_ITEMS`, `MAX_INTERACTIONS`, `NEGATIVES_PER_USER`
- If Kaggle runtime is tight, lower these first (especially `MAX_RAW_INTERACTIONS` and `MAX_USERS`).

## Expected Core Inputs
- `data/User Listening History.csv`
- `data/Music Info.csv`

## Fast Sanity Checklist Before Run-All
1. Confirm the two CSV files are readable.
2. Confirm output path is writable (`/kaggle/working/outputs` on Kaggle).
3. Run notebook top-to-bottom once; check final missing-output warning list is empty.

## Known Data Caveats
- Very sparse user-item matrix (normal here).
- High missingness in `genre`; do not over-interpret genre-only results.
