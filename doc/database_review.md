# Database Review and Code Compatibility

## Dataset Inventory
- `data/User Listening History.csv`
  - ~9,711,301 interaction rows
  - Columns: `track_id`, `user_id`, `playcount`
- `data/Music Info.csv`
  - ~50,683 track metadata rows
  - Columns include: `track_id`, `name`, `artist`, `tags`, `genre`, `year`, and Spotify-style audio features.
- `data/MP3-Example/`
  - 1,500 mp3 files across 15 genre folders (100 each).
  - Current notebook does not depend on mp3 files.

## Key Data Findings
- Unique users in interactions: ~962,037
- Unique tracks in interactions: ~30,459
- Unique tracks in metadata: ~50,683
- Track overlap: **100% of interaction tracks exist in metadata**
- Interaction sparsity: extremely high (~0.99967), expected for implicit-feedback recommenders.
- `playcount` is numeric (min 1, max 2948, mean ~2.63).
- `genre` missing rate in metadata is high (~55.9%).
- `tags` coverage is strong (~97.8% non-missing).

## Compatibility Review (Notebook vs Database)
Status: **Compatible after fixes**.

What now matches this database:
- UTF-8 BOM in headers is handled (`\ufefftrack_id` -> `track_id`).
- `playcount` is detected as interaction strength and mapped to `play_count`.
- Interaction and metadata tables are automatically discovered from CSV files.
- Loader uses chunked reading for large interaction file.
- Metadata/audio columns in `Music Info.csv` are correctly detected for audio + semantic branches.
- Filtering to tracks present in metadata is safe (no interaction track loss in this dataset).

## Practical Implications
- Because user count is very large, notebook sampling controls are necessary for Kaggle runtime.
- Because `genre` is sparse, semantic analyses should rely more on `tags` + `artist` than genre-only configs.
- MP3 files are optional for this thesis pipeline; they are not required for RQ1–RQ7 artifact generation.
