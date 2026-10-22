# Explainable Hybrid Music Recommendation

An end-to-end research notebook for explainable music recommendation using three complementary signals: user listening behavior, Spotify-style audio features, and semantic tags. The project compares collaborative filtering, audio content-based, semantic, simple hybrid, and explainable hybrid recommenders, then evaluates activity segments, fusion strategies, explanations, and cold-start robustness.

## Results

The reported experiments use a stratified sample of an integrated Million Song Dataset, Spotify, and Last.fm corpus containing 30,000 users, 19,054 tracks, and 216,170 interactions. Evaluation uses leave-one-out testing with 100 sampled negatives per user.

| Model or condition | NDCG@10 | Result |
| --- | ---: | --- |
| Collaborative Filtering | 0.334 | Baseline |
| Content-Based Audio | 0.081 | Audio-only baseline |
| Semantic Tag Model | 0.331 | Semantic-only baseline |
| Simple Hybrid Fusion | 0.389 | Best aggregate ranking score |
| Proposed Explainable Hybrid | 0.376 | Multimodal recommendations with evidence |

Key findings:

- The proposed hybrid improves on collaborative filtering by +6.1% to +27.8% across the very-low through medium activity segments, with a +13.1% lift for high-activity users.
- Audio and semantic signals support cold-start recommendations: NDCG@10 is 0.240 for a new track and 0.239 for a new user with one interaction.
- The full semantic encoder reaches NDCG@10 = 0.374 with 100% explanation coverage.
- Under combined missing-data stress, performance decreases from 0.376 to 0.303 NDCG@10, a 19.5% drop.

## Repository contents

- `explainable_hybrid_music_recommendation_all_rqs.ipynb` - full executable notebook covering the research questions and artifact generation.
- `outputs/` - generated figures, tables, summaries, and evaluation artifacts.
- `doc/RESULTS_SECTION.md` - detailed research-question results, figures, and tables.
- `doc/results_tables/` - tabular results used in the report.
- `doc/dataset_figures/` - dataset and experimental-setup figures.
- `data/Music Info.csv` - project dataset used by the repository.

## Running the notebook

Open the notebook in Jupyter or Kaggle and run it from top to bottom. The notebook searches for compatible CSV inputs, prepares the interaction, audio, and semantic features, evaluates all model variants, and writes results to `outputs/`. When a compatible external dataset is unavailable, it can generate a reproducible synthetic dataset so the pipeline still runs end to end; set `USE_SYNTHETIC_IF_NO_DATA = False` when real data is required.

## Documentation

The detailed results and research narrative are in [`doc/RESULTS_SECTION.md`](doc/RESULTS_SECTION.md). Operational notes and dataset compatibility information are in [`doc/README.md`](doc/README.md).
