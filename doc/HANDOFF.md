# Project Handoff — Explainable Hybrid Music Recommendation

This document contains everything an agent or collaborator needs to continue this thesis project. Read it in order; later sections build on earlier ones.

---

## 1. Project Overview

**Title:** Explainable Hybrid Music Recommendation Using User Behavior, Audio Features, and Semantic Tag Modeling
**Author:** Shayan Rahimi — MSc Software Engineering, University of Europe for Applied Sciences
**Target venue:** Expert Systems with Applications
**Status:** Code + first full Kaggle run complete; results section drafted; ready for an optional second Kaggle run with code improvements applied.

The project builds a single-notebook recommender system that combines three modalities — collaborative filtering, content-based audio, and semantic tags — into an attention-based explainable hybrid. It answers 7 research questions end-to-end and produces all CSV tables and PDF figures needed for the thesis Results section.

---

## 2. Directory Layout

```
/Users/shayan/Desktop/4th semester/Code/
├── explainable_hybrid_music_recommendation_all_rqs.ipynb   ← the notebook (single source of truth)
├── Shayan_Rahimi_Proposal.pdf                              ← original proposal with dummy/expected results
├── data/
│   ├── Music Info.csv                  (50,683 tracks, ~15 MB)
│   ├── User Listening History.csv      (9.7M interactions, ~600 MB)
│   └── MP3-Example/                    (1,500 mp3s — not used by notebook)
├── outputs/                            ← raw notebook outputs (42 files from last Kaggle run)
│   ├── rq*_t*.csv, rq*_f*.pdf, appendix_*.csv, artifact_index.csv
│   └── notebookddb99450aa.log          ← full Kaggle execution log
└── doc/
    ├── HANDOFF.md                      ← this file
    ├── README.md, agent_handoff.md, database_review.md   (older notes)
    ├── RESULTS_SECTION.md              ← thesis-ready Results section in markdown
    └── results_tables/                 ← 12 thesis-ready CSVs (European format)
        └── Table_1a_*.csv … Table_9_*.csv
```

---

## 3. Dataset

**Source:** Integrated Million Song Dataset + Spotify audio features + Last.fm tags (sourced from Kaggle's "Music Info" + "User Listening History" CSVs).

**Raw statistics:**
- 962,037 unique users; 9.7M interaction rows; 50,683 tracks
- Track overlap between interactions and metadata: 100%
- Interaction sparsity: ~0.99967 (typical for implicit-feedback music data)
- `genre` missingness: ~55.9% (so genre-only analyses are weak)
- `tags` coverage: ~97.8%
- `playcount` is the interaction strength signal

**Notebook scale used in the last successful Kaggle run:** 8,000 users / 11,039 tracks / 40,631 interactions (controlled via `MAX_USERS`, `MAX_ITEMS`, `MAX_RAW_INTERACTIONS`, `MAX_INTERACTIONS`, `NEGATIVES_PER_USER` env vars in Cell 1).

---

## 4. The 7 Research Questions

| RQ | Question (short form) |
|---|---|
| RQ1 | Hybrid framework vs single-source recommendation accuracy |
| RQ2 | Behavioral patterns across user activity levels |
| RQ3 | Audio feature contribution to content-aware recommendation |
| RQ4 | Genre, mood, artist, tag-based semantic metadata utility |
| RQ5 | Best multimodal fusion strategy (accuracy vs interpretability) |
| RQ6 | Quality of multimodal explanation generation |
| RQ7 | Robustness under sparse / cold-start / missing-metadata conditions |

Full text of each RQ lives in the proposal PDF (Research Questions section) and in `RESULTS_SECTION.md`.

---

## 5. Notebook Architecture

23 cells, organized as:

| Cell | Purpose |
|---|---|
| 0 | Title markdown |
| 1 | Global config, imports, runtime control variables |
| 2 | Utility functions: `save_table`, `save_figure`, `normalize_metric_values`, metric functions, `evaluate_rankings` |
| 3 | Data discovery + synthetic fallback dataset |
| 4 | Data preparation and cleaning |
| 5 | Train/test split (leave-one-out), feature matrices (CF/SVD, audio L2-norm, semantic TF-IDF), user profile builders |
| 6 | Scoring functions: `cf_scores`, `audio_scores`, `semantic_scores`, `hybrid_scores`, `proposed_scores` (attention-weighted) |
| 7 | Candidate generation + `evaluate_model_collection` framework |
| 8 | RQ1 markdown |
| 9 | RQ1 code: 5-model comparison + significance tests |
| 10 | RQ2 markdown |
| 11 | RQ2 code: per-activity-segment evaluation + 4-cluster archetype |
| 12 | RQ3 markdown |
| 13 | RQ3 code: permutation importance (Random Forest) + feature ablation + cohort lift |
| 14 | RQ4 markdown |
| 15 | RQ4 code: 4 semantic configurations + Sankey + heatmap + explanation examples |
| 16 | RQ5 markdown |
| 17 | RQ5 code: 4 fusion strategies + learned modality weights heatmap |
| 18 | RQ6 markdown |
| 19 | RQ6 code: explanation methods (template/feature/tag/multimodal) + score decomposition + case study |
| 20 | RQ7 markdown |
| 21 | RQ7 code: cold-start scenarios + stress-test degradation + robustness heatmap |
| 22 | Artifact index — lists every output file with size |

Three modalities are combined as: behavior (CF via 64-component SVD), audio (13 features L2-normalized), semantic (TF-IDF on `genre + artist + tags`, n-grams 1-2, max 3000 features). The proposed model uses an attention mechanism (`dynamic_modality_weights`) that varies weights per user context.

---

## 6. Critical History — Bugs Already Fixed

These were applied to the notebook during prior sessions. **Do not undo them.**

### 6.1 RQ7 sparse matrix flatten (was a hard crash)
**Location:** Cell 21, function `cold_start_new_user`
**Original (broken):**
```python
s_raw = s_proj.toarray().reshape(-1)
```
**Fixed:**
```python
s_raw = np.asarray(s_proj.todense()).flatten()
```
**Why:** `s_proj` is `(1, N)` 2D after sparse `.mean(axis=0).dot(...)`. `reshape(-1)` failed because `np.asarray(..., dtype=float)` inside `normalize_metric_values` couldn't handle the matrix object. The fix forces dense → flat 1D first.

### 6.2 RQ3 per-feature ablation loop vectorized
**Location:** Cell 13, function `audio_scores_with_feature_subset`
**Problem:** Iterated `for u, group in train_df.groupby("user_idx")` *inside* each of 6 ablation runs → quadratic Python loop, took 2+ hours on Kaggle even at 8000 users.
**Fix:** Replaced with vectorized `np.add.at(profiles, uids, item_sub[iids] * wts[:, None])` followed by row-sum normalization. ~100× faster.

### 6.3 Cell 5 `build_user_semantic_profiles` vectorized
**Original (broken — O(N×U) DataFrame scans):**
```python
for u in range(n_users):
    idx = train_frame.loc[train_frame["user_idx"] == u, "item_idx"].values
    ...
```
**Fixed:** Builds a sparse user×item indicator matrix once, then a single `UI.dot(semantic_matrix)`. This was the *real* root cause of the initial slowness — it ran during setup before any RQ.

### 6.4 Cell 17 `items.loc[it, "semantic_text"]` lookups replaced
**Problem:** Inside nested user loops, slow per-row pandas indexing.
**Fix:** Pre-extract `items["semantic_text"].values` to numpy and index with `sem_texts_arr[it]`. ~10× speedup.

---

## 7. Recent Notebook Improvements (Apply To Next Kaggle Run)

These were added in response to the user's review of the first results. They are already in the notebook; running it again on Kaggle will produce cleaner outputs.

| Issue | Cell | Fix |
|---|---|---|
| Inline captions baked into PDFs | 2 | `plot_caption` is now a no-op — figures save without captions (captions live in the thesis document) |
| Figure label overlap | 2 | `save_figure` now calls `tight_layout()` and uses `pad_inches=0.4` |
| RQ1-F1 model label overlap | 9 | x-axis labels rotated 20° |
| RQ2-F1 segment label overlap | 11 | x-axis labels rotated 15° |
| RQ4 Sankey shows "the" / "nan" | 15 | Stopword + nan filter applied to history and recommended-track tokens |
| RQ4 explanation examples show 3 "implicit similarity" rows | 15 | Now selects only top-5 tracks with non-zero semantic score |
| RQ5 modality weight heatmap text small | 17 | Font size 11, bold |
| RQ6 case study had all-zero contributions | 13 | `sample_user` now scans first 500 test users for one whose top-10 candidates have non-zero behavior + audio + semantic scores |
| RQ6-F1 y-tick overlap | 19 | Y-tick font size set explicitly |
| RQ6-F2 was empty (hardcoded axis limits) | 19 | Axis limits now derived from actual data range |
| RQ7 robustness heatmap unreadable | 21 | Font size 11, contrast color logic inverted |

---

## 8. Things That Look Wrong But Are Correct (Do Not "Fix" These)

The user (and reviewers) may flag these — they are all genuine results, not bugs.

| Observation | Why it's correct |
|---|---|
| p-value = 0.0 for Proposed vs Content-Based Audio | Floating-point underflow — actual value is <1e-300. Display as `<0.001`. |
| Significance test signs are mixed (some +, some −) | + = Proposed wins; − = Proposed loses. The proposed model is *worse* than Semantic Tag and Simple Hybrid at this scale (by tiny effect sizes). |
| Cluster C3 has 7,747 users while C1/C2/C4 have ~100 each | Real finding — 97% of users have ≤5 interactions, all clustered into the low-engagement archetype. |
| Recall@10 = Precision@10 × 10 exactly | Mathematical property of leave-one-out: with exactly 1 relevant item per user, Recall@k = Precision@k × k for every model. |
| Audio feature permutation importance values are small (~0.01) | Permutation importance scale depends on the underlying metric; small absolute values are normal. |
| `Full Audio Feature Set` ΔNDCG = 0.0 in ablation table | It's the reference baseline; Δ relative to itself is always 0. |
| `New Item Recommendation` behavior weight = 0.0 | By design — cold-start items have no interaction data, so behavior weight must be 0. |
| Template-Only fidelity = 0.000, coverage = 100% | By design — template explanations are static (0 fidelity) but always available (100% coverage). |
| "No Stress" performance drop = 0.0% | It's the reference baseline. |

---

## 9. First Run Results (8000-user Kaggle run — current ground truth)

The numbers below are from `outputs/` and are reflected in `doc/RESULTS_SECTION.md` and `doc/results_tables/*.csv`. They are what the user has provisionally placed in the proposal PDF.

### RQ1 — Model comparison (NDCG@10)
- Collaborative Filtering 0.182 | Content-Based Audio 0.074 | Semantic Tag 0.298 | Simple Hybrid 0.296 | **Proposed Hybrid 0.290**
- vs Collaborative: Δ=+0.108, p<0.001, d=0.31 (medium effect)
- vs Audio: Δ=+0.216, p<0.001, d=0.56 (large effect)
- vs Semantic Tag: Δ=−0.008, p=0.005, d=−0.03 (negligible)
- vs Simple Hybrid: Δ=−0.006, p<0.001, d=−0.06 (negligible)

**Key narrative:** Proposed model is statistically significantly better than the unimodal baselines but matches the two strong hybrid baselines. Frame as "comparable accuracy + uniquely explainable" — *not* as "outperforms everything."

### RQ2 — Per-segment performance + 4 clusters
| Segment | Users | CF NDCG | Proposed NDCG | Lift |
|---|---|---|---|---|
| Very Low (1–5) | 6,266 | 0.175 | 0.279 | +59.1% |
| Low (6–20) | 1,615 | 0.207 | 0.331 | +60.2% |
| Medium (21–50) | 113 | 0.181 | 0.286 | +57.7% |
| High (>50) | 6 | 0.072 | 0.161 | +123.9% |

High segment has only 6 users → big % lift but unstable. A bigger Kaggle run (30k+ users) would populate this segment meaningfully.

### RQ3 — Audio importance (top 6)
danceability (0.0135) > instrumentalness (0.0108) > acousticness (0.0099) > speechiness (0.0088) > valence (0.0086) > energy (0.0081). Ablation drops are all small (0.0002–0.0026 NDCG), indicating audio contributes only marginally to the overall hybrid score.

### RQ4 — Semantic configurations (NDCG@10)
Genres Only 0.171 → Free-form Tags 0.260 → Artist Metadata 0.275 → Full Semantic Encoder 0.287. The full encoder beats genres-only by +68%.

### RQ5 — Fusion strategies (NDCG@10 / interpretability)
- Weighted Early 0.296 / 4.30
- Late 0.276 / 4.41
- **Embedding-Level 0.300 / 4.28** (winner on NDCG)
- Attention-Based 0.290 / 4.25

Embedding-Level Fusion is the unexpected accuracy winner. Attention-Based is what the proposed framework uses because it gives modality-decomposed weights for explanations. Modality weight pattern is sensible: behavior weight ~0.5 for active users, 0.0 for new items.

### RQ6 — Explanation methods
| Method | Fidelity | Coverage | Diversity | Composite (1–5) |
|---|---|---|---|---|
| Template-Only | 0.000 | 100% | 0.0004 | 2.00 |
| Feature-Based | 0.497 | 100% | 0.0047 | 3.00 |
| Tag-Based | 0.009 | 100% | 0.0202 | 2.04 |
| **Proposed Multimodal** | **0.338** | **100%** | **0.0148** | **2.69** |

Proposed Multimodal is the only one that combines full coverage with non-trivial fidelity *and* balanced diversity. Composite Quality Score is computed algorithmically; **no human user study was conducted** — be explicit about this in the writeup.

### RQ7 — Cold-start + stress
- New User 1 interaction: NDCG=0.194 (67% of full-data ref 0.290)
- New User 5 interactions: NDCG=0.331 (exceeds full-data reference)
- Combined stress (10% audio + 30% semantic + 20% sparsity simultaneously): NDCG drops to 0.242 (16.6% drop)
- Largest single-factor stress: 30% missing semantic tags → 15.1% drop

---

## 10. Expected vs Actual — Honest Diff

The proposal (page 13+) contains dummy/aspirational numbers. Actuals diverge meaningfully:

| RQ | Expected story | Actual story |
|---|---|---|
| RQ1 | Proposed wins everything by a wide margin (NDCG 0.401) | Proposed (NDCG 0.290) ties with Simple Hybrid (0.296) and Semantic Tag (0.298) |
| RQ2 | Accuracy rises monotonically with activity (Very Low 0.246 → High 0.431) | Pattern is irregular due to small High segment (6 users) |
| RQ3 | Energy/tempo dominate audio importance | Danceability/instrumentalness dominate; ablation drops 10× smaller than expected |
| RQ4 | Same ordering | Same ordering, all NDCG values ~25% lower than expected |
| RQ5 | Attention-Based wins | Embedding-Level wins; Attention-Based is second |
| RQ6 | Proposed fidelity 0.86 | Proposed fidelity 0.338 — all explanation metrics ~50% of expected |
| RQ7 | Combined stress drop 16.7% | Combined stress drop 16.6% — closest match across all RQs |

This is **the most important thing for a thesis reviewer**: the writeup needs to honestly reframe the contribution as "comparable accuracy + uniquely explainable + graceful degradation" rather than the proposal's "significantly outperforms all baselines."

---

## 11. How To Run The Code

### Local (Mac M3 Pro, 18 GB RAM)
- Dependencies already installed: numpy, pandas, scipy, scikit-learn, matplotlib, jupyter
- Use small limits in Cell 1 (`MAX_USERS=2000`, `MAX_ITEMS=5000`, `MAX_INTERACTIONS=100000`, `MAX_RAW_INTERACTIONS=1000000`, `NEGATIVES_PER_USER=50`)
- ~5–10 minutes total runtime
- Outputs go to `./outputs/`

### Kaggle (recommended for thesis-quality results)
- Notebook auto-detects `/kaggle/working/outputs/` as output directory
- Session: **Accelerator = None** (you get 30 GB RAM with CPU-only, more than with GPU)
- Upload `Music Info.csv` and `User Listening History.csv` as a Kaggle dataset, then **+ Add Input** to the notebook
- Recommended limits for next run:
  ```python
  MAX_RAW_INTERACTIONS = "5000000"
  MAX_USERS            = "30000"
  MAX_ITEMS            = "20000"
  MAX_INTERACTIONS     = "400000"
  NEGATIVES_PER_USER   = "100"
  ```
- Expected runtime: 2–3 hours (was 30 min at 8k users; scales roughly linearly)

### Downloading Kaggle outputs
The user installed the Kaggle CLI and authenticated with an access token at `~/.kaggle/access_token`. Run from `outputs/`:
```bash
kaggle kernels output shawnerii/notebookddb99450aa -p .
```
This replaces the old `outputs/` files with the new run. The CLI fetches everything in `/kaggle/working/outputs/` plus the kernel log.

---

## 12. Thesis Results Section

`doc/RESULTS_SECTION.md` is the user-facing document for the Results chapter. It contains:
- One short factual paragraph per RQ (no interpretation — interpretation belongs in Discussion which has not been written yet)
- Figure references with file names + one-line captions
- Table references with file names + one-line captions
- Inline markdown tables for each table (8 of them split into 1a/1b, 2a/2b, 6a/6b)

`doc/results_tables/` contains 12 CSV files (one per table) in **European format** (semicolon separators, comma decimals). This was necessary because the Mac is configured with `en_US@rg=dezzzz` locale (German region), and Excel was misreading `0.297` as `297`. With `0,297` and `;` separators, Excel parses correctly. The user pastes these into Word via Open in Excel → Copy cells → Paste into Word.

**Figures intentionally excluded from the Results section:**
- Original RQ4 Sankey (`rq4_f1_semantic_sankey_flow.pdf`) — replaced by the cleaner RQ4 heatmap
- Original RQ6 fidelity-vs-usefulness scatter (`rq6_f2_explanation_fidelity_usefulness.pdf`) — was empty due to hardcoded axis limits (now fixed; will be regenerated on next run)

---

## 13. Known Limitations to Disclose in the Thesis

1. **No human user study** — RQ6 "composite quality score" is computed algorithmically from fidelity and diversity. Be explicit.
2. **Dataset sampling** — 8,000 users out of 962,000 limits statistical power for segment analyses (especially RQ2 High segment with n=6).
3. **Proposed model accuracy** does not significantly exceed Simple Hybrid Fusion. The contribution is explainability + graceful degradation, not raw accuracy.
4. **RQ6 natural-language explanations** use a fixed template. Per-recommendation differentiation comes from the score decomposition, not from varied language. A real production system would want a proper text generator.
5. **Audio features have small impact** on the hybrid. Audio contribution shows mostly in cold-start scenarios (RQ7 new-item case where audio weight = 0.57).

---

## 14. Outstanding Work

The following items are **not done** and would be the next steps:

1. **Optional second Kaggle run** with the post-fix notebook + higher limits (30k users). This would:
   - Populate RQ2 segments properly
   - Produce caption-free figures
   - Fix Sankey artifacts, heatmap readability, RQ6 case study zeros, RQ6-F2 empty plot
2. **Discussion section** of the thesis — interpret why results diverged from expected
3. **Conclusion + Limitations** sections
4. **Decide on Sankey vs heatmap for RQ4** in final document (currently using heatmap)
5. **Generate the actual Word document** — markdown is ready, CSVs are ready, but the user manually assembles the .docx

---

## 15. Useful Commands

**Re-download Kaggle outputs:**
```bash
cd "/Users/shayan/Desktop/4th semester/Code/outputs"
rm -rf ./* && kaggle kernels output shawnerii/<notebook-id> -p .
mv outputs/* . && rmdir outputs   # CLI nests into a subdirectory
```

**Verify a CSV value matches the notebook output:**
```bash
python3 -c "import csv; r=list(csv.DictReader(open('outputs/rq1_t1_comparative_performance.csv'))); print([f'{row[\"Model\"]}: {float(row[\"NDCG@10\"]):.4f}' for row in r])"
```

**Re-run the notebook locally for a quick smoke test:**
```bash
jupyter notebook "/Users/shayan/Desktop/4th semester/Code/explainable_hybrid_music_recommendation_all_rqs.ipynb"
# Set MAX_USERS=1000 in Cell 1, Restart & Run All
```

---

## 16. Quick Reference — File Inventory

| File | Purpose | Authoritative? |
|---|---|---|
| `explainable_hybrid_music_recommendation_all_rqs.ipynb` | The code | **Yes** — single source of truth |
| `Shayan_Rahimi_Proposal.pdf` | Original proposal with dummy results | Reference only — numbers in it are aspirational |
| `data/*.csv` | Raw input data | **Yes** |
| `outputs/*.csv` | Raw notebook outputs from last Kaggle run | **Yes** for raw numbers |
| `outputs/*.pdf` | Raw figures from last Kaggle run | Use for thesis if not regenerating |
| `doc/RESULTS_SECTION.md` | Thesis Results section in markdown | **Yes** for narrative |
| `doc/results_tables/Table_*.csv` | Thesis-ready CSVs (European format) | **Yes** for tables in Word |
| `doc/HANDOFF.md` | This document | **Yes** |
| `doc/README.md`, `agent_handoff.md`, `database_review.md` | Earlier notes | Historical reference |

---

## 17. Contact Context

The user (Shayan) is a graduate student. He runs the code himself; agents help with debugging, results interpretation, and document writing. He is not deeply technical with pandas/numpy — when explaining performance issues prefer plain language over jargon. He has been burned by aggressive autonomous changes — confirm before running long Kaggle jobs or making sweeping edits. He prefers Word for the final thesis document, not LaTeX.

End of handoff.
