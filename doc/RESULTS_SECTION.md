# Results

All results were obtained on a stratified sample of the integrated Million Song Dataset + Spotify + Last.fm corpus comprising 30,000 users, 19,054 tracks, and 216,170 interactions (sparsity ≈ 0.9996, average 7.21 interactions per user). Evaluation follows a leave-one-out protocol with 100 sampled negatives per user. Statistical significance is computed via paired tests on per-user NDCG@10 with Cohen's d as the effect-size measure.

---

## RQ1: How effectively can a hybrid music recommendation framework integrating user behavior, audio features, and semantic tags improve recommendation accuracy compared with single-source recommendation models?

The Proposed Explainable Hybrid achieves NDCG@10 = 0.376. Simple Hybrid Fusion reaches 0.389, Collaborative Filtering reaches 0.334, the Semantic Tag Model reaches 0.331, and Content-Based Audio reaches 0.081. Paired significance tests against the Proposed Hybrid yield p < 0.001 for all four comparisons. Cohen's d is 0.13 against Collaborative Filtering, 0.70 against Content-Based Audio, 0.16 against the Semantic Tag Model, and −0.10 against Simple Hybrid Fusion.

**Figure 1 — Aggregate ranking performance across all five models.**
Insert this figure: `outputs/rq1_f1_model_performance_comparison.pdf`
Caption: Precision@10, Recall@10, NDCG@10, and MAP@10 for Collaborative Filtering, Content-Based Audio, Semantic Tag Model, Simple Hybrid Fusion, and the Proposed Explainable Hybrid.

**Figure 2 — Per-user NDCG@10 distribution comparison.**
Insert this figure: `outputs/rq1_f2_per_user_ndcg_distribution.pdf`
Caption: Distribution of per-user NDCG@10 scores for each of the five models.

**Table 1a — Comparative ranking metrics at K = 10.**
Insert this table: `doc/results_tables/Table_1a_ranking_metrics.csv`
Caption: Precision@10, Recall@10, NDCG@10, and MAP@10 for all five models.

```
Model	Precision@10	Recall@10	NDCG@10	MAP@10
Collaborative Filtering	0.0489	0.4893	0.3338	0.2857
Content-Based (Audio)	0.0168	0.1678	0.0807	0.0547
Semantic Tag Model	0.0468	0.4680	0.3308	0.2887
Simple Hybrid Fusion	0.0534	0.5340	0.3887	0.3432
Proposed Explainable Hybrid	0.0527	0.5265	0.3764	0.3294
```

**Table 1b — Paired significance tests against the Proposed Hybrid.**
Insert this table: `doc/results_tables/Table_1b_significance_tests.csv`
Caption: Paired ΔNDCG@10, p-value, and Cohen's d for the Proposed Hybrid against each baseline.

```
Comparison	ΔNDCG@10	p-value	Cohen's d	Significant?
Proposed vs Collaborative Filtering	+0.043	<0.001	0.13	Yes
Proposed vs Content-Based Audio	+0.296	<0.001	0.70	Yes
Proposed vs Semantic Tag Model	+0.046	<0.001	0.16	Yes
Proposed vs Simple Hybrid Fusion	−0.012	<0.001	−0.10	Yes
```

---

## RQ2: To what extent can user–item interaction patterns capture personalized listening preferences under varying user activity levels?

The dataset partitions into four activity segments by historical interaction count: Very Low (1–5, n = 19,899), Low (6–20, n = 8,510), Medium (21–50, n = 1,438), and High (>50, n = 153). Collaborative Filtering NDCG@10 across the four segments is 0.319, 0.365, 0.348, and 0.379 respectively. The Proposed Explainable Hybrid reaches 0.338, 0.453, 0.445, and 0.429 across the same segments, corresponding to relative NDCG@10 lifts of +6.1%, +24.0%, +27.8%, and +13.1%. Clustering of behavioral embeddings yields four user archetypes: Mainstream Pop Repeaters (27,617 users, behavior-heavy style), High-Energy Explorers (701 users, behavior + audio), Mood-Oriented Evening Listeners (612 users, behavior + semantic), and Genre-Diverse Discovery Users (1,070 users, balanced fusion).

**Figure 3 — Recommendation quality by user activity segment.**
Insert this figure: `outputs/rq2_f1_performance_by_activity.pdf`
Caption: NDCG@10 for Collaborative Filtering, Simple Hybrid Fusion, and the Proposed Explainable Hybrid across the four user activity segments.

**Figure 4 — UMAP projection of user preference embeddings.**
Insert this figure: `outputs/rq2_f2_user_embedding_projection.pdf`
Caption: Two-dimensional UMAP projection of learned user preference embeddings, colored by behavioral cluster.

**Table 2a — Per-segment performance.**
Insert this table: `doc/results_tables/Table_2a_per_segment_performance.csv`
Caption: CF and Proposed Hybrid NDCG@10 by activity segment with relative lift.

```
User Activity Level	Users	CF NDCG@10	Proposed NDCG@10	Lift (%)
Very Low (1–5)	19,899	0.319	0.338	+6.1
Low (6–20)	8,510	0.365	0.453	+24.0
Medium (21–50)	1,438	0.348	0.445	+27.8
High (>50)	153	0.379	0.429	+13.1
```

**Table 2b — Behavioral user archetypes.**
Insert this table: `doc/results_tables/Table_2b_user_archetypes.csv`
Caption: Four behavioral clusters with archetype label, size, and dominant recommendation style.

```
Cluster	Archetype	Users	Dominant Style
C1	Mainstream Pop Repeaters	27,617	Behavior-heavy
C2	High-Energy Explorers	701	Behavior + audio
C3	Mood-Oriented Evening Listeners	612	Behavior + semantic
C4	Genre-Diverse Discovery Users	1,070	Balanced fusion
```

---

## RQ3: How do acoustic attributes contribute to improving content-aware music recommendation performance?

Permutation importance ranks the 13 audio features from highest to lowest as: danceability (0.0169), instrumentalness (0.0144), speechiness (0.0144), acousticness (0.0128), valence (0.0111), duration_ms (0.0106), loudness (0.0100), liveness (0.0085), energy (0.0085), tempo (0.0066), key (0.0054), mode (0.0011), and time_signature (0.0006). The full audio feature set produces NDCG@10 = 0.376. Single-feature ablations of the six top-ranked features yield NDCG@10 values between 0.373 and 0.376, corresponding to drops of 0.0001 to 0.0029.

**Figure 5 — Per-instance audio feature attribution.**
Insert this figure: `outputs/rq3_f1_audio_feature_importance_beeswarm.pdf`
Caption: Beeswarm-style attribution showing the per-sample contribution of each audio feature to recommendation score.

**Table 3 — Permutation importance and feature ablation.**
Insert this table: `doc/results_tables/Table_3_audio_feature_importance_and_ablation.csv`
Caption: Permutation importance for all 13 audio features and NDCG@10 obtained when each of the six top-ranked features is removed from the audio branch.

```
Audio Feature	Importance	NDCG@10 Without Feature	ΔNDCG@10
danceability	0.0169	0.3749	−0.0015
instrumentalness	0.0144	0.3735	−0.0029
speechiness	0.0144	0.3762	−0.0001
acousticness	0.0128	0.3748	−0.0015
valence	0.0111	0.3753	−0.0011
duration_ms	0.0106	0.3763	−0.0001
loudness	0.0100	—	—
liveness	0.0085	—	—
energy	0.0085	—	—
tempo	0.0066	—	—
key	0.0054	—	—
mode	0.0011	—	—
time_signature	0.0006	—	—
```

---

## RQ4: How effectively can genre, mood, artist, and tag-based semantic metadata enhance recommendation relevance and interpretability?

Four semantic configurations were evaluated. Genres Only reaches NDCG@10 = 0.273 (Precision@10 = 0.0381), Free-form Tags Only reaches 0.352 (0.0510), Artist Metadata reaches 0.372 (0.0497), and the Full Semantic Encoder reaches 0.374 (0.0525). All four configurations achieve 100% explanation coverage. Generated explanations for the proposed model contain matched semantic tokens drawn from the user's recent listening history; representative examples are shown in Table 5.

**Figure 6 — Utility of semantic metadata across tag families and granularity levels.**
Insert this figure: `outputs/rq4_f2_semantic_utility_heatmap.pdf`
Caption: Hierarchical heatmap of Precision@10 and NDCG@10 across five tag families (Genre, Artist Style, Free-form Tags, Genre–Mood, Full Semantic) and three granularity levels (Coarse, Medium, Fine).

**Table 4 — Semantic configuration comparison.**
Insert this table: `doc/results_tables/Table_4_semantic_configuration_comparison.csv`
Caption: Precision@10, NDCG@10, and explanation coverage for the four semantic configurations.

```
Semantic Configuration	Precision@10	NDCG@10	Explanation Coverage (%)
Genres Only	0.0381	0.273	100.0
Free-form Tags Only	0.0510	0.352	100.0
Artist Metadata	0.0497	0.372	100.0
Full Semantic Encoder	0.0525	0.374	100.0
```

**Table 5 — Example semantic explanations generated for top-ranked tracks.**
Insert this table: `doc/results_tables/Table_5_semantic_explanation_examples.csv`
Caption: Top-five recommended tracks with matched semantic tokens and corresponding semantic score for a representative user.

```
Rank	Track ID	Matched Semantic Evidence	Semantic Score
1	TRQRSAK128F429DE8C	killers, rock, alternative, indie	0.432
2	TRRTELE128F427EC7E	rock, alternative, indie, alternative_rock, indie_rock	0.360
3	TREVGQV128F92F39D6	rock, alternative, indie, alternative_rock, indie_rock, american	0.301
4	TRDSBPA128F429595F	rock, alternative, indie, alternative_rock, indie_rock	0.372
5	TRTCBDM12903CDEBDC	rock, alternative, indie	0.216
```

---

## RQ5: Which multimodal fusion strategy provides the best balance between recommendation accuracy and interpretability?

Four fusion strategies were evaluated. NDCG@10 values are 0.389 for Weighted Early Fusion, 0.374 for Late Fusion, 0.385 for Embedding-Level Fusion, and 0.376 for Attention-Based Fusion. Corresponding interpretability scores on a 1–5 scale are 4.29, 4.34, 4.27, and 4.23. Average learned modality weights from Attention-Based Fusion vary by user context: High-Activity Users receive weights of 0.51 / 0.24 / 0.25 (behavior / audio / semantic), Low-Activity Users 0.28 / 0.36 / 0.36, New Item Recommendation 0.00 / 0.57 / 0.43, Diverse Taste Users 0.50 / 0.24 / 0.26, and Calm-Mood Listeners 0.44 / 0.28 / 0.29.

**Figure 7 — Learned modality weights across recommendation contexts.**
Insert this figure: `outputs/rq5_f2_modality_weight_heatmap.pdf`
Caption: Heatmap of the average behavior, audio, and semantic weights produced by Attention-Based Fusion across five recommendation contexts.

**Table 6a — Fusion strategy comparison.**
Insert this table: `doc/results_tables/Table_6a_fusion_strategy_comparison.csv`
Caption: Precision@10, NDCG@10, and interpretability score for the four fusion strategies.

```
Fusion Strategy	Precision@10	NDCG@10	Interpretability (1–5)
Weighted Early Fusion	0.0534	0.389	4.29
Late Fusion	0.0550	0.374	4.34
Embedding-Level Fusion	0.0526	0.385	4.27
Attention-Based Fusion	0.0527	0.376	4.23
```

**Table 6b — Learned modality weights from Attention-Based Fusion.**
Insert this table: `doc/results_tables/Table_6b_learned_modality_weights.csv`
Caption: Average learned modality weights across five recommendation contexts.

```
Scenario	Behavior	Audio	Semantic
High-Activity User	0.51	0.24	0.25
Low-Activity User	0.28	0.36	0.36
New Item Recommendation	0.00	0.57	0.43
Diverse Taste User	0.50	0.24	0.26
Calm-Mood Listener	0.44	0.28	0.29
```

---

## RQ6: How can recommendation explanations generated from behavioral, acoustic, and semantic evidence improve transparency and user trust?

Four explanation methods were evaluated on fidelity, coverage, diversity, and a composite quality score on a 1–5 scale. Template-Only achieves fidelity 0.000, coverage 100%, diversity 0.0004, and composite score 2.00. Feature-Based achieves 0.498, 100%, 0.0047, and 3.00. Tag-Based achieves 0.006, 100%, 0.0214, and 2.03. The Proposed Multimodal Explanation achieves 0.337, 100%, 0.0154, and 2.69. The per-recommendation case study decomposes each top-ranked track into behavioral, acoustic, and semantic contributions plus a weighted final score derived from the user's learned modality weights. A formal user study is out of scope for this work; the composite quality score is computed algorithmically from fidelity and diversity components.

**Figure 8 — Score decomposition of top recommendations by modality.**
Insert this figure: `outputs/rq6_f1_score_decomposition.pdf`
Caption: Stacked bar chart of the behavior, audio, and semantic contributions to the final score for the top-ranked recommendations of a representative user.

**Table 7 — Explanation quality across methods.**
Insert this table: `doc/results_tables/Table_7_explanation_quality.csv`
Caption: Fidelity, coverage, diversity, and composite quality score for the four explanation methods.

```
Explanation Method	Fidelity	Coverage (%)	Diversity	Composite Quality Score (1–5)
Template-Only	0.000	100.0	0.0004	2.00
Feature-Based	0.498	100.0	0.0047	3.00
Tag-Based	0.006	100.0	0.0214	2.03
Proposed Multimodal Explanation	0.337	100.0	0.0154	2.69
```

---

## RQ7: How robust is the proposed explainable hybrid recommendation framework under sparse interaction data, cold-start users, cold-start tracks, and incomplete metadata conditions?

NDCG@10 across cold-start scenarios is 0.239 for a New User with 1 interaction, 0.406 for a New User with 5 interactions, 0.421 for a New User with 10 interactions, 0.240 for a New Track, 0.358 for a Sparse Metadata Track, and 0.376 for the full-data reference. Under stress conditions, NDCG@10 is 0.376 at No Stress (0.0% drop), 0.348 with 10% Missing Audio Features (7.7% drop), 0.342 with a 20% Interaction Sparsity Increase (9.1% drop), 0.309 with 30% Missing Semantic Tags (17.8% drop), and 0.303 under Combined Stress (19.5% drop).

**Figure 9 — Cold-start performance under varying signal availability.**
Insert this figure: `outputs/rq7_f1_cold_start_performance.pdf`
Caption: NDCG@10 across six cold-start scenarios spanning new users, new tracks, and sparse metadata conditions.

**Figure 10 — Robustness heatmap across stress conditions.**
Insert this figure: `outputs/rq7_f2_robustness_heatmap.pdf`
Caption: Heatmap of NDCG@10 under increasing levels of interaction sparsity, missing audio features, missing semantic tags, and combined stress.

**Table 8 — Cold-start scenario results.**
Insert this table: `doc/results_tables/Table_8_cold_start_scenarios.csv`
Caption: Available signals and NDCG@10 for each cold-start scenario including the full-data reference.

```
Scenario	Available Signals	NDCG@10
New User (1 interaction)	Audio + Semantic + minimal behavior	0.239
New User (5 interactions)	Behavior + Audio + Semantic	0.406
New User (10 interactions)	Behavior + Audio + Semantic	0.421
New Track	Audio + Semantic	0.240
Sparse Metadata Track	Behavior + Partial Audio	0.358
Full-data reference	All signals	0.376
```

**Table 9 — Stress-test degradation.**
Insert this table: `doc/results_tables/Table_9_stress_test_degradation.csv`
Caption: NDCG@10 and percentage performance drop relative to the full-data reference for each stress condition.

```
Stress Condition	NDCG@10	Performance Drop
No Stress (Full Data)	0.376	0.0%
10% Missing Audio Features	0.348	7.7%
20% Interaction Sparsity Increase	0.342	9.1%
30% Missing Semantic Tags	0.309	17.8%
Combined Stress Condition	0.303	19.5%
```
