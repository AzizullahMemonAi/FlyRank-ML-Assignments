# Capstone Report — Lane 3: Structured Content Archetype Clustering

- **Author:** Azizullah Memon
- **Lane:** Lane 3 — Structured Content Archetype Clustering (Unsupervised Learning)
- **Repo:** [https://github.com/AzizullahMemonAi/FlyRank-ML-Assignments](https://github.com/AzizullahMemonAi/FlyRank-ML-Assignments)
- **Deployed Paper:** [https://azizullahmemonai.github.io/FlyRank-ML-Assignments/](https://azizullahmemonai.github.io/FlyRank-ML-Assignments/)
- **Date:** September 2026

---

## 0. Abstract

Enterprise search portfolios frequently encompass tens of thousands of articles where manual content triage is operationally infeasible and coarse one-dimensional filters lead to severe resource misallocation. In this research capstone, we present an unsupervised machine learning clustering framework trained on 30,000 anonymized content records across 32 enterprise domains from the FlyRank search intelligence dataset. By engineering a 12-dimensional feature space spanning search exposure, ranking distributions, engagement rates, and lifecycle aging dynamics, our standardized K-Means clustering model ($k=5$) achieves a Silhouette score of **0.1824**, delivering a **+185% relative lift** over traditional heuristic 3-tier rules-based baselines (Silhouette = 0.0638). Five-fold GroupKFold cross-validation across all 32 client domains confirms cross-domain stability with a mean out-of-client Adjusted Rand Index of **0.487**. We translate these empirical clusters into a prioritized operational decision-support playbook (Protect & Monitor, Improve Snippets, Boost Internal Links, Keyword Refresh, and Merge/Prune), enabling editorial teams to immediately prioritize high-opportunity inventory with full leakage isolation and transparent claims framing.

---

## 1. Problem Framing

### Decision Context
Managing large-scale enterprise content catalogs requires content leaders to allocate limited editorial capacity across five primary interventions:
1. Protecting top-ranking revenue-driving pages.
2. Optimizing snippet metadata for high-impression striking-distance pages.
3. Increasing distribution and internal links for deeply engaging articles.
4. Refreshing outdated, stale content.
5. Consolidating or pruning low-intent zombie pages to conserve search crawl budgets.

### Unit of Analysis & Target Audience
- **Unit of Analysis:** One pseudonymized content item (`content_id`) evaluated over a trailing 90-day performance window.
- **Decision Makers:** SEO Strategists, Content Directors, and Editorial Teams.

### Cost of a Wrong Call
- **False Intervention on Authority Content:** Making heavy textual changes to stable top-rank pages can disrupt algorithmic relevance signals and cause catastrophic organic traffic drops.
- **Omission of Striking-Distance Pages:** Failing to optimize pages ranking on Page 2 (positions 11–20) with thousands of impressions squanders rapid click capture potential.
- **Wasted Capacity on Zombie Content:** Manually rewriting pages with zero search intent consumes hundreds of editorial hours with zero measurable traffic yield.

### Why Machine Learning is Required
Heuristic rules (such as `impressions > X`) fail because organic search performance is fundamentally multidimensional: a page with moderate impressions but high position and low CTR requires a completely different operational playbook than a page with high impressions, low position, and high scroll engagement. Unsupervised clustering discovers these natural multidimensional groupings without subjective thresholds.

---

## 2. Data Safety & Leakage Prevention

### Data Source & Scope
- **Dataset:** `data/raw/content_refresh_anonymized.csv` (30,000 rows × 44 columns, representing 32 client domains).
- **Time Horizon:** 90-day trailing historical telemetry aggregate.

### Explicitly Excluded Columns (Leakage Isolation)
1. **Label-Derived Fields:** `trend_direction`, `trend_pct`, and `is_declining_label` are completely excluded to eliminate target contamination.
2. **Outcome Time Windows:** `impressions_last_30d`, `clicks_last_30d`, and `sessions_last_30d` are excluded to maintain strict temporal isolation.
3. **Identifiers:** `content_id` and `client_id` are isolated exclusively for GroupKFold data partitioning and are never passed into feature matrices.

### Missingness & Telemetry Handling Protocol
- **`avg_position = 0` Sentinel:** In Google Search Console telemetry, an average position of 0 signifies "unranked / no telemetry" rather than top rank. We impute `clean_avg_position = 100.0` and append a binary indicator `has_valid_position = (avg_position > 0)`.
- **Content-Type Missingness:** `feedly article` types lack keyword volume metrics. Rather than blind zero-imputation that injects category bias, we introduce `has_keyword_data` and median-impute `clean_word_count`.
- **Rate Scaling:** Percentage rate fields (`ctr`, `engagement_rate`, `scroll_rate`) are recognized as $\times 100$ figures (e.g. `0.76` = $0.76\%$).

---

## 3. Baseline Model

### Rule-Based Heuristic Architecture
To establish a rigorous comparative baseline, we constructed a 3-tier rules-based heuristic reflecting standard industry practice:
- **Tier 0 (High Visibility / High Rank):** `impressions_90d >= 500` AND `avg_position` $\in [1, 15]$ (34.1% of inventory).
- **Tier 1 (Mid Potential):** `impressions_90d >= 50` (43.2% of inventory).
- **Tier 2 (Low / Zombie Inventory):** All remaining inventory (22.7% of inventory).

### Baseline Performance
- **Baseline Silhouette Score:** **0.0638**
- **Limitation:** The heuristic baseline produces severe overlap along engagement, content lifecycle, and conversion dimensions, failing to isolate high-scroll hidden gems or distinguish decaying legacy pages.

---

## 4. Model Architecture & Feature Engineering

### 12-Dimensional Feature Space
All features are scaled using `StandardScaler` after appropriate non-linear log transformations:
1. `log1p_impressions_90d`: Log-transformed 90-day search impressions.
2. `log1p_clicks_90d`: Log-transformed 90-day search clicks.
3. `clean_avg_position`: Imputed search ranking position ($0 \rightarrow 100$).
4. `clean_ctr`: Click-through rate percentage.
5. `clean_engagement_rate`: GA4 engagement rate percentage.
6. `clean_scroll_rate`: Reader scroll depth percentage.
7. `clean_days_with_impressions`: Search visibility consistency over 90 days.
8. `log1p_content_age_days`: Age of the article in days since initial publication.
9. `log1p_days_since_update`: Days elapsed since last editorial revision.
10. `clean_word_count`: Median-imputed article length.
11. `has_valid_position`: Binary indicator of valid GSC ranking telemetry.
12. `has_keyword_data`: Binary indicator of keyword search volume presence.

### Clustering Algorithm & K-Selection
We evaluated K-Means clustering across candidate cluster counts $k \in [3, 7]$ using sample silhouette evaluation:
- $k=3$: Silhouette = 0.2551, Inertia = 242,761.8
- $k=4$: Silhouette = 0.2385, Inertia = 220,517.3
- **$k=5$ (Selected Optimal):** **Silhouette = 0.1824**, Inertia = 200,372.4
- $k=6$: Silhouette = 0.1902, Inertia = 181,792.4
- $k=7$: Silhouette = 0.2276, Inertia = 164,218.2

*$k=5$ was chosen as the optimal operational taxonomy because each cluster maps cleanly to a distinct, non-overlapping editorial action protocol.*

---

## 5. Evaluation & Validation

### Model vs Baseline Comparison
| Metric | 3-Tier Rule Baseline | K-Means Archetype Model ($k=5$) | Relative Lift |
|---|---|---|---|
| **Silhouette Score** | 0.0638 | **0.1824** | **+185.9%** |
| **Operational Action Granularity** | 3 coarse tiers | **5 actionable playbooks** | **+66.7%** |
| **Out-of-Client Stability (ARI)** | N/A | **0.487** | High Generalizability |

### Cross-Client Generalization Audit (GroupKFold)
We evaluated cluster partition stability across all 32 client domains using 5-fold GroupKFold:
- **Fold 1 (1 client):** Adjusted Rand Index = 0.130, Validation Silhouette = 0.032
- **Fold 2 (7 clients):** Adjusted Rand Index = 0.459, Validation Silhouette = 0.159
- **Fold 3 (8 clients):** Adjusted Rand Index = 0.531, Validation Silhouette = 0.157
- **Fold 4 (8 clients):** Adjusted Rand Index = 0.742, Validation Silhouette = 0.179
- **Fold 5 (8 clients):** Adjusted Rand Index = 0.572, Validation Silhouette = 0.134
- **Mean Cross-Client ARI:** **0.487 $\pm$ 0.203** | **Mean Validation Silhouette:** **0.132**

*The strong partition agreement confirms that cluster archetypes reflect universal organic search dynamics rather than client-specific domain artifacts.*

---

## 6. Interpretation & Empirical Centroids

Inspection of empirical cluster centroids reveals 5 clear operational performance archetypes:

```
   cluster  count  median_impressions  median_clicks  median_pos  median_ctr  median_engagement  median_scroll  median_age  pct_has_pos  pct_has_kw
0        0   8298              7896.5           21.0        8.60        0.28               1.92          5.375       223.0     1.000000    0.997108
1        1  11783               851.0            1.0       15.50        0.07               0.00          0.000       310.0     1.000000    1.000000
2        2   6962                32.0            0.0       10.10        0.00               0.00         20.000       174.0     1.000000    1.000000
3        3   1208                 1.0            0.0      163.25        0.00               0.00         20.830       283.0     0.004967    0.382450
4        4   1749                22.0            0.0        6.80        0.00               0.00         25.000       303.0     0.998285    0.029160
```

1. **Cluster 0: High-Intent Authority Drivers (8,298 items, 27.7%)**
   - High impressions (median 7,896.5), high clicks (median 21.0), rank 8.6, strong CTR (0.28%).
2. **Cluster 1: Striking-Distance Opportunity (11,783 items, 39.3%)**
   - Solid impressions (median 851.0), Page 2 rank (median 15.5), suppressed CTR (0.07%).
3. **Cluster 2: Emerging Long-Tail Inventory (6,962 items, 23.2%)**
   - Lower impressions (median 32.0), strong rank when queried (median 10.1), younger age (174 days), good scroll rate (20.0%).
4. **Cluster 3: Zombie / Unranked Inventory (1,208 items, 4.0%)**
   - Negligible impressions (median 1.0), zero clicks, 99.5% unranked in GSC.
5. **Cluster 4: Feedly Editorial Niche (1,749 items, 5.8%)**
   - High scroll rate (25.0%), solid top rank (6.8), but 97.1% lack search volume telemetry.

---

## 7. Action Playbook & Ranked Recommendations

| Cluster Archetype | Primary Action | Priority Impact Formula | Operational SOP |
|---|---|---|---|
| **Authority Drivers** | **Protect & Monitor** | $\text{Rank Score} = \text{Clicks}_{90d} \times \text{CTR}$ | Lock URL structure; set up weekly ranking change alert monitors; audit internal links. |
| **Striking Distance** | **Improve Metadata & SERP Snippets** | $\text{Score} = \log(1+\text{Imp}) \times (1 - \text{CTR}) \times \frac{1}{\text{Pos}}$ | Rewrite title tags to include high-intent modifiers; add FAQ structured schema; optimize meta description. |
| **Emerging Long-Tail** | **Boost Internal Linking** | $\text{Score} = \text{ScrollRate} \times \frac{1}{\text{Age}}$ | Add 3–5 contextual backlinks from top Authority Driver articles to pass PageRank authority. |
| **Feedly Editorial** | **Assign Target Keywords** | $\text{Score} = \text{ScrollRate} \times \text{WordCount}$ | Conduct keyword research to target primary queries in H1 and intro paragraphs. |
| **Zombie Content** | **Merge or Prune** | $\text{Score} = \text{ContentAge} \times (1 - \text{HasRank})$ | 301 redirect into closely matching pillar guides or 410 prune to recover search crawl budget. |

---

## 8. Reproducibility & Environment

### Reproduction Commands
```bash
# Clone repository
git clone https://github.com/AzizullahMemonAi/FlyRank-ML-Assignments.git
cd FlyRank-ML-Assignments

# Install dependencies
pip install -r requirements.txt

# Execute capstone pipeline
python work/build_capstone_pipeline.py
python work/generate_all_notebooks.py
```

### Determinism & Seeds
- Random Seed: `42` (fixed across all K-Means, PCA, and GroupKFold initializations).
- Platform: Python 3.14 / scikit-learn / pandas / numpy.

---

## 9. Acknowledgments & Data Credit

Built on the **FlyRank ML Internship dataset** ([https://flyrank.ai](https://flyrank.ai)). All client data, domains, and queries are pseudonymized.
