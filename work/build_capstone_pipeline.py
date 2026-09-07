"""
Capstone Pipeline and Notebook Builder for Lane 3: Structured Content Archetype Clustering
FlyRank ML Internship
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "content_refresh_anonymized.csv"
OUTPUT_DIR = ROOT / "work" / "outputs"
CHART_DIR = ROOT / "outputs" / "charts"
NOTEBOOKS_DIR = ROOT / "work" / "notebooks"
DOCS_DIR = ROOT / "docs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

print("1. Loading raw dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows and {len(df.columns)} columns across {df['client_id'].nunique()} clients.")

# Feature Engineering for Unsupervised Clustering
# Explicitly omitting label sources: trend_direction, trend_pct, is_declining_label
# Explicitly omitting pseudonyms as features: client_id, content_id

print("2. Engineering clustering features and handling edge cases...")
clean_df = df.copy()

# Fix position 0 (no data)
clean_df["has_valid_position"] = (clean_df["avg_position"] > 0).astype(int)
pos_median = clean_df.loc[clean_df["avg_position"] > 0, "avg_position"].median()
clean_df["clean_avg_position"] = clean_df["avg_position"].replace(0, 100.0) # Rank 100 as unranked sentinel

# Missing indicators
clean_df["has_keyword_data"] = clean_df["search_volume"].notna().astype(int)
clean_df["has_word_count"] = clean_df["word_count"].notna().astype(int)
clean_df["clean_search_volume"] = clean_df["search_volume"].fillna(0)
clean_df["clean_word_count"] = clean_df["word_count"].fillna(clean_df["word_count"].median())

# Log-transform skewed counts (impressions, clicks, sessions)
clean_df["log1p_impressions_90d"] = np.log1p(clean_df["impressions_90d"])
clean_df["log1p_clicks_90d"] = np.log1p(clean_df["clicks_90d"])
clean_df["log1p_sessions_90d"] = np.log1p(clean_df["sessions_90d"])
clean_df["log1p_search_volume"] = np.log1p(clean_df["clean_search_volume"])
clean_df["log1p_content_age_days"] = np.log1p(clean_df["content_age_days"])
clean_df["log1p_days_since_update"] = np.log1p(clean_df["days_since_last_update"])

# Key rate features
clean_df["clean_ctr"] = clean_df["ctr"].fillna(0)
clean_df["clean_engagement_rate"] = clean_df["engagement_rate"].fillna(0)
clean_df["clean_scroll_rate"] = clean_df["scroll_rate"].fillna(0)
clean_df["clean_ai_traffic_pct"] = clean_df["ai_traffic_pct"].fillna(0)
clean_df["clean_days_with_impressions"] = clean_df["days_with_impressions"].fillna(0)

# Define feature subset for clustering
CLUSTER_FEATURES = [
    "log1p_impressions_90d",
    "log1p_clicks_90d",
    "clean_avg_position",
    "clean_ctr",
    "clean_engagement_rate",
    "clean_scroll_rate",
    "clean_days_with_impressions",
    "log1p_content_age_days",
    "log1p_days_since_update",
    "clean_word_count",
    "has_valid_position",
    "has_keyword_data"
]

X = clean_df[CLUSTER_FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Clustering matrix: {X_scaled.shape}")

# 3. Rule Baseline Comparison
print("3. Evaluating heuristic baseline...")
def assign_baseline_cluster(row):
    if row["impressions_90d"] >= 500 and row["avg_position"] > 0 and row["avg_position"] <= 15:
        return 0 # "High Visibility / High Rank"
    elif row["impressions_90d"] >= 50:
        return 1 # "Mid Tier"
    else:
        return 2 # "Low / Zombie"

clean_df["baseline_cluster"] = clean_df.apply(assign_baseline_cluster, axis=1)
np.random.seed(42)
sample_idx = np.random.choice(len(X_scaled), size=5000, replace=False)
baseline_sil = silhouette_score(X_scaled[sample_idx], clean_df["baseline_cluster"].iloc[sample_idx])
print(f"Rule Baseline Silhouette Score (k=3): {baseline_sil:.4f}")

# 4. K-Means Grid Search for k in [3, 4, 5, 6, 7]
print("4. Evaluating K-Means models...")
k_results = {}
for k in [3, 4, 5, 6, 7]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled[sample_idx], labels[sample_idx])
    k_results[k] = sil
    print(f"K={k}: Silhouette = {sil:.4f}, Inertia = {km.inertia_:.1f}")

OPTIMAL_K = 5
final_km = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
clean_df["cluster"] = final_km.fit_predict(X_scaled)
final_sil = silhouette_score(X_scaled[sample_idx], clean_df["cluster"].iloc[sample_idx])
print(f"Optimal K={OPTIMAL_K} Model Silhouette: {final_sil:.4f} (vs Baseline: {baseline_sil:.4f})")

# Archetype Naming and Action Mapping
cluster_profiles = clean_df.groupby("cluster").agg(
    count=("content_id", "count"),
    median_impressions=("impressions_90d", "median"),
    median_clicks=("clicks_90d", "median"),
    median_pos=("avg_position", lambda x: x[x > 0].median() if len(x[x > 0]) > 0 else 0),
    median_ctr=("ctr", "median"),
    median_engagement=("engagement_rate", "median"),
    median_scroll=("scroll_rate", "median"),
    median_age=("content_age_days", "median"),
    median_update_days=("days_since_last_update", "median"),
    pct_has_pos=("has_valid_position", "mean"),
    pct_has_kw=("has_keyword_data", "mean")
).reset_index()

print("\nCluster Centroid Profiles:")
print(cluster_profiles.to_string())

# Interpretation & archetype definitions based on empirical centroids
archetype_map = {}
action_map = {}
rationale_map = {}

for _, row in cluster_profiles.iterrows():
    c = int(row["cluster"])
    imp = row["median_impressions"]
    pos = row["median_pos"]
    ctr = row["median_ctr"]
    eng = row["median_engagement"]
    age = row["median_age"]
    upd = row["median_update_days"]
    has_pos = row["pct_has_pos"]
    
    if imp > 200 and pos > 0 and pos <= 12 and ctr > 0.4:
        archetype_map[c] = "High-Intent Authority Drivers"
        action_map[c] = "Protect & Monitor"
        rationale_map[c] = "Top ranking and strong click conversion; protect against link rot and monitor competitive keyword movements."
    elif imp > 150 and (pos > 12 or ctr <= 0.4):
        archetype_map[c] = "Striking-Distance Under-Capturers"
        action_map[c] = "Improve Metadata & SERP Snippets"
        rationale_map[c] = "High search exposure but under-capturing CTR due to sub-optimal title tags or ranking on page 2."
    elif eng > 60 and row["median_scroll"] > 60 and imp < 100:
        archetype_map[c] = "Deep-Engagement Hidden Gems"
        action_map[c] = "Internal Linking & Distribution Boost"
        rationale_map[c] = "Exceptional reader engagement and scroll depth but constrained search visibility; needs internal linking hubs."
    elif upd > 250 or age > 400:
        archetype_map[c] = "Stale / Decaying Legacy Pages"
        action_map[c] = "Rewrite / Content Refresh"
        rationale_map[c] = "Older inventory with lagging update cadences losing search velocity; prioritized for content modernization."
    else:
        archetype_map[c] = "Low-Discovery Zombie Inventory"
        action_map[c] = "Merge or Prune"
        rationale_map[c] = "Negligible search visibility, near-zero impressions, and weak reader engagement; candidate for consolidation or 301 redirect."

clean_df["archetype"] = clean_df["cluster"].map(archetype_map)
clean_df["action"] = clean_df["cluster"].map(action_map)
clean_df["action_rationale"] = clean_df["cluster"].map(rationale_map)

# 5. Stability Audit (GroupKFold across 32 clients)
print("\n5. Running Stability Audit across 32 clients (GroupKFold)...")
gkf = GroupKFold(n_splits=5)
ari_scores = []
fold_sil_scores = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, groups=clean_df["client_id"])):
    val_clients = clean_df["client_id"].iloc[val_idx].unique()
    km_fold = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=5)
    km_fold.fit(X_scaled[train_idx])
    val_pred = km_fold.predict(X_scaled[val_idx])
    full_pred_val = clean_df["cluster"].iloc[val_idx].values
    
    ari = adjusted_rand_score(full_pred_val, val_pred)
    ari_scores.append(ari)
    val_sil = silhouette_score(X_scaled[val_idx][:1000], val_pred[:1000])
    fold_sil_scores.append(val_sil)
    print(f"  Fold {fold+1} ({len(val_clients)} clients): ARI = {ari:.3f}, Silhouette = {val_sil:.3f}")

mean_ari = np.mean(ari_scores)
mean_val_sil = np.mean(fold_sil_scores)
print(f"Mean Out-of-Client Adjusted Rand Index: {mean_ari:.3f} | Mean Val Silhouette: {mean_val_sil:.3f}")

# 6. PCA 2D Projection
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
clean_df["pca_1"] = X_pca[:, 0]
clean_df["pca_2"] = X_pca[:, 1]
explained_variance = pca.explained_variance_ratio_

# Save processed summaries
summary_stats = {
    "total_records": int(len(clean_df)),
    "num_clients": int(clean_df["client_id"].nunique()),
    "optimal_k": OPTIMAL_K,
    "baseline_silhouette": float(baseline_sil),
    "model_silhouette": float(final_sil),
    "silhouette_lift": float(final_sil - baseline_sil),
    "mean_client_ari": float(mean_ari),
    "pca_variance_explained": [float(explained_variance[0]), float(explained_variance[1])],
    "cluster_distribution": clean_df["archetype"].value_counts().to_dict(),
    "action_distribution": clean_df["action"].value_counts().to_dict()
}

with open(OUTPUT_DIR / "capstone_summary.json", "w") as f:
    json.dump(summary_stats, f, indent=2)

print("\nSummary stats saved successfully to work/outputs/capstone_summary.json")
