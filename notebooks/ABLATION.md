# HDFS / BGL ablation campaign

End-to-end protocol: **preprocess HDFS locally**, **train/eval on Colab**. BGL graphs may be prepared locally or in [`7_AblationStudy_BGL_Colab.ipynb`](./7_AblationStudy_BGL_Colab.ipynb) (`RUN_PREPARE=True`). Keep a comparable artifact pack for every configuration.

This file is the operational contract. Implementation lives in this repository (`run_ablation.py`, `configs/ablation_*.yaml`, `src/modules/ablation.py`).

## Families (do not mix in one `train-only` graph)

| Family | YAML | Rebuild `graph_dataset.pt`? | Colab mode |
|---|---|---|---|
| **A — representation (HDFS)** | `configs/ablation_representation.yaml` | Yes | `--mode train-only --family A --campaign-dir …` |
| **A — LLM closed (BGL)** | `configs/ablation_bgl_llm_closed.yaml` | Yes | same campaign dir; 5 arms × 5 seeds |
| **A — PB3 feature groups (BGL)** | `configs/ablation_bgl_pb3.yaml` | Yes | same campaign dir after LLM closed |
| **A — representation (BGL, broader)** | `configs/ablation_representation_bgl.yaml` | Yes | separate campaign; `baseline_full` |
| **A — fusion / grounded text (BGL)** | `configs/ablation_fusion_bgl.yaml` | Yes | same, new campaign id, reuse the inductive `split_lock.npz` |
| **B — train / architecture** | `configs/ablation_train.yaml` | No | `--mode train-only --family B` on `baseline_full` |

`configs/ablation_matrix.yaml` is deprecated. It now re-exports Family B only so an accidental shared-graph `train-only` run cannot fake a TF-IDF vs SBERT ablation.

Train-only **fails** if the config’s graph identity (LLM on/off, TF-IDF/SBERT, edge features, `feature_contract`, `unique_sequences`, `sbert_text`) disagrees with `dataset_meta.json` on the bundle. `ablation.fusion.node_reconstruct` is a train-time decoder mask and does **not** require a new graph.

## Invariants

- One Drain parse, one HDFS block sequencing, **one split lock** (`split_lock.npz`) shared by every Family A graph.
- Family A does **not** rebuild collapsed topology seven times. Stage `stage45_graph_structure` caches per-block extras and 10-d edges keyed by dataset + `unique_sequences` + `feature_contract` + `fit_on` + `split_protocol`. Embedding arms (`tfidf_only`, `sbert_only`, `no_llm_enrichment`, …) splice a new node matrix; `no_edge_features` keeps the first edge column. Only `feature_contract_stabilized_v2` needs a second structure pass.
- HDFS Family A graphs use **one PyG graph per block** (`ablation.graph.unique_sequences: false`): blocks with the same ordered `cluster_id` list can still differ in parameter statistics, edge timing, and labels, so they remain distinct training and evaluation examples. A topology-only representative dataset (`unique_sequences: true`) is an optional efficiency experiment, not the default benchmark; it changes the class distribution and must use its own `split_lock.npz`.
- Default legacy matrices use seed 42; the focused HDFS experiment uses paired
  seeds `[13, 29, 42, 71, 101]`. Use clean-train, a val-F1 threshold, and report
  test **F1 / PR-AUC / ROC-AUC** (rank by PR-AUC).
- LLM template enrichment is **Deepseek v4 Pro only** (`AZURE_OPENAI_DEPLOYMENT_DEEPSEEK_V4_PRO`). Prompt version `distinctive_v3` is part of the stage-2 cache key; changing it re-enriches. MiniLM (`sbert_text: grounded_v1`) encodes enrichment fields only — not Drain templates or raw examples.
- Default `ablation.feature_contract: notebook_raw_v1` (thesis-comparable). `stabilized_v2` is an explicit Family A arm.
- HDFS default is **`inductive_v1`**: `ablation.representation.fit_on: train_only`, per-block graphs, directed `structure_decoder: mlp`, fail-closed labels, `missing_embedding=fail`. BGL published campaigns use `sequencing.bgl.split: time` plus a 20-minute embargo, but `ablation.representation.fit_on_by_dataset.bgl: all` — Drain and TF-IDF see the **full unlabeled corpus**. Notebook / `hdfs_baseline.yaml` numbers are **`transductive_v1`** (Drain+TF-IDF fit on the full log, random window split, inner-product structure). Do **not** mix `split_lock.npz` / `graph_dataset.pt.gz` across protocols or compare PR-AUC between them.
- New campaigns need a new id. Keep `hdfs_ablation_20260916` / notebook ROC-AUC 0.976 as the transductive ceiling. Suggested inductive ids: `hdfs_inductive_v1_20260918`, `bgl_inductive_v1_20260918`. Fusion / re-enrich graphs: `bgl_inductive_fusion_v1_20260918` (reuse that campaign's `split_lock.npz`).
- Structure component AUC in PROCESS.md (~0.52) used batch-level negatives, a symmetric decoder, and positive-only eval scores. After the GAE restore, report structure/node/edge AUCs **before vs after** that change; do not treat a move away from 0.52 as a regression against the notebook.
- Do not mix sentence-transformers versions inside one dataset campaign. HDFS stays local (ST 2.2.2). A BGL campaign is either fully local (ST 2.2.2) or fully Colab-prepared (ST 5.x via `RUN_PREPARE`). Train-only consumes frozen tensors.
- Never train from the Drive mount. Stage `.pt.gz` → `/content/workspace`, decompress there.
- Pin `GIT_REF` for the Colab clone; do not `git pull` mid-campaign. Cache fingerprints no longer include git SHA (SHA still lands on run manifests).

## Local prepare (Family A graphs)

### HDFS hybrid-representation experiment (five paired seeds)

The focused experiment uses:

- `configs/ablation_hdfs_representation.yaml` for graph-changing arms:
  `baseline_full`, `tfidf_only`, `sbert_only`, `no_edge_features`;
- `configs/ablation_hdfs_fusion_train.yaml` on the **same campaign's**
  `baseline_full` graph for `hybrid_projected_gated`,
  `hybrid_lexical_recon`, `hybrid_block_balanced`, and `alpha_0`;
- paired seeds `[13, 29, 42, 71, 101]` declared by both matrices.

Prepare two separate all-block campaigns. Each campaign has one split lock shared
by all representation bundles; graph tensors are not deduplicated.

```bash
# IID all-block protocol
python scripts/prepare_hdfs_campaign.py \
  --campaign-id hdfs_hybrid_stratified_v2 \
  --workspace-root /path/to/workspace \
  --matrix configs/ablation_hdfs_representation.yaml \
  --set sequencing.hdfs.split=stratified

# Generalisation to unseen ordered-template topologies
python scripts/prepare_hdfs_campaign.py \
  --campaign-id hdfs_hybrid_topology_grouped_v2 \
  --workspace-root /path/to/workspace \
  --matrix configs/ablation_hdfs_representation.yaml \
  --set sequencing.hdfs.split=topology_grouped
```

For `topology_grouped`, a provisional Drain pass defines only the grouping;
the final Drain parser is recreated and fitted exclusively on grouped-train
blocks. Every provisional topology fingerprint is assigned to exactly one
partition. The final graph set still contains every block.

On Colab, run Family A and then Family B for each campaign. Family B must be
given the matching campaign directory so the runner selects that campaign's
`baseline_full`; do not pass an older standalone graph.

```bash
python run_ablation.py --mode train-only --family A \
  --campaign-dir /content/workspace/campaigns/hdfs_hybrid_stratified_v2 \
  --campaign-id hdfs_hybrid_stratified_v2 \
  --matrix configs/ablation_hdfs_representation.yaml \
  --workspace-root /content/workspace

python run_ablation.py --mode train-only --family B \
  --campaign-dir /content/workspace/campaigns/hdfs_hybrid_stratified_v2 \
  --campaign-id hdfs_hybrid_stratified_v2 \
  --matrix configs/ablation_hdfs_fusion_train.yaml \
  --workspace-root /content/workspace
```

Repeat both training commands for the topology-grouped campaign. The report
writes raw per-seed rows to `leaderboard.csv`, arm means/standard deviations to
`summary_by_arm.csv`, and paired seed-bootstrap 95% intervals for deltas versus
`baseline_full` to `paired_bootstrap_ci.csv`. Per-run score files contain
`tfidf`, `sbert`, `extras`, `edge`, and `structure` errors. Dataset metadata and
run manifests include OOV line/graph rates.

From this repository root, Intel Mac PyTorch/PyG plus `requirements.txt`, Azure env for enrichment:

```bash
python scripts/prepare_hdfs_campaign.py \
  --campaign-id hdfs_inductive_v1_20260918 \
  --workspace-root /path/to/workspace \
  --set experiment.dataset=hdfs
```

Equivalent:

```bash
python run_ablation.py --mode prepare-campaign \
  --campaign-id hdfs_inductive_v1_20260918 \
  --workspace-root /path/to/workspace \
  --set experiment.dataset=hdfs
```

Raw log and labels: `data/raw/hdfs/HDFS_full.log` and `data/raw/hdfs/anomaly_label.csv` (via the `data/` symlink into the sibling project repo).

Writes `campaigns/<id>/`:

```
manifest.json
split_lock.npz
graphs/baseline_full/graph_dataset.pt.gz
graphs/baseline_full/dataset_meta.json
graphs/tfidf_only/...
```

Upload that folder to Drive:

`MyDrive/hybrid-log-analyzer-artifacts/campaigns/<id>/`

```bash
python scripts/drive_sync.py push \
  --remote "gdrive:hybrid-log-analyzer-artifacts" \
  --local-root /path/to/workspace \
  --dataset hdfs
```

## Colab train / eval

1. Open [`7_AblationStudy_Colab.ipynb`](./7_AblationStudy_Colab.ipynb) for HDFS, or [`7_AblationStudy_BGL_Colab.ipynb`](./7_AblationStudy_BGL_Colab.ipynb) for BGL (clones this repository).
2. Set `CAMPAIGN_ID`, `FAMILY`, `GIT_REF` (campaign branch), `SMOKE=True` first.
3. Family B can start as soon as `graphs/baseline_full/graph_dataset.pt.gz` is on Drive.
4. Family A stages the whole campaign dir; the CLI unpacks **one** graph at a time.
5. Replay leaderboard from `outputs/hdfs/campaigns/<id>/` without retraining.

Suggested order: smoke → Family B → Family A → freeze leaderboard.

## Per-run artifact pack

`outputs/{dataset}/{campaign_id}_{arm}_seed{seed}/`

- `metrics.json`, `component_metrics.json`, `history.json`, `config.yaml`, `manifest.json`
- `attribute_gae.pt`
- `figures/training_history.png`, `test_pr_roc.png`, `test_score_distribution.png`, `confusion_matrix.png`, `component_contribution.png`
- `scores/test_component_scores.csv`

Campaign rollup: `outputs/{dataset}/campaigns/{campaign_id}/` (`leaderboard.csv`, `delta_vs_baseline.csv`, comparison PNGs, `README.md`).

## Drive layout

```
hybrid-log-analyzer-artifacts/
  campaigns/<id>/manifest.json
  campaigns/<id>/graphs/<arm>/graph_dataset.pt.gz
  outputs/hdfs/<campaign_id>_<arm>/
  outputs/hdfs/campaigns/<id>/
  outputs/bgl/<campaign_id>_<arm>/
  outputs/bgl/campaigns/<id>/
  artifacts/cache/hdfs/
  artifacts/cache/bgl/
  artifacts/runs/
```

## BGL

Graphs are **time-cut then windowed** (`sequencing.bgl.split: time`): train/val/test event slices are chosen first, then 20-minute windows are built **inside** each slice (`step_minutes` may equal `window_minutes` or overlap; overlap never crosses a cut). A window is anomalous if any event has `is_anomaly`. Do not reuse a random-window `split_lock.npz`. LLM enrichment is **Deepseek v4 Pro only**.

Published campaign metrics require `SMOKE=False` (25 epochs and complete splits). A smoke run uses one epoch and at most 5,000 graphs per split and is only a pipeline diagnostic. Use a new campaign ID when replacing an already published smoke campaign. Transductive notebook BGL numbers (overlapping windows + random split) are not comparable. Do **not** expect HDFS notebook ROC-AUC 0.976 / PR-AUC 0.936 / F1 0.915.

Raw log: `data/raw/bgl/BGL_full.log`. You can prepare **locally** (`prepare_bgl_campaign.py`) or **on Colab** via [`7_AblationStudy_BGL_Colab.ipynb`](./7_AblationStudy_BGL_Colab.ipynb) (`RUN_PREPARE=True`). Colab prepare uses sentence-transformers 5.x — do not mix those graphs with a local BGL campaign (ST 2.2.2).

Colab prepare needs:

- Drive: `MyDrive/hybrid-log-analyzer-artifacts/data/raw/bgl/BGL_full.log`
- Secrets: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_DEEPSEEK_V4_PRO`

### Published LLM + PB3 campaign (full monty)

Prepare is **not** training. `--matrix configs/ablation_bgl_pb3.yaml` alone builds only the three feature-group arms. The PB3 baseline and Isolation Forest bundle is `hybrid_llm` from `configs/ablation_bgl_llm_closed.yaml`. Prepare **LLM closed first, then PB3**, same `--campaign-id`, so `manifest.json` merges to eight graphs.

```bash
python3 scripts/prepare_bgl_campaign.py \
  --campaign-id bgl-full-monty-v1 \
  --workspace-root . \
  --matrix configs/ablation_bgl_llm_closed.yaml

python3 scripts/prepare_bgl_campaign.py \
  --campaign-id bgl-full-monty-v1 \
  --workspace-root . \
  --matrix configs/ablation_bgl_pb3.yaml
```

Writes `campaigns/<id>/manifest.json`, `split_lock.npz`, and `graphs/<arm>/graph_dataset.pt.gz`. Parse → enrich → sequence run once; the second prepare reuses that cache and the split lock.

Then train Family A (40 GAE runs if both matrices are prepared; 15 if PB3-only), Isolation Forest, and the report:

```bash
python3 run_ablation.py --mode train-only --family A \
  --campaign-dir campaigns/bgl-full-monty-v1 \
  --campaign-id bgl-full-monty-v1 \
  --workspace-root . \
  --config configs/ablation_base.yaml \
  --set experiment.dataset=bgl

python3 run_ablation.py --mode isolation-forest \
  --campaign-dir campaigns/bgl-full-monty-v1 \
  --campaign-id bgl-full-monty-v1 \
  --workspace-root . \
  --set experiment.dataset=bgl

python3 run_ablation.py --mode report \
  --campaign-id bgl-full-monty-v1 \
  --workspace-root . \
  --set experiment.dataset=bgl
```

On Colab: same campaign id, `RUN_PREPARE=False`, `RUN_TRAIN=True`, `FAMILY="A"`, then `RUN_BASELINE=True`. Never train from the Drive mount. Rank by test PR-AUC under `outputs/bgl/campaigns/<id>/` (`leaderboard.csv`, `summary_by_arm.csv`, `paired_bootstrap_ci.csv`, `paired_time_block_bootstrap.csv`). Deltas are vs **`hybrid_llm`**.

| Arm | What it tests |
|---|---|
| `tfidf_only` / `sbert_raw` / `sbert_llm` | Single embedding modality; raw Drain text vs LLM paragraph |
| `hybrid_raw` vs `hybrid_llm` | **Primary LLM contrast** — same TF-IDF+SBERT GAE; only SBERT source text changes |
| `no_positional_features` | Drop node/edge position stats |
| `no_temporal_features` | Drop edge time-delta stats |
| `no_temporal_or_positional_features` | **Primary PB3 contrast** vs `hybrid_llm` — embeddings + topology + transition count |
| `isolation_forest` | Bag-of-templates counts + OOV share + `log1p` window length on the same frozen windows |

Treat a PR-AUC delta as real when the paired seed-bootstrap 95% CI excludes 0 (and, for BGL time dependence, when the day / 7-day block bootstrap agrees). Architecture is unchanged across these arms: clean-train AttributeAwareGAE (GINE, concat fusion, α=β=γ=1.0). Family B (`ablation_train.yaml`) is out of scope for this campaign.

### How to read F1, PR-AUC, ROC-AUC

Positive class = anomalous window. Checkpoint = epoch with best **validation F1**; that threshold is frozen for test.

| Metric | Threshold? | How to read it |
|---|---|---|
| **PR-AUC** | No | Ranking under imbalance. Chance ≈ **positive-class rate**. Primary ranking metric. |
| **F1** | Yes (`scores >` val-tuned threshold) | Precision/recall at the chosen alert point. Noisier across seeds than PR-AUC. |
| **ROC-AUC** | No | P(anomalous window scored higher than a normal one). Chance = **0.5**. Secondary; can look strong while precision is poor. |

Read test PR-AUC mean ± std first, then paired Δ vs `hybrid_llm`, then F1 at the deployed threshold, then ROC-AUC as a sanity check. Do not compare campaign F1 to `src/modules/evaluate.py` (fixed threshold 0.5).

### Broader representation matrix

Default `prepare_bgl_campaign.py` (no `--matrix`) still uses `configs/ablation_representation_bgl.yaml` (`baseline_full`, no `feature_contract_stabilized_v2`):

```bash
python scripts/prepare_bgl_campaign.py \
  --campaign-id bgl_inductive_v1_20260918 \
  --workspace-root /path/to/workspace
```

Equivalent:

```bash
python run_ablation.py --mode prepare-campaign \
  --campaign-id bgl_inductive_v1_20260918 \
  --workspace-root /path/to/workspace \
  --set experiment.dataset=bgl \
  --matrix configs/ablation_representation_bgl.yaml
```

Fusion follow-up (re-enrich + grounded MiniLM text + lexical node reconstruction):

```bash
python scripts/prepare_bgl_campaign.py \
  --campaign-id bgl_inductive_fusion_v1_20260918 \
  --workspace-root /path/to/workspace \
  --matrix configs/ablation_fusion_bgl.yaml
```

Copy `campaigns/bgl_inductive_v1_20260918/split_lock.npz` into the new campaign dir first so the time cut stays locked. Family B arm `lexical_node_recon` trains on the **existing** `baseline_full` graph (`ablation.fusion.node_reconstruct: without_sbert`). Spot-check cid 1 / 2 / 3 `embedding_text` after stage 2 before splicing graphs.

Window/step changes would be extra Family A rebuilds, not train-only. Single-graph debug: [`6_GAE_Training_BGL_Colab.ipynb`](./6_GAE_Training_BGL_Colab.ipynb).

## Off the critical path

See [`ARCHIVE.md`](./ARCHIVE.md). `6_GAE_Training_Colab.ipynb` remains a single-run debugger, not a second matrix.
