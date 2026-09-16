# HDFS ablation campaign

End-to-end protocol: **preprocess locally**, **train/eval on Colab**, keep a comparable artifact pack for every configuration.

This file is the operational contract. Implementation lives in this repository (`run_ablation.py`, `configs/ablation_*.yaml`, `src/modules/ablation.py`).

## Families (do not mix in one `train-only` graph)

| Family | YAML | Rebuild `graph_dataset.pt`? | Colab mode |
|---|---|---|---|
| **A — representation** | `configs/ablation_representation.yaml` | Yes | `--mode train-only --family A --campaign-dir …` |
| **B — train / architecture** | `configs/ablation_train.yaml` | No | `--mode train-only --family B` on `baseline_full` |

`configs/ablation_matrix.yaml` is deprecated. It now re-exports Family B only so an accidental shared-graph `train-only` run cannot fake a TF-IDF vs SBERT ablation.

Train-only **fails** if the config’s graph identity (LLM on/off, TF-IDF/SBERT, edge features, `feature_contract`) disagrees with `dataset_meta.json` on the bundle.

## Invariants

- One Drain parse, one HDFS block sequencing, **one split lock** (`split_lock.npz`) shared by every Family A graph.
- Seed 42, clean-train, val-F1 threshold, report test **F1 / PR-AUC / ROC-AUC** (rank by PR-AUC).
- Default `ablation.feature_contract: notebook_raw_v1` (thesis-comparable). `stabilized_v2` is an explicit Family A arm.
- Do not recompute SBERT on Colab (local ST 2.2.2 vs Colab ST 5.x). Train-only consumes frozen tensors.
- Never train from the Drive mount. Stage `.pt.gz` → `/content/workspace`, decompress there.
- Pin `GIT_REF` for the Colab clone; do not `git pull` mid-campaign. Cache fingerprints no longer include git SHA (SHA still lands on run manifests).

## Local prepare (Family A graphs)

From this repository root, Intel Mac PyTorch/PyG plus `requirements.txt`, Azure env for enrichment:

```bash
python scripts/prepare_hdfs_campaign.py \
  --campaign-id hdfs_ablation_20260916 \
  --workspace-root /path/to/workspace \
  --set experiment.dataset=hdfs
```

Equivalent:

```bash
python run_ablation.py --mode prepare-campaign \
  --campaign-id hdfs_ablation_20260916 \
  --workspace-root /path/to/workspace \
  --set experiment.dataset=hdfs
```

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

1. Open [`7_AblationStudy_Colab.ipynb`](./7_AblationStudy_Colab.ipynb) (clones this repository).
2. Set `CAMPAIGN_ID`, `FAMILY`, `GIT_REF` (campaign branch), `SMOKE=True` first.
3. Family B can start as soon as `graphs/baseline_full/graph_dataset.pt.gz` is on Drive.
4. Family A stages the whole campaign dir; the CLI unpacks **one** graph at a time.
5. Replay leaderboard from `outputs/hdfs/campaigns/<id>/` without retraining.

Suggested order: smoke → Family B → Family A → freeze leaderboard.

## Per-run artifact pack

`outputs/{dataset}/{campaign_id}_{arm}/`

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
  artifacts/cache/hdfs/          # resumable stage cache
  artifacts/runs/
```

## BGL

Out of scope for campaign 1 (path and sequence-filename mismatches). Reuse this two-family protocol after those are fixed; window/step changes are Family A (rebuild), not train-only.

## Off the critical path

See [`ARCHIVE.md`](./ARCHIVE.md). `6_GAE_Training_Colab.ipynb` remains a single-run debugger, not a second matrix.
