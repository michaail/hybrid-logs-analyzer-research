# HDFS / BGL ablation campaign

End-to-end protocol: **preprocess HDFS locally**, **train/eval on Colab**. BGL graphs may be prepared locally or in [`7_AblationStudy_BGL_Colab.ipynb`](./7_AblationStudy_BGL_Colab.ipynb) (`RUN_PREPARE=True`). Keep a comparable artifact pack for every configuration.

This file is the operational contract. Implementation lives in this repository (`run_ablation.py`, `configs/ablation_*.yaml`, `src/modules/ablation.py`).

## Families (do not mix in one `train-only` graph)

| Family | YAML | Rebuild `graph_dataset.pt`? | Colab mode |
|---|---|---|---|
| **A — representation (HDFS)** | `configs/ablation_representation.yaml` | Yes | `--mode train-only --family A --campaign-dir …` |
| **A — representation (BGL)** | `configs/ablation_representation_bgl.yaml` | Yes | same, BGL campaign dir |
| **B — train / architecture** | `configs/ablation_train.yaml` | No | `--mode train-only --family B` on `baseline_full` |

`configs/ablation_matrix.yaml` is deprecated. It now re-exports Family B only so an accidental shared-graph `train-only` run cannot fake a TF-IDF vs SBERT ablation.

Train-only **fails** if the config’s graph identity (LLM on/off, TF-IDF/SBERT, edge features, `feature_contract`) disagrees with `dataset_meta.json` on the bundle.

## Invariants

- One Drain parse, one HDFS block sequencing, **one split lock** (`split_lock.npz`) shared by every Family A graph.
- Seed 42, clean-train, val-F1 threshold, report test **F1 / PR-AUC / ROC-AUC** (rank by PR-AUC).
- LLM template enrichment is **Deepseek v4 Pro only** (`AZURE_OPENAI_DEPLOYMENT_DEEPSEEK_V4_PRO`). Mistral large/small are not ablation arms.
- Default `ablation.feature_contract: notebook_raw_v1` (thesis-comparable). `stabilized_v2` is an explicit Family A arm.
- Do not mix sentence-transformers versions inside one dataset campaign. HDFS stays local (ST 2.2.2). A BGL campaign is either fully local (ST 2.2.2) or fully Colab-prepared (ST 5.x via `RUN_PREPARE`). Train-only consumes frozen tensors.
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
  outputs/bgl/<campaign_id>_<arm>/
  outputs/bgl/campaigns/<id>/
  artifacts/cache/hdfs/
  artifacts/cache/bgl/
  artifacts/runs/
```

## BGL

Same two-family protocol as HDFS. Graphs are 20 min / 10 min windows. Family A YAML is `configs/ablation_representation_bgl.yaml` (no `feature_contract_stabilized_v2`). LLM enrichment is **Deepseek v4 Pro only**.

```bash
python scripts/prepare_bgl_campaign.py \
  --campaign-id bgl_ablation_20260917 \
  --workspace-root /path/to/workspace
```

Equivalent:

```bash
python run_ablation.py --mode prepare-campaign \
  --campaign-id bgl_ablation_20260917 \
  --workspace-root /path/to/workspace \
  --set experiment.dataset=bgl \
  --matrix configs/ablation_representation_bgl.yaml
```

Raw log: `data/raw/bgl/BGL_full.log`. You can prepare **locally** (`prepare_bgl_campaign.py`) or **on Colab** via [`7_AblationStudy_BGL_Colab.ipynb`](./7_AblationStudy_BGL_Colab.ipynb) (`RUN_PREPARE=True`). Colab prepare uses sentence-transformers 5.x — do not mix those graphs with a local BGL campaign (ST 2.2.2).

Colab prepare needs:

- Drive: `MyDrive/hybrid-log-analyzer-artifacts/data/raw/bgl/BGL_full.log`
- Secrets: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_DEEPSEEK_V4_PRO`

Then set `RUN_TRAIN=True` for Family B / A. Rank by PR-AUC under `outputs/bgl/campaigns/<id>/`.

Window/step changes would be extra Family A rebuilds, not train-only. Single-graph debug: [`6_GAE_Training_BGL_Colab.ipynb`](./6_GAE_Training_BGL_Colab.ipynb).

## Off the critical path

See [`ARCHIVE.md`](./ARCHIVE.md). `6_GAE_Training_Colab.ipynb` remains a single-run debugger, not a second matrix.
