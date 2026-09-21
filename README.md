# Hybrid Logs Analyzer — research pipeline

Jupyter notebooks, Colab helpers, and the HDFS/BGL ablation CLI. This
repository now contains the training pipeline (`src/modules/`,
`run_ablation.py`, `configs/`) so parse → graph → train can run from this
checkout.

The product web app (API, frontend, inference service) stays in
[`hybrid-logs-analyzer`](https://github.com/michaail/hybrid-logs-analyzer).

## Layout

```
src/modules/          # parser, enricher, sequencer, graphs, GAE, ablation
modules -> src/modules  # notebook imports (`from modules.parser import …`)
configs/              # Drain INI + ablation YAML
run_ablation.py       # parse / prepare-campaign / train-only / report
scripts/              # prepare_hdfs_campaign.py, install_colab.py, drive_sync.py
notebooks/            # PROCESS, ABLATION, Colab runners
```

Open notebooks from this repository root so `import src.modules` and
`import modules` both resolve. Start with `notebooks/PROCESS.md` for the
stage-by-stage narrative and `notebooks/ABLATION.md` for the HDFS/BGL campaign.

Local ML environment: `pip install -r requirements.txt` plus PyTorch / PyG.
Do not install `requirements-colab.txt` on the Intel Mac baseline.

## HDFS ablation campaign

Preprocess locally, train on Colab, keep a comparable eval pack per config.

1. **Local:** build Family A graphs (one split lock, gzipped bundles):

   ```bash
   python scripts/prepare_hdfs_campaign.py \
     --campaign-id hdfs_ablation_YYYYMMDD \
     --workspace-root /path/to/workspace
   ```

2. **Upload** `campaigns/<id>/` to Drive `hybrid-log-analyzer-artifacts/`
   (see Drive sync below).
3. **Colab:** [`notebooks/7_AblationStudy_Colab.ipynb`](notebooks/7_AblationStudy_Colab.ipynb)
   with `FAMILY="B"` (reuse `baseline_full`) then `FAMILY="A"` (one graph at a
   time). Smoke first (`SMOKE=True`).
4. **Compare** `outputs/hdfs/campaigns/<id>/leaderboard.csv` and the PNG rollup.
   Replay does not require retraining.

Family A changes embeddings or graph tensors and **must not** share a single
`--graph-dataset`. Family B may. Details: [`notebooks/ABLATION.md`](notebooks/ABLATION.md).
Off-path notebooks: [`notebooks/ARCHIVE.md`](notebooks/ARCHIVE.md).

## BGL ablation campaign

Published research campaign: prepare `configs/ablation_bgl_llm_closed.yaml`
then `configs/ablation_bgl_pb3.yaml` into the same campaign id, train Family A,
then Isolation Forest. Rank by test PR-AUC vs `hybrid_llm`. Details:
[`notebooks/ABLATION.md`](notebooks/ABLATION.md).

## Colab

[`notebooks/7_AblationStudy_Colab.ipynb`](notebooks/7_AblationStudy_Colab.ipynb)
is the campaign runner. [`notebooks/6_GAE_Training_Colab.ipynb`](notebooks/6_GAE_Training_Colab.ipynb)
is a single-graph debugger only.

Both clone **this** repository, then install PyG via
`scripts/install_colab.py` and `requirements-colab.txt`. Pin `GIT_REF` to the
campaign branch and do not pull mid-run.

Do not clone the repository into Google Drive. Keep large artifacts on Drive
and stage them into `/content/workspace`.

## Drive sync

`scripts/drive_sync.py` copies ignored `data/`, `artifacts/`, `outputs/`, and
`campaigns/` trees. Point `--local-root` at the workspace used for
`prepare-campaign`:

```bash
python scripts/drive_sync.py push \
  --remote "gdrive:hybrid-log-analyzer-artifacts" \
  --local-root /path/to/workspace \
  --dataset hdfs --dry-run
```

Expected Drive layout:

```
hybrid-log-analyzer-artifacts/
  campaigns/<id>/manifest.json
  campaigns/<id>/graphs/<arm>/graph_dataset.pt.gz
  outputs/hdfs/<campaign_id>_<arm>/
  outputs/hdfs/campaigns/<id>/
```

## Ablation CLI

Run from this repository root:

```bash
python run_ablation.py --mode prepare-campaign --campaign-id hdfs_ablation_YYYYMMDD --workspace-root .
python run_ablation.py --mode train-only --family B --campaign-dir campaigns/<id> --workspace-root .
python run_ablation.py --mode train-only --family A --campaign-dir campaigns/<id> --workspace-root .
python run_ablation.py --mode report --campaign-id hdfs_ablation_YYYYMMDD --workspace-root .
```
