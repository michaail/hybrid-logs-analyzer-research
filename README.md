# Hybrid Logs Analyzer — research notebooks

Jupyter notebooks, Colab helpers, and the PROCESS narrative for HDFS/BGL
training and ablation. This repository is **not** standalone.

Training, parsing, and graph code live in the web-app repo
[`hybrid-logs-analyzer`](https://github.com/michaail/hybrid-logs-analyzer)
(`src/modules/`, `run_ablation.py`, `configs/`). Clone that repo and mount
this one as the `research/` git submodule.

## Use with the web-app checkout

```bash
git clone https://github.com/michaail/hybrid-logs-analyzer.git
cd hybrid-logs-analyzer
git submodule update --init
```

Run notebooks from the parent root so `import src.modules` resolves. Open
files under `research/notebooks/` (start with `notebooks/PROCESS.md` for the
stage-by-stage narrative).

Local ML environment still uses the parent's
`requirements-macos-intel.lock.txt`. Do not install `requirements-colab.txt`
on the Intel Mac baseline.

## Colab

[`notebooks/7_AblationStudy_Colab.ipynb`](notebooks/7_AblationStudy_Colab.ipynb)
and [`notebooks/6_GAE_Training_Colab.ipynb`](notebooks/6_GAE_Training_Colab.ipynb)
clone **the web-app repo** (not this repository) with submodules, then install
PyG via `research/scripts/install_colab.py` and `requirements-colab.txt`.

Do not clone either repository into Google Drive. Keep large artifacts on Drive
and stage them into `/content/workspace`.

## Drive sync

`scripts/drive_sync.py` copies ignored `data/`, `artifacts/`, and `outputs/`
trees. Point `--local-root` at the **web-app** checkout:

```bash
python research/scripts/drive_sync.py push \
  --remote "gdrive:hybrid-log-analyzer-artifacts" \
  --local-root /path/to/hybrid-logs-analyzer \
  --dataset hdfs --dry-run
```

## Ablation CLI

`run_ablation.py` remains in the parent repository:

```bash
python run_ablation.py --mode full --workspace-root . --set experiment.dataset=hdfs
```
