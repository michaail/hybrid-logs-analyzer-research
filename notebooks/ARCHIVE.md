# Notebooks off the HDFS ablation critical path

Keep these for archaeology; do not use them to produce campaign numbers.

| Notebook | Why it is off-path |
|---|---|
| `6_1_GAE_Training copy.ipynb` | Duplicate local trainer (`TEST_RUN`, different LR). |
| `6_GAE_Training.ipynb` | Inline GAE (not `src.modules.models.gae`). Local debug only. |
| `6_GAE_Training_BGL.ipynb` | Unfixed BGL trainer (edge scale). Prefer `_fixed` locally or the BGL Colab notebook. |
| `6_GAE_Training_BGL_fixed.ipynb` | Local BGL trainer (inline GAE). Use [`6_GAE_Training_BGL_Colab.ipynb`](./6_GAE_Training_BGL_Colab.ipynb) on Colab. |
| `GraphMining.ipynb` | Neo4j sidecar, unversioned paths. |
| `ParserBGL.ipynb`, `3_Sequencer_TimeWindow.ipynb`, `4_Logs2Graphs_BGL.ipynb`, `5_PrepareDataset_BGL.ipynb` | BGL path/filename mismatches (`BGL/` vs `bgl/`, `*_W20S10.parquet`). Fix in a later campaign. |
| `4_Logs2Graphs.ipynb` | Topology/embedding design notes; does not emit `.pt` bundles. |

**On-path for campaign 1**

- Local HDFS stages 1–5 (or this repo’s `prepare-campaign` CLI) with an explicit run/campaign id — no “newest glob.”
- [`7_AblationStudy_Colab.ipynb`](./7_AblationStudy_Colab.ipynb) for HDFS train/eval + comparison.
- [`7_AblationStudy_BGL_Colab.ipynb`](./7_AblationStudy_BGL_Colab.ipynb) for BGL prepare (optional, Colab) + train/eval (`RUN_PREPARE` / `RUN_TRAIN`).
- [`6_GAE_Training_Colab.ipynb`](./6_GAE_Training_Colab.ipynb) only to debug one HDFS graph.
- [`6_GAE_Training_BGL_Colab.ipynb`](./6_GAE_Training_BGL_Colab.ipynb) to debug one BGL graph.
