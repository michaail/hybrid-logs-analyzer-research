# Notebooks off the HDFS ablation critical path

Keep these for archaeology; do not use them to produce campaign numbers.

| Notebook | Why it is off-path |
|---|---|
| `6_1_GAE_Training copy.ipynb` | Duplicate local trainer (`TEST_RUN`, different LR). |
| `6_GAE_Training.ipynb` | Inline GAE (not `src.modules.models.gae`). Local debug only. |
| `6_GAE_Training_BGL.ipynb` | Unfixed BGL trainer (edge scale). Prefer `_fixed` if exploring BGL later. |
| `6_GAE_Training_BGL_fixed.ipynb` | BGL-only; campaign 1 is HDFS. |
| `GraphMining.ipynb` | Neo4j sidecar, unversioned paths. |
| `ParserBGL.ipynb`, `3_Sequencer_TimeWindow.ipynb`, `4_Logs2Graphs_BGL.ipynb`, `5_PrepareDataset_BGL.ipynb` | BGL path/filename mismatches (`BGL/` vs `bgl/`, `*_W20S10.parquet`). Fix in a later campaign. |
| `4_Logs2Graphs.ipynb` | Topology/embedding design notes; does not emit `.pt` bundles. |

**On-path for campaign 1**

- Local HDFS stages 1–5 (or this repo’s `prepare-campaign` CLI) with an explicit run/campaign id — no “newest glob.”
- [`7_AblationStudy_Colab.ipynb`](./7_AblationStudy_Colab.ipynb) for train/eval + comparison.
- [`6_GAE_Training_Colab.ipynb`](./6_GAE_Training_Colab.ipynb) only to debug one graph.
