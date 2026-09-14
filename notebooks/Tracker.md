# Here's how to integrate tracker into each notebook:


**Notebook 1 — Parser (creates the run tag)**
```python
from modules.run_tracker import RunTracker

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M")

tracker = RunTracker.init(
    run_tag=RUN_TAG,
    dataset="HDFS",
    changelog="Initial parse run with Drain.",
    justification="Baseline HDFS pipeline.",
    pipeline_stage="1_Parser.ipynb",
)

# ... after saving files ...
parser.export_templates(f"../data/processed/HDFS/{RUN_TAG}_hdfs_templates.json")
tracker.add_artifact("processed", f"data/processed/HDFS/{RUN_TAG}_hdfs_templates.json")
```

**Notebooks 2–5 — pick up an existing run**
```python
from modules.run_tracker import RunTracker

RUN_TAG = "20260407_2026"   # or however it's set
tracker = RunTracker.open(RUN_TAG)

# ... after saving an artifact ...
tracker.add_artifact("processed", f"data/processed/HDFS/{RUN_TAG}_embeddings.npz")
tracker.set_dataset_stats({ "total_blocks": 575061, ... })
```

**Notebook 6 — after validation and test evaluation**
```python
tracker.update_val_metrics("gae_main",
    best_threshold=best_threshold,
    f1=float(f1_scores[best_idx]),
    pr_auc=float(auc(recall, precision)),
)

tracker.update_test_metrics("gae_main",
    f1=float(f1_score(test_labels, test_preds)),
    pr_auc=float(test_pr_auc),
    roc_auc=float(test_roc_auc),
    confusion_matrix=confusion_matrix(test_labels, test_preds).tolist(),
)

tracker.add_artifact("models", f"models/{RUN_TAG}_attribute_gae.pt")
```

**If a later run depends on an earlier one** (e.g., a fine-tune that re-uses `20260407_2026` embeddings):
```python
tracker = RunTracker.init(
    run_tag=NEW_RUN_TAG,
    dataset="HDFS",
    changelog="Fine-tuned with lower LR.",
    justification="...",
    depends_on=["20260407_2026"],   # ← input artifacts auto-imported
)
```

The `depends_on` list causes `_resolve_input_artifacts` to copy the dependency run's `output_artifacts` into the new run's `input_artifacts` automatically, so the lineage is always explicit.

Made changes.