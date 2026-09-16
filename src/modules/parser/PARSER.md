# Parser Module

Drain3-based log parsing. Produces a uniform `cluster_id`-keyed event
stream that every downstream stage of the pipeline consumes.

## Class hierarchy

```
DrainParser                          (base — HDFS default)
├── fit_file(log_path)               — pass 1: learn templates online
├── annotate_file(log_path)          — pass 2: attribute each line to final templates
├── _extract_row(line, …) → dict     — HOOK: subclass overrides for dataset-specific columns
├── export_templates(out_path)       — dump {cluster_id, template, count, examples} JSON
├── save / load                      — Drain3 FilePersistence snapshot round-trip
└── validate()                       — heuristic quality checks
    │
    └── BGLParser(DrainParser)       (subclass — Blue Gene/L RAS logs)
        ├── _HEADER_TOKENS = 6       — strips label/ts/date/node/datetime/node_rep
        └── _extract_row()           — emits BGL-specific columns
```

## Subclassing for a new log format

Two hooks are usually enough:

1. **`_HEADER_TOKENS`** (class attribute, int) — how many whitespace-separated
   prefix tokens to strip before feeding the line to Drain. Everything after
   that is what Drain will cluster on. For HDFS it is 3
   (`<date> <time> <thread>`); for BGL it is 6
   (`<label> <unix_ts> <date> <node> <datetime> <node_rep>`).

2. **`_extract_row(line, cluster_id, template_str, params) → dict`**
   (instance method) — build the dataframe row for one matched line. The base
   class provides an HDFS-flavoured default that emits `date / time / thread /
   timestamp / raw / cluster_id / template / parameters / block_id`. Override
   to emit whatever columns your dataset needs.

Nothing else has to be overridden — `fit_file`, `annotate_file`,
`export_templates`, `validate`, `save`, `load` are format-agnostic and inherit
cleanly.

## Configs

Drain hyperparameters and **regex masking rules** live in plain INI files:

| File                           | For       | Key masks                                  |
|--------------------------------|-----------|--------------------------------------------|
| `configs/drain.ini`            | HDFS      | `BLK`, `IP`, `UUID`                        |
| `configs/drain_bgl.ini`        | BGL       | `NODE`, `IP`, `HEX`, `ADDR`, `CORE`, `PATH`, `REASON` |

Masking is applied *before* Drain tokenises the line; without it the tree
would branch on every unique block-id / node-id / hex address and explode
the template count.

## Drain3 primer

Drain3 is an online implementation of the Drain parser: it incrementally
builds a parse tree and assigns each incoming log message to one of a set of
learned templates. Two key properties we rely on:

- **`cluster_id` is stable** across the whole parsing session. Template
  strings may still generalise (additional tokens replaced by `<*>`) as new
  variability is seen, but the integer id assigned at cluster creation is
  immutable. Downstream code therefore always joins on `cluster_id`, never on
  the template string.
- **`match()` vs `add_log_message()`**. The first is read-only — it reports
  which cluster a line belongs to against the current tree state. The second
  additionally mutates the tree (may create / re-parameterise clusters).
  `fit_file` uses the latter; `annotate_file` uses the former so the second
  pass only ever sees *final* templates.

## Usage

### HDFS

```python
from src.modules.parser import DrainParser

parser = DrainParser(config_path="configs/drain.ini")
parser.fit_file("data/raw/HDFS_full.log")
df = parser.annotate_file("data/raw/HDFS_full.log")
parser.export_templates("data/processed/hdfs_templates.json")
parser.save("models/hdfs_drain_parser.bin")
```

### BGL

```python
from src.modules.parser import BGLParser

parser = BGLParser(config_path="configs/drain_bgl.ini")
parser.fit_file("data/raw/BGL_full.log")
df = parser.annotate_file("data/raw/BGL_full.log")
parser.export_templates("data/processed/bgl_templates.json")
parser.save("models/bgl_drain_parser.bin")

# BGL-specific columns, inline labels, multi-class alert codes:
df[["label", "is_anomaly", "node_id", "component", "subcomponent", "level", "cluster_id"]].head()
```

See `notebooks/Parser.ipynb` and `notebooks/ParserBGL.ipynb` for the full
walkthrough of each dataset.

## Testing

```bash
pytest tests/test_bgl_parser.py -v
```

The test suite covers header stripping, all row-schema invariants,
malformed-input handling (bad datetime, bad unix_ts, truncated lines,
`NULL` nodes, I/O nodes), and a full `fit_file` + `annotate_file`
roundtrip on a 4-line synthetic BGL file.
