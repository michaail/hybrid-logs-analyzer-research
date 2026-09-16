from typing import List, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path
import json
import re

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig


class UnmatchedLogLine(ValueError):
  """Raised when strict annotation encounters a line with no frozen template."""

  def __init__(self, line_number: int) -> None:
    self.line_number = line_number
    super().__init__(f"Log line {line_number} did not match a frozen template.")


class DrainParser:
  """
  Wrapper around Drain3's TemplateMiner for:
    - training on a log file
    - exporting learned templates
    - validating template quality heuristically

  Default implementation targets the HDFS log format. Dataset-specific
  subclasses (e.g. :class:`BGLParser`) can override:
    - the class attribute ``_HEADER_TOKENS`` (how many prefix tokens to strip)
    - the hook method :meth:`_extract_row` (what columns to emit per line)
  """

  # Number of leading whitespace-separated tokens to strip before passing
  # the line to Drain.
  # HDFS format: <DDMMYY> <HHMMSS> <thread_id> <LEVEL> <component>: <message>
  # Stripping the first 3 (date, time, thread) removes always-variable header
  # fields; LEVEL and component are kept because they help Drain cluster lines.
  _HEADER_TOKENS = 3
  _HDFS_BLOCK_ID_RE = re.compile(r"blk_-?\d+")

  @staticmethod
  def _preprocess_line(line: str, strip_tokens: int = 3) -> str:
    """Strip the first `strip_tokens` whitespace-separated tokens from *line*.

    This removes the variable header (e.g. date/time/thread for HDFS,
    label/timestamp/node for BGL) so Drain only sees the stable
    LEVEL + COMPONENT + MESSAGE portion.
    Returns the original line unchanged if it has fewer tokens than requested.
    """
    parts = line.split(None, strip_tokens)
    return parts[strip_tokens] if len(parts) > strip_tokens else line

  @classmethod
  def extract_hdfs_block_id(cls, line: str) -> str | None:
    """Return the first HDFS block ID using the parser's sequence identity rule."""
    block_match = cls._HDFS_BLOCK_ID_RE.search(line)
    return block_match.group(0) if block_match else None

  def __init__(self, config_path: str | None = None, persistence_path: Optional[str] = None):
    cfg = TemplateMinerConfig()
    if config_path:
      cfg.load(config_path)

    persistence = None
    if persistence_path:
      Path(persistence_path).parent.mkdir(parents=True, exist_ok=True)
      persistence = FilePersistence(persistence_path)

    self.miner = TemplateMiner(persistence_handler=persistence, config=cfg)
    self._persistence_path = persistence_path
    self._config_path = config_path

    # cluster_id is stable throughout the online parsing process;
    # template strings evolve as tokens are replaced with <*>, so we
    # key example lines by ID to avoid orphaned entries.
    self.cluster_id_to_lines: Dict[int, List[str]] = defaultdict(list)
    self.line_template_ids: List[str] = []


  def fit_file(self, log_path: str, max_lines: int | None = None) -> None:
    log_file = Path(log_path)
    print(f"[INFO] Training Drain3 on: {log_file}")

    with log_file.open("r", errors="replace") as f:
      for i, raw in enumerate(f, start=1):
        line = raw.rstrip("\n")
        if not line:
          continue

        content = self._preprocess_line(line, self._HEADER_TOKENS)
        result = self.miner.add_log_message(content) # Process the log line through Drain3
        cluster_id = result.get("cluster_id")

        if cluster_id is None:
          print(f"[WARN] No cluster for line {i}: {line[:120]}...")
          continue

        # Store for validation — keyed by stable cluster_id, not by the
        # template string which may still evolve after this point.
        self.line_template_ids.append(cluster_id)
        self.cluster_id_to_lines[cluster_id].append(line)

        # Early stopping
        if max_lines is not None and i >= max_lines:
          print(f"[INFO] Stopped early at {max_lines} lines")
          break

        if i % 100000 == 0:
          print(f"[INFO] Processed {i} lines...")

      print(f"[INFO] Parsed {len(self.line_template_ids)} log lines total.")
      print(f"[INFO] Learned {len(self.miner.drain.id_to_cluster)} distinct templates (final).")


  def _extract_row(
    self,
    line: str,
    cluster_id: int,
    template_str: str,
    params: List[Any],
  ) -> Dict[str, Any]:
    """Hook: build one output row from a matched log line.

    Default implementation produces HDFS-specific columns:
      - date, time, thread   : raw header fields
      - timestamp            : datetime parsed from date+time
      - raw                  : full original log line
      - cluster_id           : stable Drain cluster id
      - template             : final (fully generalised) template string
      - parameters           : list of values extracted for each <*> token
      - block_id             : first blk_XXX value found in the line (or None)

    Subclasses targeting other log formats (e.g. :class:`BGLParser`)
    should override this method to return an alternative column dict.
    """
    from datetime import datetime

    parts = line.split(None, 3)  # date time thread rest
    date_s, time_s, thread_s = (parts + ["", "", ""])[:3]

    try:
      ts = datetime.strptime(f"{date_s} {time_s}", "%d%m%y %H%M%S")
    except ValueError:
      ts = None

    block_id = self.extract_hdfs_block_id(line)

    return {
      "date":       date_s,
      "time":       time_s,
      "thread":     thread_s,
      "timestamp":  ts,
      "raw":        line,
      "cluster_id": cluster_id,
      "template":   template_str,
      "parameters": list(params) if params else [],
      "block_id":   block_id,
    }


  def annotate_file(
    self,
    log_path: str,
    max_lines: int | None = None,
    *,
    unmatched: str = "skip",
  ):
    """Second pass over the log file using the final learned templates.

    Returns a pandas DataFrame with one row per log line. Column schema is
    determined by :meth:`_extract_row` (overridable by subclasses).
    ``unmatched="fail"`` raises :class:`UnmatchedLogLine` instead of skipping.
    """
    if unmatched not in {"skip", "fail"}:
      raise ValueError("unmatched must be 'skip' or 'fail'.")
    import pandas as pd

    rows = []
    log_file = Path(log_path)

    with log_file.open("r", errors="replace") as f:
      for i, raw in enumerate(f, start=1):
        line = raw.rstrip("\n")
        if not line:
          continue

        # Strip the same header tokens that were removed during fit
        content = self._preprocess_line(line, self._HEADER_TOKENS)

        # Match against final templates (does not mutate clusters)
        match = self.miner.match(content)
        if match is None:
          if unmatched == "fail":
            raise UnmatchedLogLine(i)
          continue

        template_tokens = match.get_template()  # list[str]
        template_str    = " ".join(template_tokens)
        params          = self.miner.get_parameter_list(template_tokens, content)

        row = self._extract_row(line, match.cluster_id, template_str, list(params) if params else [])
        row["line_number"] = i
        rows.append(row)

        if max_lines is not None and i >= max_lines:
          break

        if i % 500_000 == 0:
          print(f"[INFO] Annotated {i} lines...")

    print(f"[INFO] Annotated {len(rows)} lines.")
    return pd.DataFrame(rows)


  def export_templates(self, out_path: str) -> None:
    """Export learned templates to a file."""
    records = []
    for cluster in self.miner.drain.id_to_cluster.values():
        tmpl = " ".join(cluster.log_template_tokens)
        lines = self.cluster_id_to_lines.get(cluster.cluster_id, [])
        records.append({
            "cluster_id": cluster.cluster_id, # Stable ID for this template cluster
            "template": tmpl,                 # The template string learned by Drain3
            "count": cluster.size,            # authoritative count from Drain3
            "examples": lines[:5]             # Include up to 5 example lines for context
        })

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
      json.dump(records, f, indent=2)

    print(f"[INFO] Exported {len(records)} templates → {output_path}")


  def save(self, path: Optional[str] = None) -> None:
    """Manually trigger a drain3 snapshot save.

    If *path* is given and differs from the current persistence path, a new
    FilePersistence pointing at that path is used for this one-off save.
    Otherwise the persistence handler configured at construction time is used.
    """
    target_path = path or self._persistence_path
    if target_path is None:
      raise ValueError("No persistence_path configured and no path argument given.")

    out = Path(target_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if target_path != self._persistence_path:
      # One-off save to a different file — swap handler temporarily
      original = self.miner.persistence_handler
      self.miner.persistence_handler = FilePersistence(target_path)
      self.miner.save_state("manual_save")
      self.miner.persistence_handler = original
    else:
      self.miner.save_state("manual_save")

    print(f"[INFO] DrainParser snapshot saved → {out}  ({out.stat().st_size / 1_048_576:.2f} MB)")

  @classmethod
  def load(cls, path: str, config_path: Optional[str] = None) -> "DrainParser":
    """Restore a previously saved DrainParser from a drain3 FilePersistence snapshot.

    Passing the same *config_path* as during training ensures masking rules and
    Drain hyperparameters are identical — only cluster state is loaded from the
    snapshot file (as per drain3 design).

    Drain3 FilePersistence reconstructs objects with jsonpickle. Production inference
    may call this only on a checksum-bound ``drain_parser.bin`` from a Publisher-admitted
    preprocessing bundle. The public API and isolated validator must not deserialize it.
    """
    if not Path(path).exists():
      raise FileNotFoundError(f"Snapshot file not found: {path}")
    # Constructing with FilePersistence triggers load_state() automatically
    obj = cls(config_path=config_path, persistence_path=path)
    n_clusters = len(obj.miner.drain.id_to_cluster)
    print(f"[INFO] DrainParser loaded ← {path}  ({n_clusters} templates)")
    return obj

  def _final_clusters(self) -> List[tuple]:
    """Return list of (template_str, size) from Drain's authoritative cluster store."""
    return [
      (" ".join(c.log_template_tokens), c.size)
      for c in self.miner.drain.id_to_cluster.values()
    ]


  def _print_template_support_distribution(self) -> List[int]:
    support_counts = sorted(size for _, size in self._final_clusters())
    min_sup = support_counts[0]
    max_sup = support_counts[-1]
    avg_sup = sum(support_counts) / len(support_counts)
    print(
      f"[INFO] Template support (lines per template): "
      f"min={min_sup}, max={max_sup}, avg={avg_sup:.1f}"
    )
    return support_counts


  def _validate_singleton_templates(self, support_counts: List[int], n_templates: int) -> None:
    singletons = sum(1 for s in support_counts if s == 1)
    if singletons / n_templates > 0.3:
      print(
        f"[WARN] High fraction of singleton templates "
        f"({singletons}/{n_templates} ≈ {singletons/n_templates:.1%}). "
        f"Drain might be overfitting (simply memorizing lines)."
      )
    else:
      print("[OK] Singleton template fraction looks reasonable.")


  def _validate_overly_generic_templates(self) -> None:
    too_generic = []
    for tmpl, size in self._final_clusters():
      num_tokens = len(tmpl.split())
      num_wild = tmpl.count("<*>")
      if num_tokens > 0 and num_wild / num_tokens > 0.7:
        too_generic.append((tmpl, size))

    if too_generic:
      print(
        f"[WARN] Found {len(too_generic)} templates that look very generic "
        f"(>70% wildcard tokens). Examples:"
      )
      for tmpl, sup in too_generic[:5]:
        print(f"      support={sup:4d} | template='{tmpl}'")
      if len(too_generic) > 5:
        print("      ...")
    else:
      print("[OK] No overly generic templates detected by wildcard heuristic.")


  def _validate_overly_specific_templates(self) -> None:
    too_specific = []
    for tmpl, size in self._final_clusters():
      if size == 1 and "<*>" not in tmpl:
        too_specific.append(tmpl)

    if too_specific:
      print(
        f"[WARN] Found {len(too_specific)} templates that are "
        f"single-use with no wildcards (likely overfitting)."
      )
      for tmpl in too_specific[:5]:
        print(f"      '{tmpl}'")
      if len(too_specific) > 5:
        print("      ...")
    else:
      print("[OK] No obviously over-specific single-use templates found.")


  def validate(self) -> None:
    """Heuristic validation of learned templates."""
    print("\n[INFO] Running template validation...")

    total_lines = len(self.line_template_ids)
    n_templates = len(self.miner.drain.id_to_cluster)

    if total_lines == 0:
        print("[ERROR] No lines were parsed. Check your log path / format.")
        return

    print(f"[INFO] Total lines parsed   : {total_lines}")
    print(f"[INFO] Distinct templates   : {n_templates}")

    assigned = len(self.line_template_ids)
    if assigned != total_lines:
        print(f"[WARN] Coverage mismatch: {assigned}/{total_lines} lines have a cluster_id.")
    else:
        print("[OK] 100% of lines received a template assignment.")

    support_counts = self._print_template_support_distribution()
    self._validate_singleton_templates(support_counts, n_templates)
    self._validate_overly_generic_templates()
    self._validate_overly_specific_templates()
