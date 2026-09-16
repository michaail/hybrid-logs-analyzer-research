"""BGL (BlueGene/L) log parser.

Specializes :class:`DrainParser` for the Blue Gene/L RAS event log format
released as part of the LogHub benchmark collection.

BGL line format
---------------
Each line is whitespace-separated::

    <LABEL> <UNIX_TS> <DATE> <NODE_ID> <DATETIME> <NODE_ID> <COMPONENT> <SUBCOMPONENT> <LEVEL> <MESSAGE>

Token-by-token:

===  ========================  =====================================================
 #   Field                     Example
===  ========================  =====================================================
 0   alert label               ``-`` (normal) or ``KERNDTLB``, ``APPSEV`` (anomaly)
 1   unix timestamp            ``1117838570``
 2   date (YYYY.MM.DD)         ``2005.06.03``
 3   node id                   ``R02-M1-N0-C:J12-U11``
 4   full datetime             ``2005-06-03-15.42.50.363779``
 5   node id (repeated)        ``R02-M1-N0-C:J12-U11``
 6   component                 ``RAS``
 7   subcomponent              ``KERNEL`` / ``APP`` / ``MMCS`` / ``DISCOVERY`` / ...
 8   log level                 ``INFO`` / ``WARNING`` / ``ERROR`` / ``FATAL`` / ...
 9+  free-form message         ``instruction cache parity error corrected``
===  ========================  =====================================================

The first six tokens (label … node_id_repeat) are stripped before being fed
to Drain so that the tree clusters on ``<component> <subcomponent> <level>
<message>`` — the stable, semantically-meaningful part.

Anomaly labelling
-----------------
Unlike HDFS, BGL labels are **inline**: a line is anomalous iff its first
token is anything other than ``-``. The alert code itself (e.g. ``KERNDTLB``
— kernel data-TLB error) can be used as a multi-class label; an
``is_anomaly`` boolean is exported for convenience.

Grouping unit
-------------
BGL has no native equivalent of HDFS' ``block_id``. Downstream sequence
builders typically group by either:

  * ``node_id`` — all events from one compute node form a per-node sequence
  * **fixed time windows** — e.g. 6 h sliding windows (the classical
    DeepLog / LogBERT setup on BGL)

Both fields (``node_id`` and ``timestamp``) are emitted on every row so the
grouping decision can be deferred to a sequencer notebook.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from .drain_parser import DrainParser


class BGLParser(DrainParser):
  """Drain3-based parser for Blue Gene/L (BGL) RAS event logs.

  Differences vs :class:`DrainParser` (HDFS default):
    * ``_HEADER_TOKENS = 6`` — strips label, unix_ts, date, node_id,
      datetime, node_id_repeat before Drain sees the line.
    * :meth:`_extract_row` returns BGL-specific columns (label,
      ``is_anomaly``, node_id, component, subcomponent, level, …).

  Example
  -------
  >>> parser = BGLParser(config_path="configs/drain_bgl.ini")
  >>> parser.fit_file("data/raw/BGL_full.log")
  >>> df = parser.annotate_file("data/raw/BGL_full.log")
  >>> df["is_anomaly"].mean()   # ~0.072 on the full LogHub BGL
  """

  # BGL has 6 fixed prefix tokens before the loggable payload:
  # label, unix_ts, date, node_id, datetime, node_id(repeat).
  _HEADER_TOKENS = 6

  # Parses the BGL full-precision datetime token, e.g. "2005-06-03-15.42.50.363779".
  _BGL_DATETIME_FMT = "%Y-%m-%d-%H.%M.%S.%f"

  # Node-id sanity regex; used only as a fallback — the primary node_id is
  # taken positionally from token index 3.
  _NODE_RE = re.compile(r"R\d+-M\d+-N[A-Z0-9]+(?:-[CI])?(?::J\d+-U\d+)?")

  def _extract_row(
    self,
    line: str,
    cluster_id: int,
    template_str: str,
    params: list[Any],
  ) -> dict[str, Any]:
    """Parse one BGL line into the annotated-row column dict.

    Notes
    -----
    * When a token is missing (e.g. truncated line), empty string is
      returned rather than ``None`` to keep the column dtype stable in
      pandas / parquet.
    * ``unix_ts`` is returned as ``int`` when parseable, else ``None``.
    * ``timestamp`` is returned as ``datetime`` when parseable, else
      ``None`` — downstream code must treat nulls explicitly.
    """
    # Split into at most 10 whitespace tokens: the 9 fixed header fields
    # plus a single "rest" token that contains the free-form message.
    parts = line.split(None, 9)
    # Pad missing trailing tokens so index access is safe on short lines.
    parts += [""] * (10 - len(parts))

    label        = parts[0]
    unix_ts_s    = parts[1]
    date_s       = parts[2]
    node_id      = parts[3]
    datetime_s   = parts[4]
    # parts[5] is the repeated node_id; intentionally discarded.
    component    = parts[6]
    subcomponent = parts[7]
    level        = parts[8]
    # parts[9] is the message body; the full line is retained under `raw`.

    try:
      unix_ts = int(unix_ts_s)
    except ValueError:
      unix_ts = None

    try:
      ts = datetime.strptime(datetime_s, self._BGL_DATETIME_FMT)
    except ValueError:
      ts = None

    # Safety net: if token 3 doesn't look like a node id (malformed line)
    # fall back to the first node-id match anywhere in the line.
    if not self._NODE_RE.fullmatch(node_id):
      m = self._NODE_RE.search(line)
      node_id = m.group(0) if m else node_id

    return {
      "label":        label,                    # "-" for normal, otherwise alert code
      "is_anomaly":   label != "-",             # bool, directly usable as a target
      "unix_ts":      unix_ts,                  # int | None
      "date":         date_s,                   # YYYY.MM.DD
      "node_id":      node_id,                  # Rxx-Mx-Nx-[CI]:Jxx-Uxx
      "timestamp":    ts,                       # datetime | None
      "component":    component,                # e.g. RAS
      "subcomponent": subcomponent,             # e.g. KERNEL, APP, MMCS
      "level":        level,                    # INFO, WARNING, ERROR, FATAL, …
      "raw":          line,
      "cluster_id":   cluster_id,
      "template":     template_str,
      "parameters":   list(params) if params else [],
    }
