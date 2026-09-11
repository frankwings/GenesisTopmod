"""
trajectory_logger.py — JSONL trajectory logger for the VLM operator loop.

Each call to log_step() appends one JSON record to a .jsonl file and
optionally saves a PNG triptych and/or an OBJ snapshot.

Record schema
-------------
{
  "step"          : int,
  "iou_before"    : float,
  "iou_after"     : float | null,
  "delta_iou"     : float | null,
  "candidates"    : [...],     # candidate dicts (labels, face_ids, descriptions)
  "vlm_choice"    : int,       # region label the VLM chose
  "vlm_rationale" : str,       # raw VLM reply
  "op"            : str,       # operator name
  "op_kwargs"     : {...},     # operator kwargs (dist, etc.)
  "view_png_path" : str | null,
  "obj_path"      : str | null,
  "timestamp_s"   : float,
}

Usage
-----
    from trajectory_logger import TrajectoryLogger

    logger = TrajectoryLogger(out_dir="eval_out/vlm_run")

    # Before operator execution:
    logger.log_step(
        step        = 150,
        iou_before  = 0.912,
        candidates  = candidates,
        vlm_choice  = 2,
        vlm_rationale = vlm_reply,
        op          = "extrude",
        op_kwargs   = {"dist": 0.05},
        view_png    = png_bytes,     # from encode_state; None to skip
        verts_np    = verts_np,      # for OBJ snapshot; None to skip
        tris_np     = tris_np,
    )

    # After operator + one eval step:
    logger.update_last_iou_after(iou_after=0.934)

    logger.close()
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class TrajectoryLogger:
    """Append-only JSONL trajectory logger.

    Parameters
    ----------
    out_dir : str or Path
        Directory where trajectory.jsonl and assets (PNG, OBJ) are written.
        Created if it doesn't exist.
    save_obj : bool
        If True, write an OBJ snapshot for each logged step.
    """

    def __init__(
        self,
        out_dir:  str | Path = "eval_out/vlm_run",
        save_obj: bool       = False,
    ) -> None:
        self._out_dir  = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._save_obj = save_obj

        self._jsonl_path = self._out_dir / "trajectory.jsonl"
        self._fh         = open(self._jsonl_path, "a+", encoding="utf-8")
        self._last_pos: Optional[int] = None   # byte offset of last written record

    # ── public API ────────────────────────────────────────────────────────

    def log_step(
        self,
        step:          int,
        iou_before:    float,
        candidates:    List[Dict[str, Any]],
        vlm_choice:    int,
        vlm_rationale: str,
        op:            str,
        op_kwargs:     Dict[str, Any],
        view_png:      Optional[bytes]      = None,
        verts_np:      Optional[np.ndarray] = None,
        tris_np:       Optional[np.ndarray] = None,
        iou_after:     Optional[float]      = None,
    ) -> None:
        """Write one record to the JSONL file.

        Parameters
        ----------
        step : int
            Optimisation step number at which the VLM operator was triggered.
        iou_before : float
            IoU just before the operator was applied.
        candidates : list[dict]
            Candidate descriptors from build_candidates().
        vlm_choice : int
            Region label chosen by the VLM.
        vlm_rationale : str
            Raw VLM reply text.
        op : str
            Operator name (e.g. "extrude").
        op_kwargs : dict
            Keyword arguments passed to execute_operator().
        view_png : bytes or None
            PNG triptych bytes from encode_state().  Saved as
            ``step_{step:04d}_state.png`` if provided.
        verts_np : ndarray or None
            Vertex array for OBJ snapshot (only if save_obj=True).
        tris_np : ndarray or None
            Triangle array for OBJ snapshot (only if save_obj=True).
        iou_after : float or None
            IoU after applying the operator (can be filled in later via
            update_last_iou_after()).
        """
        png_path = None
        if view_png is not None:
            png_name = f"step_{step:04d}_state.png"
            png_path = str(self._out_dir / png_name)
            with open(png_path, "wb") as f:
                f.write(view_png)

        obj_path = None
        if self._save_obj and verts_np is not None and tris_np is not None:
            obj_name = f"step_{step:04d}_mesh.obj"
            obj_path = str(self._out_dir / obj_name)
            _write_obj(obj_path, verts_np, tris_np)

        delta_iou = (
            float(iou_after) - float(iou_before)
            if iou_after is not None else None
        )

        # Strip heavy array data from candidates before serialising
        cands_lite = [
            {k: v for k, v in c.items() if k not in ("extrude_dir",)}
            for c in candidates
        ]

        record: Dict[str, Any] = {
            "step":          step,
            "iou_before":    float(iou_before),
            "iou_after":     float(iou_after) if iou_after is not None else None,
            "delta_iou":     delta_iou,
            "candidates":    cands_lite,
            "vlm_choice":    vlm_choice,
            "vlm_rationale": vlm_rationale[:1000],   # truncate long replies
            "op":            op,
            "op_kwargs":     {
                k: (float(v) if isinstance(v, (np.floating, float)) else
                    v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in op_kwargs.items()
            },
            "view_png_path": png_path,
            "obj_path":      obj_path,
            "timestamp_s":   time.time(),
        }

        self._last_pos = self._fh.tell()
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def update_last_iou_after(self, iou_after: float) -> None:
        """Patch the last written record's iou_after and delta_iou in-place.

        This is safe because JSONL records are newline-terminated and the file
        is append-only — we re-write only the last line.
        """
        if self._last_pos is None:
            return

        # Read the last line
        self._fh.flush()
        pos = self._last_pos
        self._fh.seek(pos)
        line = self._fh.readline()
        if not line.strip():
            return

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return

        record["iou_after"] = float(iou_after)
        record["delta_iou"] = float(iou_after) - float(record["iou_before"])

        # Truncate to pos and rewrite the updated record
        self._fh.seek(pos)
        self._fh.truncate()
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        # Reset position to end for next append
        self._fh.seek(0, 2)

    def close(self) -> None:
        """Flush and close the JSONL file."""
        self._fh.flush()
        self._fh.close()

    # ── context manager ───────────────────────────────────────────────────

    def __enter__(self) -> "TrajectoryLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── utility ───────────────────────────────────────────────────────────

    @property
    def jsonl_path(self) -> Path:
        """Path to the JSONL trajectory file."""
        return self._jsonl_path

    def load_trajectory(self) -> List[Dict[str, Any]]:
        """Load and return all records from the JSONL file."""
        records = []
        with open(self._jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records


# ── OBJ writer ────────────────────────────────────────────────────────────────

def _write_obj(
    path:     str,
    verts_np: np.ndarray,   # [V, 3]
    tris_np:  np.ndarray,   # [F, 3]
) -> None:
    """Write a minimal Wavefront OBJ file (vertices + triangles, no UVs)."""
    with open(path, "w") as f:
        f.write(f"# {len(verts_np)} vertices, {len(tris_np)} faces\n")
        for v in verts_np:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in tris_np:
            # OBJ indices are 1-based
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
