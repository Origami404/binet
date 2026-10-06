"""Eager, offline probe records. NPZ storage is readable with NumPy alone."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from binet.records import RECORD_BYTES, RECORD_LAYOUT, depth_checked, geometry, sites_checked


RECORD_DTYPE = np.dtype([("warp_id", "<u4"), ("site_id", "<u4"),
                         ("timestamp_ns", "<u8"), ("cta_id", "<u4")])
_RAW_DTYPE = np.dtype([("tag", "<u4"), ("cta_id", "<u4"), ("timestamp", "<u8")])
_CAPTURE_FIELDS = ("kernel", "arch", "source_cubin_sha256", "grid", "block",
                   "clock", "timestamp_unit", "ring_depth", "retention")


class Trace:
    """One launch's compact records and metadata, fully resident in memory.

    Records are a one-dimensional structured NumPy array with RECORD_DTYPE.
    Compaction retains physical ring-slot order within each launch-global warp;
    this is not execution order. Repeated hits and timestamp ties are preserved.
    Event pairing and correlation belong to the consumer.
    """

    def __init__(self, records, metadata):
        self.records = records
        self.metadata = deepcopy(metadata)
        self._validate()

    @classmethod
    def from_buffer(cls, data, metadata):
        """Compact a completed device buffer, discarding only slots whose tag is zero.

        Unknown nonzero tags fail with their warp/slot location. Observed CTA IDs
        are preserved, including inconsistent IDs, for downstream inspection.
        No CUDA driver or Torch import is needed; data must support the buffer protocol.
        """
        m = metadata
        if not isinstance(m, dict) or not {*_CAPTURE_FIELDS, "sites"} <= m.keys():
            raise ValueError("missing producer metadata fields")
        if m.get("record_layout") != RECORD_LAYOUT:
            raise ValueError("invalid device record layout")
        grid, block, warp_count = geometry(m["grid"], m["block"])
        depth = depth_checked(m["ring_depth"])
        source = memoryview(data)
        expected_size = warp_count * depth * RECORD_BYTES
        if source.nbytes != expected_size:
            raise ValueError(f"buffer size {source.nbytes} differs from geometry/layout size {expected_size}")
        tags = {}
        if not isinstance(m["sites"], list):
            raise ValueError("producer sites must be a tag/site mapping list")
        for entry in m["sites"]:
            if not isinstance(entry, dict) or not {"tag", "site"} <= entry.keys():
                raise ValueError("invalid tag/site mapping entry")
            tag, site = entry["tag"], entry["site"]
            if (type(tag) is not int or not 0 < tag <= 0xffffffff or tag in tags
                    or type(site) is not int or not 0 < site <= 0xffffffff or site in tags.values()):
                raise ValueError("invalid or duplicate tag/site mapping")
            tags[tag] = site
        if not tags:
            raise ValueError("empty tag mapping")
        raw = np.frombuffer(source, dtype=_RAW_DTYPE)
        occupied = np.flatnonzero(raw["tag"])
        hits = raw[occupied]
        keys = np.array(sorted(tags), dtype="<u4")
        positions = np.searchsorted(keys, hits["tag"])
        known = keys[np.minimum(positions, len(keys) - 1)] == hits["tag"]
        if not known.all():
            index = int(np.flatnonzero(~known)[0])
            warp, slot = divmod(int(occupied[index]), depth)
            raise ValueError(f"unknown nonzero tag {int(hits['tag'][index])} at warp {warp}, slot {slot}")
        records = np.empty(len(hits), dtype=RECORD_DTYPE)
        records["warp_id"] = occupied // depth
        records["site_id"] = np.array([tags[int(tag)] for tag in keys], dtype="<u4")[positions]
        records["timestamp_ns"] = hits["timestamp"]
        records["cta_id"] = hits["cta_id"]
        info = {key: deepcopy(m[key]) for key in _CAPTURE_FIELDS}
        info.update(grid=list(grid), block=list(block), sites=sorted(tags.values()))
        return cls(records, info)

    @classmethod
    def load(cls, path):
        """Load trace.npz (or a profile directory) without its source buffer or sidecars."""
        path = Path(path)
        if path.is_dir():
            path = path / "trace.npz"
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"records", "metadata"}:
                raise ValueError("trace archive must contain records and metadata")
            text = archive["metadata"]
            if text.ndim != 0 or text.dtype.kind != "U":
                raise ValueError("trace metadata must be a scalar JSON string")
            metadata = json.loads(text.item())
            return cls(archive["records"], metadata)

    def save(self, path):
        """Save a new NPZ file with records.npy and JSON in metadata.npy; never overwrite."""
        self._validate()
        text = json.dumps(self.metadata, ensure_ascii=True, allow_nan=False)
        path = Path(path)
        raw = path.open("xb")
        try:
            with raw:
                np.savez(raw, records=self.records, metadata=np.array(text))
        except BaseException:
            path.unlink()
            raise

    def _validate(self):
        m, records = self.metadata, self.records
        if not isinstance(m, dict) or not {*_CAPTURE_FIELDS, "sites"} <= m.keys():
            raise ValueError("missing trace metadata fields")
        if m["clock"] != "globaltimer" or m["timestamp_unit"] != "ns":
            raise ValueError("trace requires globaltimer timestamps in nanoseconds")
        if m["retention"] != "KEEP_LATEST":
            raise ValueError("unsupported trace retention policy")
        for key in ("kernel", "arch", "source_cubin_sha256"):
            if not isinstance(m[key], str) or not m[key]:
                raise ValueError(f"trace {key} must be a nonempty string")
        grid, block, warp_count = geometry(m["grid"], m["block"])
        if m["grid"] != list(grid) or m["block"] != list(block):
            raise ValueError("trace grid/block must be three-element lists")
        depth = depth_checked(m["ring_depth"])
        sites = sites_checked(m["sites"])
        if m["sites"] != list(sites):
            raise ValueError("trace sites must be a sorted list of unique instruction indices")
        if not isinstance(records, np.ndarray) or records.ndim != 1 or records.dtype != RECORD_DTYPE:
            raise ValueError("trace records must be a one-dimensional array with RECORD_DTYPE")
        if len(records):
            if int(records["warp_id"].max()) >= warp_count:
                raise ValueError("record warp_id is outside the launch geometry")
            if not np.isin(records["site_id"], sites).all():
                raise ValueError("record site_id is missing from the recorded sites")
            _, counts = np.unique(records["warp_id"], return_counts=True)
            if int(counts.max()) > depth:
                raise ValueError("retained records exceed ring depth for a warp")
