"""The device record layout and launch geometry shared by capture and compaction."""
from copy import deepcopy
import math

RECORD_BYTES = 16
RECORD_LAYOUT = {"byte_order": "little", "bytes": 16,
                 "fields": [{"name": "tag", "type": "u32", "offset": 0},
                            {"name": "cta_id", "type": "u32", "offset": 4},
                            {"name": "timestamp", "type": "u64", "offset": 8}]}


def sites_checked(sites):
    """Return unique, sorted original instruction indices; zero belongs to INIT."""
    sites = tuple(sites)
    if not sites or any(type(site) is not int or not 0 < site <= 0xffffffff for site in sites):
        raise ValueError("sites must be positive u32 original instruction indices")
    return tuple(sorted(set(sites)))


def depth_checked(depth):
    if type(depth) is not int or depth < 1 or depth & (depth - 1) or depth * RECORD_BYTES > 0xffffffff:
        raise ValueError("ring depth must be a positive power of two with a u32 slot size")
    return depth


def geometry(grid, block):
    """Validate the u32 geometry used by INIT; return (grid, block, warp_count)."""
    def xyz(v):
        v = (v,) if type(v) is int else tuple(v)
        if not 1 <= len(v) <= 3 or any(type(x) is not int or not 1 <= x <= 0xffffffff for x in v):
            raise ValueError("launch dimensions must be one to three positive u32 integers")
        return v + (1,) * (3 - len(v))
    g, b = xyz(grid), xyz(block)
    threads, ctas = math.prod(b), math.prod(g)
    if threads > 1024:
        raise ValueError("block contains more than 1024 threads")
    warps = ctas * ((threads + 31) // 32)
    if ctas > 1 << 32 or warps > 1 << 32:
        raise ValueError("CTA id / global warp index exceeds u32 range")
    return g, b, warps


def cta_id(block_index, grid):
    """Launch-local x + grid.x * (y + grid.y * z), not a hardware slot."""
    g, _, _ = geometry(grid, 1)
    p = tuple(block_index)
    if len(p) != 3 or any(type(x) is not int or not 0 <= x < n for x, n in zip(p, g)):
        raise ValueError("CTA coordinates outside grid")
    return p[0] + g[0] * (p[1] + g[1] * p[2])


def metadata(sites, depth, grid, block):
    """Describe a producer's exact tag map, layout and launch geometry."""
    sites = list(sites)
    if tuple(sites) != sites_checked(sites):
        raise ValueError("sites must be sorted unique positive original instruction indices")
    if len(sites) > 0xffffffff:
        raise ValueError("too many tags")
    g, b, _ = geometry(grid, block)
    return dict(record_layout=deepcopy(RECORD_LAYOUT),
                retention="KEEP_LATEST", ring_depth=depth_checked(depth),
                sites=[dict(tag=t, site=s) for t, s in enumerate(sites, 1)], grid=list(g), block=list(b),
                clock="globaltimer", timestamp_unit="ns")
