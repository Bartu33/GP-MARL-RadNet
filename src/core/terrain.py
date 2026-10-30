"""Terrain height model and radar line-of-sight (LOS).

The physics note (Section 5) lists "line-of-sight occlusion by terrain or
other objects" among the visibility checks made before a detection is
attempted.  This module provides a height grid around the scenario site and
a vectorised LOS test along the radar-target segment.

Sources
-------
* ``tiles``      -- real elevation from AWS Terrain Tiles ("terrarium" PNG
                    encoding: h = R*256 + G + B/256 - 32768 metres), fetched
                    once and cached under data/raw/terrain/.
* ``procedural`` -- synthetic ridges and hills, used offline or on request.
* ``flat``       -- no terrain (h = 0 everywhere, LOS only limited by the
                    curvature of the Earth).

Vertical datum
--------------
Simulator heights are relative to the *base* of the area: the median ground
height around the scenario centre.  Target altitudes in the simulator are
measured from that base, radar masts stand on the local ground.

The tile layout (zoom 12, 4 x 4 tiles centred on the site) is the same one
the 3-D viewer uses, and the grid is passed to the viewer, so what is drawn
is exactly what the LOS test uses.
"""
from __future__ import annotations

import io
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "raw" / "terrain"
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ZOOM, N_TILES, TILE_PX = 12, 4, 256
EARTH_RADIUS_M = 6_371_000.0
K_REFRACTION = 4.0 / 3.0            # standard-atmosphere effective Earth radius factor


def _lon2x(lon, z):
    return (lon + 180.0) / 360.0 * 2 ** z


def _lat2y(lat, z):
    r = math.radians(lat)
    return (1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2 * 2 ** z


@dataclass
class Terrain:
    """Height grid H[row, col] in metres relative to the area base.

    Row 0 is the northern edge, column 0 the western edge.  ``x_west`` and
    ``y_north`` are the ENU coordinates (m) of the outer grid edges and
    ``cell`` is the cell size in metres.
    """
    H: np.ndarray
    x_west: float
    y_north: float
    cell: float
    source: str = "flat"
    base_msl: float = 0.0           # absolute height of the datum (display only)

    # ------------------------------------------------------------ queries
    def height(self, x, y):
        """Bilinear ground height at ENU (x, y); vectorised."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if self.source == "flat":
            return np.zeros(np.broadcast(x, y).shape)
        rows, cols = self.H.shape
        c = np.clip((x - self.x_west) / self.cell - 0.5, 0, cols - 1.001)
        r = np.clip((self.y_north - y) / self.cell - 0.5, 0, rows - 1.001)
        c0, r0 = c.astype(int), r.astype(int)
        fc, fr = c - c0, r - r0
        H = self.H
        top = H[r0, c0] * (1 - fc) + H[r0, c0 + 1] * fc
        bot = H[r0 + 1, c0] * (1 - fc) + H[r0 + 1, c0 + 1] * fc
        return top * (1 - fr) + bot * fr

    def line_of_sight(self, p_from, p_to, clearance_m: float = 5.0, step_m: float | None = None) -> bool:
        """True if the straight segment p_from -> p_to clears the terrain.

        The ground along the path is lowered by the Earth-curvature bulge
        d1*d2 / (2 k R) so that long paths behave like a 4/3-Earth model.
        """
        p0 = np.asarray(p_from, float)
        p1 = np.asarray(p_to, float)
        d = np.linalg.norm(p1[:2] - p0[:2])
        if d < 1.0:
            return True
        step = step_m or float(np.clip(self.cell * 0.5, 10.0, 200.0))
        n = int(np.ceil(d / step))
        s = np.linspace(0.0, 1.0, n + 1)[1:-1]
        if len(s) == 0:
            return True
        pts = p0 + s[:, None] * (p1 - p0)
        bulge = (s * d) * ((1 - s) * d) / (2 * K_REFRACTION * EARTH_RADIUS_M)
        ground = self.height(pts[:, 0], pts[:, 1]) + bulge
        return bool(np.all(pts[:, 2] > ground + clearance_m))

    # ------------------------------------------------------------ builders
    @classmethod
    def flat(cls) -> "Terrain":
        return cls(np.zeros((2, 2)), -1e6, 1e6, 1e6, "flat")

    @classmethod
    def procedural(cls, lat: float, lon: float, seed: int = 0, resolution: int = 512) -> "Terrain":
        """Synthetic ridges and hills on the same grid as the tile mosaic."""
        x_west, y_north, size = _grid_extent(lat, lon)
        rng = np.random.default_rng(seed)
        u = (np.arange(resolution) + 0.5) / resolution
        U, V = np.meshgrid(u, u)                       # V = 0 at the northern edge
        h = 600 * np.clip(U - 0.68, 0, None) ** 1.4 * 4          # mountain range to the east
        for _ in range(9):                                        # scattered hills
            cu, cv = rng.uniform(0.15, 0.85, 2)
            amp, wid = rng.uniform(120, 420), rng.uniform(0.03, 0.08)
            h += amp * np.exp(-((U - cu) ** 2 + (V - cv) ** 2) / (2 * wid ** 2))
        h += 15 * np.sin(U * 31 + V * 7) + 10 * np.sin(V * 27 - U * 13)
        return cls._finish(h + 1200.0, x_west, y_north, size / resolution, "procedural")

    @classmethod
    def from_tiles(cls, lat: float, lon: float, resolution: int = 512, timeout_s: float = 8.0) -> "Terrain":
        """Real elevation from AWS Terrain Tiles (raises on any failure)."""
        from PIL import Image

        fx, fy = _lon2x(lon, ZOOM), _lat2y(lat, ZOOM)
        x0, y0 = math.floor(fx - N_TILES / 2), math.floor(fy - N_TILES / 2)
        full = np.zeros((N_TILES * TILE_PX, N_TILES * TILE_PX), np.float32)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for j in range(N_TILES):
            for i in range(N_TILES):
                path = CACHE_DIR / f"{ZOOM}_{x0 + i}_{y0 + j}.png"
                if not path.exists():
                    url = TILE_URL.format(z=ZOOM, x=x0 + i, y=y0 + j)
                    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                        path.write_bytes(resp.read())
                img = np.asarray(Image.open(io.BytesIO(path.read_bytes())).convert("RGB"), np.float32)
                h = img[..., 0] * 256 + img[..., 1] + img[..., 2] / 256 - 32768
                full[j * TILE_PX:(j + 1) * TILE_PX, i * TILE_PX:(i + 1) * TILE_PX] = h
        k = full.shape[0] // resolution
        if k > 1:
            full = full[: resolution * k, : resolution * k].reshape(resolution, k, resolution, k).mean((1, 3))
        x_west, y_north, size = _grid_extent(lat, lon)
        return cls._finish(full, x_west, y_north, size / full.shape[0], "tiles")

    @classmethod
    def _finish(cls, h_msl, x_west, y_north, cell, source) -> "Terrain":
        t = cls(np.asarray(h_msl, float), x_west, y_north, cell, source)
        # datum = median ground height over the central +-7 km
        g = np.linspace(-7000, 7000, 15)
        X, Y = np.meshgrid(g, g)
        base = float(np.median(t.height(X, Y)))
        t.H = t.H - base
        t.base_msl = base
        return t

    def to_payload(self, max_px: int = 512) -> dict:
        """Compact description for the 3-D viewer (heights in metres, int16)."""
        import base64
        H = self.H
        if self.source == "flat":
            return {"source": "flat"}
        if H.shape[0] > max_px:
            k = H.shape[0] // max_px
            H = H[: max_px * k, : max_px * k].reshape(max_px, k, max_px, k).mean((1, 3))
        data = np.round(H).astype("<i2").tobytes()
        return {"source": self.source, "w": H.shape[1], "h": H.shape[0],
                "x_west": self.x_west, "y_north": self.y_north,
                "size": self.cell * self.H.shape[0], "base_msl": self.base_msl,
                "data": base64.b64encode(data).decode("ascii")}


def _grid_extent(lat: float, lon: float):
    """ENU extent (west edge, north edge, size) of the 4x4 zoom-12 tile mosaic
    whose centre tile block contains the site; the site is the ENU origin."""
    fx, fy = _lon2x(lon, ZOOM), _lat2y(lat, ZOOM)
    x0, y0 = math.floor(fx - N_TILES / 2), math.floor(fy - N_TILES / 2)
    tile_m = 40_075_016.0 * math.cos(math.radians(lat)) / 2 ** ZOOM
    return -(fx - x0) * tile_m, (fy - y0) * tile_m, N_TILES * tile_m


def build_terrain(site: dict, spec: dict | str | None, seed: int = 0) -> Terrain:
    """Terrain for a scenario.  spec: "flat" | "procedural" | "tiles" | "auto"
    (auto = tiles, falling back to procedural when offline)."""
    source = spec.get("source", "flat") if isinstance(spec, dict) else (spec or "flat")
    res = spec.get("resolution", 512) if isinstance(spec, dict) else 512
    if source == "flat":
        return Terrain.flat()
    if source == "procedural":
        return Terrain.procedural(site["lat"], site["lon"], seed, res)
    try:
        return Terrain.from_tiles(site["lat"], site["lon"], res)
    except Exception:
        if source == "tiles":
            raise
        return Terrain.procedural(site["lat"], site["lon"], seed, res)
