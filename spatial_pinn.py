# -*- coding: utf-8 -*-
"""Spatiotemporal reaction--diffusion PINN for 10 m satellite windows.

Inputs are fixed, georeferenced raster windows produced by
``basins/sentinel2_pattern_export_v2.js``.  The model maps ``(x, y, t)`` to
``(P, T, R)`` and trains against raster observations while penalising

    dP/dt = f(P,T,R) + D_P laplacian(P)
    dT/dt = g(P,T,R) + D_T laplacian(T)
    dR/dt = c1 (F-R) + D_R laplacian(R)

with zero-flux boundary conditions.  P is vegetation cover, T is a
temperature/dryness state derived from the raster NDVI--LST feature space, and
R is a water-availability state derived from NDWI.  F is an external forcing
field built from the annual downstream water record and a distance-to-water
proxy.  This keeps R a state variable rather than treating observed NDWI as a
deterministic input.

The script also provides a fully synthetic smoke test.  The smoke test checks
the numerical implementation; it is not a substitute for a georeferenced run
on downloaded Sentinel-2 windows.

Examples
--------
    python spatial_pinn.py --synthetic --epochs 120 --outdir results_spatial_pinn/smoke
    python spatial_pinn.py --stack rasters/W1 --window W1 \
        --train-years 2015-2021 --val-years 2022 --test-years 2023 \
        --epochs 1500 --outdir results_spatial_pinn/W1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Anaconda ships a second OpenMP runtime alongside the CPU PyTorch wheel.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as fn
from scipy import ndimage, optimize
from scipy.stats import ks_2samp


PLT_STYLE = {
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
}
plt.rcParams.update(PLT_STYLE)


# ---------------------------------------------------------------------------
# data containers and raster input
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SpatialStack:
    years: np.ndarray
    P: np.ndarray
    T: np.ndarray
    R: np.ndarray
    F: np.ndarray
    mask: np.ndarray
    x: np.ndarray
    y: np.ndarray
    dx: float
    dy: float
    transform: Sequence[float]
    window: str = "synthetic"
    provenance: Dict[str, object] = dataclasses.field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.P.shape)  # type: ignore[return-value]

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            years=self.years,
            P=self.P,
            T=self.T,
            R=self.R,
            F=self.F,
            mask=self.mask,
            x=self.x,
            y=self.y,
            dx=self.dx,
            dy=self.dy,
            transform=np.asarray(self.transform, dtype=float),
            window=np.asarray(self.window),
            provenance=np.asarray(json.dumps(self.provenance, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path: Path) -> "SpatialStack":
        z = np.load(path, allow_pickle=False)
        provenance = json.loads(str(z["provenance"].item())) if "provenance" in z else {}
        return cls(
            years=z["years"].astype(int),
            P=z["P"].astype(float),
            T=z["T"].astype(float),
            R=z["R"].astype(float),
            F=z["F"].astype(float),
            mask=z["mask"].astype(bool),
            x=z["x"].astype(float),
            y=z["y"].astype(float),
            dx=float(z["dx"]),
            dy=float(z["dy"]),
            transform=tuple(float(v) for v in z["transform"]),
            window=str(z["window"].item()) if "window" in z else "window",
            provenance=provenance,
        )


def _read_geotiff(path: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    with tifffile.TiffFile(path) as tf:
        arr = tf.asarray()
        tags = tf.pages[0].tags
        scale = tags["ModelPixelScaleTag"].value if "ModelPixelScaleTag" in tags else None
        tie = tags["ModelTiepointTag"].value if "ModelTiepointTag" in tags else None
    arr = arr.astype(float)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3 and arr.shape[0] > arr.shape[2]:
        arr = np.moveaxis(arr, -1, 0)
    arr[arr <= -1e30] = np.nan
    if scale is None or tie is None:
        raise ValueError(f"{path} is not a north-up GeoTIFF with ModelPixelScaleTag/TiepointTag")
    dx, dy = float(scale[0]), float(scale[1])
    x0, y0 = float(tie[3]), float(tie[4])
    return arr, {"dx": dx, "dy": dy, "x0": x0, "y0": y0,
                 "transform": [dx, 0.0, x0, 0.0, -dy, y0]}


def _find_window_files(folder: Path, window: Optional[str]) -> Tuple[Optional[str], Dict[int, Path], Optional[Path]]:
    files = sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff")))
    if not files:
        raise FileNotFoundError(f"no GeoTIFF files in {folder}")
    by_year: Dict[int, Path] = {}
    ids = set()
    for f in files:
        # Accept browser-style duplicate suffixes such as W1_2015(1).tif.
        m = re.match(r"^([A-Za-z][A-Za-z0-9_-]*?)_?(\d{4})(?:[_(\-].*)?$", f.stem)
        if m:
            ids.add(m.group(1))
    if window is None:
        if not ids:
            raise ValueError("could not infer a window id; use --window W1")
        window = sorted(ids)[0]
    for f in files:
        m = re.match(rf"^{re.escape(window)}_?(\d{{4}})(?:[_(\-].*)?$", f.stem)
        if m:
            by_year[int(m.group(1))] = f
    mask = next((f for f in files if re.match(rf"^{re.escape(window)}.*MASK.*$", f.stem, re.I)), None)
    if not by_year:
        raise FileNotFoundError(f"no annual {window}_YYYY GeoTIFFs in {folder}")
    return window, by_year, mask


def _fit_tvdi_edges(ndvi: np.ndarray, lst: np.ndarray, mask: np.ndarray,
                    nbins: int = 20) -> Tuple[float, float, float, float]:
    n = ndvi[mask].astype(float)
    t = lst[mask].astype(float)
    ok = np.isfinite(n) & np.isfinite(t)
    n, t = n[ok], t[ok]
    if n.size < 200:
        raise ValueError("too few paired NDVI-LST pixels for TVDI")
    edges = np.linspace(np.nanpercentile(n, 2), np.nanpercentile(n, 98), nbins + 1)
    centres, dry, wet = [], [], []
    for i in range(nbins):
        sel = (n >= edges[i]) & (n < edges[i + 1])
        if sel.sum() < 20:
            continue
        centres.append(0.5 * (edges[i] + edges[i + 1]))
        dry.append(np.nanpercentile(t[sel], 97.5))
        wet.append(np.nanpercentile(t[sel], 2.5))
    if len(centres) < 4:
        raise ValueError("too few NDVI bins for TVDI")
    ad, bd = np.polyfit(centres, dry, 1)
    aw, bw = np.polyfit(centres, wet, 1)
    return float(ad), float(bd), float(aw), float(bw)


def _read_water_series(path: Optional[Path]) -> Dict[int, float]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    ycol = next((c for c in df.columns if str(c).lower() == "year"), None)
    vcol = next((c for c in df.columns if c in {"W_downstream_1e8m3", "R_raw", "water_1e8m3"}), None)
    if ycol is None or vcol is None:
        return {}
    out = {}
    for _, row in df.iterrows():
        try:
            out[int(float(row[ycol]))] = float(row[vcol])
        except (TypeError, ValueError):
            continue
    return out


def build_stack(folder: Path, window: Optional[str], train_years: Sequence[int],
                water_csv: Optional[Path] = None, mask_override: Optional[Path] = None) -> SpatialStack:
    window_id, files, mask_file = _find_window_files(folder, window)
    years = np.array(sorted(files), dtype=int)
    stack, usable_years, skipped, meta0 = [], [], [], None
    for year in years:
        arr, meta = _read_geotiff(files[year])
        if arr.shape[0] < 3:
            raise ValueError(f"{files[year]} must contain NDVI, LST_C and NDWI bands")
        if not (0.5 <= meta["dx"] <= 50.0 and 0.5 <= meta["dy"] <= 50.0):
            raise ValueError(
                f"{files[year]} has pixel size {meta['dx']:.3f} x {meta['dy']:.3f} m; "
                "this is not a 10 m window. The GEE export fell back to a projection-wide grid."
            )
        band_max = np.nanmax(np.abs(arr[:3]), axis=(1, 2))
        if not np.all(band_max > 0):
            skipped.append({"year": int(year), "file": files[year].name,
                            "reason": "one or more gridded bands are empty"})
            print(f"Skipping {year}: {files[year].name} has an empty NDVI/LST/NDWI band")
            continue
        if meta0 is None:
            meta0 = meta
        elif not np.allclose(meta["transform"], meta0["transform"], atol=1e-6):
            raise ValueError(f"grid transform changes at {files[year]}")
        stack.append(arr[:4] if arr.shape[0] >= 4 else arr[:3])
        usable_years.append(int(year))
    cube = np.stack(stack)  # [year, band, y, x]
    years = np.asarray(usable_years, dtype=int)
    if meta0 is None:
        raise ValueError("empty raster stack")
    ndvi, lst, ndwi = cube[:, 0], cube[:, 1], cube[:, 2]
    if cube.shape[1] >= 4:
        valid_band = cube[:, 3] > 0.5
    else:
        valid_band = np.ones_like(ndvi, dtype=bool)
    if mask_file is not None:
        m, _ = _read_geotiff(mask_file)
        mask = np.squeeze(m[0] > 0.5)
    else:
        mask = np.isfinite(ndvi).mean(axis=0) >= 0.5
    train_sel = np.isin(years, train_years)
    if train_sel.sum() < 2:
        train_sel = np.ones_like(years, dtype=bool)
    fit_mask = mask[None, :, :] & valid_band & np.isfinite(ndvi) & np.isfinite(lst)
    fit_mask &= train_sel[:, None, None]
    if fit_mask.sum() < 200:
        raise ValueError("fewer than 200 valid NDVI-LST pixels in the calibration years")
    ad, bd, aw, bw = _fit_tvdi_edges(ndvi, lst, fit_mask)
    dry = ad * ndvi + bd
    wet = aw * ndvi + bw
    span = np.maximum(dry - wet, 0.5)
    tvdi = np.clip((lst - wet) / span, 0.0, 1.0)

    p_lo, p_hi = np.nanpercentile(ndvi[fit_mask], [3, 97])
    r_ok = mask[None, :, :] & valid_band & np.isfinite(ndwi) & train_sel[:, None, None]
    r_lo, r_hi = np.nanpercentile(ndwi[r_ok], [3, 97])
    P = np.clip((ndvi - p_lo) / max(p_hi - p_lo, 1e-6), 0.0, 1.0)
    T = tvdi.astype(float)
    R = np.clip((ndwi - r_lo) / max(r_hi - r_lo, 1e-6), 0.0, 1.0)
    valid = mask[None, :, :] & valid_band & np.isfinite(P) & np.isfinite(T) & np.isfinite(R)
    P = np.where(valid, P, np.nan)
    T = np.where(valid, T, np.nan)
    R = np.where(valid, R, np.nan)

    # Distance-to-water is estimated from repeatedly observed water pixels;
    # it is a spatial availability proxy, not a river-vector ROI.
    water_frac = np.nanmean(np.where(valid, ndwi > 0.0, np.nan), axis=0)
    water = np.isfinite(water_frac) & (water_frac >= 0.1)
    if water.sum() < 8:
        rmean = np.nanmean(np.where(valid, R, np.nan), axis=0)
        threshold = np.nanpercentile(rmean[mask], 95) if mask.any() else 0.5
        water = np.isfinite(rmean) & (rmean >= threshold)
    dist = ndimage.distance_transform_edt(~water, sampling=(meta0["dy"], meta0["dx"]))
    prox = np.exp(-dist / 500.0)
    water_record = _read_water_series(water_csv)
    q = np.zeros(len(years), dtype=float)
    for i, year in enumerate(years):
        if int(year) in water_record:
            q[i] = water_record[int(year)]
        else:
            q[i] = np.nanmedian(R[i][mask])
    q_train = q[train_sel]
    q_lo, q_hi = np.nanmin(q_train), np.nanmax(q_train)
    q = np.clip((q - q_lo) / max(q_hi - q_lo, 1e-6), 0.0, 1.0)
    F = q[:, None, None] * prox[None, :, :]
    F = np.where(mask[None, :, :], F, 0.0)

    x = meta0["x0"] + (np.arange(P.shape[2]) + 0.5) * meta0["dx"]
    y = meta0["y0"] - (np.arange(P.shape[1]) + 0.5) * meta0["dy"]
    provenance = {
        "source": "Sentinel-2/Landsat fixed 10 m window",
        "window": window_id,
        "epsg": "read from GeoTIFF transform (GEE export uses EPSG:32645)",
        "product": "S2_SR_HARMONIZED 2017-2023; S2 TOA 2015-2016",
        "files": {str(y): str(files[int(y)]) for y in years},
        "skipped_years": skipped,
        "P": "NDVI, normalized with calibration-period 3rd/97th percentiles",
        "T": "TVDI from calibration-period NDVI-LST edges",
        "R": "NDWI, normalized with calibration-period 3rd/97th percentiles",
        "F": "annual downstream water record x exp(-distance_to_water/500 m)",
        "tvdi_edges": {"a_dry": ad, "b_dry": bd, "a_wet": aw, "b_wet": bw},
        "ndvi_range": [float(p_lo), float(p_hi)],
        "ndwi_range": [float(r_lo), float(r_hi)],
        "note": "LST is 100 m native and resampled to the 10 m grid; P is the only 10 m-native optical state.",
    }
    return SpatialStack(years, P, T, R, F, mask, x, y, meta0["dx"], meta0["dy"],
                        meta0["transform"], window_id, provenance)


def save_observation_csv(stack: SpatialStack, path: Path) -> None:
    """Write the long-form spatial table used for audit and external plotting."""
    X, Y = np.meshgrid(stack.x, stack.y)
    valid = stack.mask[None, :, :] & np.isfinite(stack.P) & np.isfinite(stack.T) & np.isfinite(stack.R)
    ti, yi, xi = np.nonzero(valid)
    df = pd.DataFrame({
        "Year": stack.years[ti].astype(int),
        "x_m": X[yi, xi],
        "y_m": Y[yi, xi],
        "P": stack.P[ti, yi, xi],
        "T": stack.T[ti, yi, xi],
        "R": stack.R[ti, yi, xi],
        "F": stack.F[ti, yi, xi],
        "window": stack.window,
    })
    df.to_csv(path, index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "window": stack.window,
        "epsg": stack.provenance.get("epsg", ""),
        "dx_m": stack.dx,
        "dy_m": stack.dy,
        "x0": stack.transform[2],
        "y0": stack.transform[5],
        "width": stack.P.shape[2],
        "height": stack.P.shape[1],
        "crs_transform": ",".join(f"{v:.10g}" for v in stack.transform),
    }]).to_csv(path.with_name("grid_metadata.csv"), index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# synthetic smoke test
# ---------------------------------------------------------------------------


def make_synthetic_stack(n: int = 64, years: Sequence[int] = (2015, 2023),
                         seed: int = 21) -> SpatialStack:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    years_arr = np.arange(int(years[0]), int(years[1]) + 1)
    P, T, R = [], [], []
    for k, year in enumerate(years_arr):
        phase = 2.0 * np.pi * k / max(len(years_arr) - 1, 1)
        p = 0.50 + 0.18 * np.sin(2 * np.pi * xx / 13.0 + phase) * np.cos(2 * np.pi * yy / 17.0)
        p += 0.06 * np.sin(2 * np.pi * (xx + yy) / 31.0 - 0.5 * phase)
        t = 0.52 + 0.14 * np.cos(2 * np.pi * (xx - yy) / 19.0 + phase)
        r = 0.50 + 0.16 * np.sin(2 * np.pi * (2 * xx + yy) / 29.0 - phase)
        p += 0.025 * rng.normal(size=(n, n))
        t += 0.020 * rng.normal(size=(n, n))
        r += 0.015 * rng.normal(size=(n, n))
        P.append(np.clip(p, 0.02, 0.98))
        T.append(np.clip(t, 0.02, 0.98))
        R.append(np.clip(r, 0.02, 0.98))
    P, T, R = np.stack(P), np.stack(T), np.stack(R)
    q = np.linspace(0.25, 0.8, len(years_arr))[:, None, None]
    prox = 0.55 + 0.45 * np.exp(-((xx - n * 0.55) ** 2 + (yy - n * 0.45) ** 2) / (2 * (n / 7) ** 2))
    F = np.clip(q * prox[None], 0.0, 1.0)
    mask = np.ones((n, n), dtype=bool)
    x = 500000.0 + (np.arange(n) + 0.5) * 10.0
    y = 4500000.0 - (np.arange(n) + 0.5) * 10.0
    return SpatialStack(years_arr, P, T, R, F, mask, x, y, 10.0, 10.0,
                        [10.0, 0.0, x[0] - 5.0, 0.0, -10.0, y[0] + 5.0],
                        "SYN", {"source": "synthetic smoke test", "note": "not a satellite observation"})


# ---------------------------------------------------------------------------
# PINN model
# ---------------------------------------------------------------------------


class SineLayer(nn.Module):
    """SIREN layer, which fits band-limited spatial fields more reliably."""

    def __init__(self, in_features: int, out_features: int, omega: float = 12.0,
                 first: bool = False):
        super().__init__()
        self.omega = float(omega)
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if first:
                bound = 1.0 / max(in_features, 1)
            else:
                bound = math.sqrt(6.0 / max(in_features, 1)) / self.omega
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(z))


class FourierMLP(nn.Module):
    def __init__(self, width: int = 64, layers: int = 4, n_freq: int = 4,
                 arch: str = "siren"):
        super().__init__()
        self.arch = arch
        if arch == "siren":
            modules: List[nn.Module] = [SineLayer(3, width, first=True)]
            for _ in range(layers - 1):
                modules.append(SineLayer(width, width))
            modules.append(nn.Linear(width, 3))
            with torch.no_grad():
                modules[-1].weight.uniform_(-0.02, 0.02)
                modules[-1].bias.zero_()
            self.net = nn.Sequential(*modules)
            self.register_buffer("freqs", torch.empty(0))
        else:
            self.register_buffer("freqs", torch.tensor([1.0, 2.0, 4.0, 8.0][:n_freq]) * math.pi)
            enc = 3 + 3 * 2 * n_freq
            modules = [nn.Linear(enc, width)]
            for _ in range(layers - 1):
                modules += [nn.SiLU(), nn.Linear(width, width)]
            modules += [nn.SiLU(), nn.Linear(width, 3)]
            self.net = nn.Sequential(*modules)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        if self.arch == "siren":
            return z
        parts = [z]
        for f in self.freqs:
            parts.append(torch.sin(z * f))
            parts.append(torch.cos(z * f))
        return torch.cat(parts, dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(self.encode(z)))


class RDPhysics(nn.Module):
    """Positive reaction rates and diffusion coefficients."""

    def __init__(self, init: Optional[Dict[str, float]] = None, freeze_reaction: bool = False):
        super().__init__()
        defaults = {
            "a1": 4.443, "a2": 2.213, "a3": 1e-4, "a4": 2.200,
            "b1": 0.0182, "b2": 1e-4, "b3": 0.0344, "c1": 0.482,
            "DP": 10.0, "DT": 1000.0, "DR": 100.0,
        }
        if init:
            defaults.update({k: float(v) for k, v in init.items() if k in defaults and float(v) > 0})
        self.freeze_reaction = bool(freeze_reaction)
        self.init_values = defaults
        for key in ("a1", "a2", "a3", "a4", "b1", "b2", "b3", "c1", "DP", "DT", "DR"):
            val = torch.tensor(math.log(defaults[key]), dtype=torch.float32)
            p = nn.Parameter(val, requires_grad=not (freeze_reaction and key[0] in "abc"))
            self.register_parameter("raw_" + key, p)

    def _pos(self, key: str) -> torch.Tensor:
        return fn.softplus(getattr(self, "raw_" + key)) + 1e-8

    def rates(self) -> Dict[str, torch.Tensor]:
        return {k: self._pos(k) for k in ("a1", "a2", "a3", "a4", "b1", "b2", "b3", "c1")}

    def diffusion(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dp = self._pos("DP")
        # Make the vegetation state the low-diffusivity component by construction.
        dt = dp + self._pos("DT")
        dr = dp + self._pos("DR")
        return dp, dt, dr

    def prior(self) -> torch.Tensor:
        loss = 0.0
        for key, val in self.init_values.items():
            loss = loss + (getattr(self, "raw_" + key) - math.log(val)) ** 2
        return loss / len(self.init_values)

    def numpy_parameters(self) -> Dict[str, float]:
        out = {k: float(v.detach().cpu()) for k, v in self.rates().items()}
        dp, dt, dr = self.diffusion()
        out.update(DP=float(dp.detach().cpu()), DT=float(dt.detach().cpu()), DR=float(dr.detach().cpu()))
        return out


class SpatioTemporalPINN(nn.Module):
    def __init__(self, width: int = 64, layers: int = 4, n_freq: int = 4,
                 init: Optional[Dict[str, float]] = None, freeze_reaction: bool = False,
                 arch: str = "siren", anchor_mode: bool = False,
                 residual_scale: float = 0.2, anchor_year: float = 0.0,
                 t_center: float = 0.0, t_half: float = 1.0,
                 time_span: float = 1.0):
        super().__init__()
        self.net = FourierMLP(width, layers, n_freq, arch)
        self.physics = RDPhysics(init, freeze_reaction)
        self.anchor_mode = bool(anchor_mode)
        self.residual_scale = float(residual_scale)
        self.anchor_year = float(anchor_year)
        self.t_center = float(t_center)
        self.t_half = max(float(t_half), 1e-6)
        self.time_span = max(float(time_span), 1.0)

    def forward(self, z: torch.Tensor, anchor: Optional[torch.Tensor] = None) -> torch.Tensor:
        base = self.net(z)
        if not self.anchor_mode:
            return base
        if anchor is None:
            raise ValueError("anchor values are required in persistence-anchor mode")
        physical_t = z[:, 2:3] * self.t_half + self.t_center
        dt = (physical_t - self.anchor_year) / self.time_span
        residual = self.residual_scale * (2.0 * base - 1.0) * dt
        return torch.clamp(anchor + residual, 0.0, 1.0)


def _grad(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, z, grad_outputs=torch.ones_like(y),
                               create_graph=True, retain_graph=True)[0]


def pde_residual(model: SpatioTemporalPINN, z: torch.Tensor,
                 scales: Dict[str, float], forcing: torch.Tensor,
                 anchor: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
    z = z.clone().detach().requires_grad_(True)
    out = model(z, anchor)
    P, T, R = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    gP, gT, gR = _grad(P, z), _grad(T, z), _grad(R, z)
    laplace = []
    for g in (gP, gT, gR):
        gxx = _grad(g[:, 0:1], z)[:, 0:1] / (scales["x"] ** 2)
        gyy = _grad(g[:, 1:2], z)[:, 1:2] / (scales["y"] ** 2)
        laplace.append(gxx + gyy)
    dPdt = gP[:, 2:3] / scales["t"]
    dTdt = gT[:, 2:3] / scales["t"]
    dRdt = gR[:, 2:3] / scales["t"]
    p = model.physics.rates()
    dp, dt, dr = model.physics.diffusion()
    fp = p["a1"] * P * (1 - P) * (1 - T) + p["a2"] * R - p["a3"] * T * P - p["a4"] * P
    ft = p["b1"] * (1 - P) - p["b2"] * R - p["b3"] * T
    fr = p["c1"] * (forcing - R)
    return dPdt - (fp + dp * laplace[0]), dTdt - (ft + dt * laplace[1]), dRdt - (fr + dr * laplace[2])


def _normal_derivative(model: SpatioTemporalPINN, z: torch.Tensor, axis: int,
                       anchor: Optional[torch.Tensor] = None) -> torch.Tensor:
    z = z.clone().detach().requires_grad_(True)
    out = model(z, anchor)
    grads = [_grad(out[:, i:i + 1], z) for i in range(3)]
    return torch.cat([g[:, axis:axis + 1] for g in grads], dim=1)


# ---------------------------------------------------------------------------
# sampling and training
# ---------------------------------------------------------------------------


class StackSampler:
    def __init__(self, stack: SpatialStack, train_years: Sequence[int], device: str):
        self.stack = stack
        self.device = device
        self.year_index = {int(y): i for i, y in enumerate(stack.years)}
        self.train_idx = np.array([self.year_index[int(y)] for y in train_years if int(y) in self.year_index], dtype=int)
        if len(self.train_idx) < 2:
            raise ValueError("need at least two training years")
        self.tmin = float(np.min(stack.years[self.train_idx]))
        self.tmax = float(np.max(stack.years[self.train_idx]))
        self.anchor_year = int(self.tmax)
        self.xmin, self.xmax = float(stack.x.min()), float(stack.x.max())
        self.ymin, self.ymax = float(stack.y.min()), float(stack.y.max())
        self.scales = {
            "x": max((self.xmax - self.xmin) / 2.0, stack.dx),
            "y": max((self.ymax - self.ymin) / 2.0, stack.dy),
            "t": max((self.tmax - self.tmin) / 2.0, 0.5),
        }
        self.mask = stack.mask
        anchor_values = [getattr(stack, name)[self.year_index[self.anchor_year]]
                         for name in ("P", "T", "R")]
        self.anchor_channels = np.stack(anchor_values, axis=0)
        self.anchor_flat = np.stack(anchor_values, axis=-1).reshape(-1, 3)
        self._prepare_arrays()

    def norm(self, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        xn = (x - (self.xmin + self.xmax) / 2.0) / self.scales["x"]
        yn = (y - (self.ymin + self.ymax) / 2.0) / self.scales["y"]
        tn = (t - (self.tmin + self.tmax) / 2.0) / self.scales["t"]
        return np.stack([xn, yn, tn], axis=-1)

    def _prepare_arrays(self) -> None:
        X, Y = np.meshgrid(self.stack.x, self.stack.y)
        self.flat = {}
        for name in ("P", "T", "R", "F"):
            arr = getattr(self.stack, name)[self.train_idx]
            self.flat[name] = arr.reshape(len(self.train_idx), -1)
        observed = np.stack([self.stack.mask] * len(self.train_idx))
        for name in ("P", "T", "R"):
            observed &= np.isfinite(getattr(self.stack, name)[self.train_idx])
        self.obs_mask = observed.reshape(len(self.train_idx), -1)
        if not np.any(self.obs_mask):
            raise ValueError("no finite P/T/R observations in the requested training years")
        self.Xf = X.ravel()
        self.Yf = Y.ravel()
        self.tf = self.stack.years[self.train_idx].astype(float)
        self.valid_rows = np.argwhere(self.obs_mask)

    def anchor_at_xy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ix = (np.asarray(x, dtype=float) - self.stack.x[0]) / self.stack.dx
        iy = (self.stack.y[0] - np.asarray(y, dtype=float)) / self.stack.dy
        coords = np.vstack([iy.ravel(), ix.ravel()])
        vals = [ndimage.map_coordinates(ch, coords, order=1, mode="nearest").reshape(np.shape(x))
                for ch in self.anchor_channels]
        return np.stack(vals, axis=-1)

    def obs_batch(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.valid_rows[np.random.randint(0, len(self.valid_rows), size=n)]
        yi, pi = idx[:, 0], idx[:, 1]
        z = self.norm(self.Xf[pi], self.Yf[pi], self.tf[yi])
        target = np.stack([self.flat[k][yi, pi] for k in ("P", "T", "R")], axis=-1)
        anchor = self.anchor_flat[pi]
        return (torch.tensor(z, dtype=torch.float32, device=self.device),
                torch.tensor(target, dtype=torch.float32, device=self.device),
                torch.tensor(anchor, dtype=torch.float32, device=self.device))

    def collocation_batch(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.valid_rows[np.random.randint(0, len(self.valid_rows), size=n)]
        yi, pi = idx[:, 0], idx[:, 1]
        z = self.norm(self.Xf[pi], self.Yf[pi], self.tf[yi])
        forcing = self.flat["F"][yi, pi][:, None]
        anchor = self.anchor_flat[pi]
        return (torch.tensor(z, dtype=torch.float32, device=self.device),
                torch.tensor(forcing, dtype=torch.float32, device=self.device),
                torch.tensor(anchor, dtype=torch.float32, device=self.device))

    def persistence_batch(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random space--time points constrained toward the anchor-year field."""
        pi = np.random.choice(np.flatnonzero(self.mask), size=n, replace=True)
        t_hi = float(np.max(self.stack.years))
        t = np.random.uniform(self.tmin, t_hi, size=n)
        z = self.norm(self.Xf[pi], self.Yf[pi], t)
        anchor = self.anchor_flat[pi]
        return (torch.tensor(z, dtype=torch.float32, device=self.device),
                torch.tensor(anchor, dtype=torch.float32, device=self.device))

    def boundary_batch(self, n: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        t = np.random.uniform(self.tmin, self.tmax, size=n)
        xl, xr = np.full(n, self.xmin), np.full(n, self.xmax)
        yb, yt = np.full(n, self.ymin), np.full(n, self.ymax)
        yr = np.random.uniform(self.ymin, self.ymax, size=n)
        xrnd = np.random.uniform(self.xmin, self.xmax, size=n)
        pairs = [(xl, yr), (xr, yr), (xrnd, yb), (xrnd, yt)]
        out = []
        for x, y in pairs:
            out.append((torch.tensor(self.norm(x, y, t), dtype=torch.float32, device=self.device),
                        torch.tensor(self.anchor_at_xy(x, y), dtype=torch.float32, device=self.device)))
        return out

    def initial_batch(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = self.train_idx[0]
        pi = np.flatnonzero(
            self.mask
            & np.isfinite(self.stack.P[i])
            & np.isfinite(self.stack.T[i])
            & np.isfinite(self.stack.R[i])
        )
        if len(pi) > n:
            pi = np.random.choice(pi, n, replace=False)
        z = self.norm(self.Xf[pi], self.Yf[pi], np.full(len(pi), self.stack.years[i]))
        target = np.stack([self.stack.P[i].ravel()[pi], self.stack.T[i].ravel()[pi], self.stack.R[i].ravel()[pi]], axis=-1)
        return (torch.tensor(z, dtype=torch.float32, device=self.device),
                torch.tensor(target, dtype=torch.float32, device=self.device),
                torch.tensor(self.anchor_flat[pi], dtype=torch.float32, device=self.device))


def train_model(stack: SpatialStack, train_years: Sequence[int], args: argparse.Namespace,
                outdir: Path) -> Tuple[SpatioTemporalPINN, StackSampler, pd.DataFrame]:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sampler = StackSampler(stack, train_years, device)
    init = {}
    if args.init_params and Path(args.init_params).exists():
        df = pd.read_csv(args.init_params)
        if len(df):
            row = df.iloc[0]
            init = {k: float(row[k]) for k in ("a1", "a2", "a3", "a4", "b1", "b2", "b3", "c1") if k in row}
    model = SpatioTemporalPINN(
        args.width, args.layers, args.n_freq, init, args.freeze_reaction, args.arch,
        anchor_mode=args.persistence_anchor,
        residual_scale=args.residual_scale,
        anchor_year=sampler.anchor_year,
        t_center=(sampler.tmin + sampler.tmax) / 2.0,
        t_half=(sampler.tmax - sampler.tmin) / 2.0,
        time_span=max(sampler.tmax - sampler.tmin, 1.0),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        zo, yo, ao = sampler.obs_batch(args.batch_obs)
        zc, fc, ac = sampler.collocation_batch(args.batch_col)
        rp, rt, rr = pde_residual(model, zc, sampler.scales, fc, ac)
        pde_loss = (rp.square().mean() + rt.square().mean() + rr.square().mean()) / 3.0
        data_loss = fn.mse_loss(model(zo, ao), yo)
        bc_loss = 0.0
        for side, (zb, ab) in enumerate(sampler.boundary_batch(args.batch_bc)):
            axis = 0 if side < 2 else 1
            bc_loss = bc_loss + _normal_derivative(model, zb, axis, ab).square().mean()
        bc_loss = bc_loss / 4.0
        zi, yi, ai = sampler.initial_batch(args.batch_ic)
        ic_loss = fn.mse_loss(model(zi, ai), yi)
        if args.persistence_weight > 0:
            za, ya = sampler.persistence_batch(args.batch_persist)
            persistence_loss = fn.mse_loss(model(za, ya), ya)
        else:
            persistence_loss = torch.zeros((), dtype=torch.float32, device=device)
        prior = model.physics.prior()
        loss = (args.data_weight * data_loss + args.pde_weight * pde_loss
                + args.bc_weight * bc_loss + args.ic_weight * ic_loss
                + args.prior_weight * prior
                + args.persistence_weight * persistence_loss)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        row = {"epoch": epoch, "loss": float(loss.detach().cpu()),
               "data": float(data_loss.detach().cpu()), "pde": float(pde_loss.detach().cpu()),
               "boundary": float(bc_loss.detach().cpu()), "initial": float(ic_loss.detach().cpu()),
               "prior": float(prior.detach().cpu()),
               "persistence": float(persistence_loss.detach().cpu())}
        history.append(row)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:5d} loss={row['loss']:.5e} data={row['data']:.3e} "
                  f"pde={row['pde']:.3e} ic={row['initial']:.3e} "
                  f"persist={row['persistence']:.3e}")
    hist = pd.DataFrame(history)
    hist.to_csv(outdir / "loss_history.csv", index=False)
    return model, sampler, hist


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def predict_year(model: SpatioTemporalPINN, sampler: StackSampler, year: int,
                 batch: int = 32768) -> np.ndarray:
    model.eval()
    h, w = sampler.stack.P.shape[1:]
    X, Y = np.meshgrid(sampler.stack.x, sampler.stack.y)
    z = sampler.norm(X.ravel(), Y.ravel(), np.full(X.size, float(year)))
    out = []
    anchor = sampler.anchor_flat
    with torch.no_grad():
        for i in range(0, len(z), batch):
            zz = torch.tensor(z[i:i + batch], dtype=torch.float32, device=sampler.device)
            aa = torch.tensor(anchor[i:i + batch], dtype=torch.float32, device=sampler.device)
            out.append(model(zz, aa).cpu().numpy())
    return np.concatenate(out).reshape(h, w, 3)


def _metric(obs: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    ok = mask & np.isfinite(obs) & np.isfinite(pred)
    if ok.sum() < 3:
        return {"n": int(ok.sum()), "rmse": np.nan, "mae": np.nan, "r2": np.nan, "bias": np.nan}
    e = pred[ok] - obs[ok]
    ss = float(np.sum(e ** 2))
    st = float(np.sum((obs[ok] - np.mean(obs[ok])) ** 2))
    return {"n": int(ok.sum()), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))), "r2": float(1 - ss / st) if st > 0 else np.nan,
            "bias": float(np.mean(e))}


def pattern_residual(field: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    z = np.where(mask & np.isfinite(field), field, np.nan)
    fill = np.where(np.isfinite(z), z, np.nanmean(z))
    low = ndimage.gaussian_filter(fill, sigma)
    res = z - low
    sd = np.nanstd(res)
    return res / max(float(sd), 1e-12)


def radial_spectrum(field: np.ndarray, dx: float, nbins: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    z = np.nan_to_num(field, nan=0.0)
    z = z - z.mean()
    ny, nx = z.shape
    F = np.abs(np.fft.fftshift(np.fft.fft2(z))) ** 2
    kx = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(ny, d=dx))
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    KR = np.sqrt(KX ** 2 + KY ** 2)
    bins = np.linspace(0.0, float(KR.max()), nbins)
    ids = np.clip(np.digitize(KR.ravel(), bins) - 1, 0, nbins - 1)
    spec = np.array([F.ravel()[ids == i].mean() if np.any(ids == i) else np.nan for i in range(nbins)])
    return bins, spec


def spectral_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.nan_to_num(a / (np.nansum(a) + 1e-30))
    bb = np.nan_to_num(b / (np.nansum(b) + 1e-30))
    return float(np.sum(aa * bb) / np.sqrt(np.sum(aa ** 2) * np.sum(bb ** 2) + 1e-30))


def peak_wavelength(k: np.ndarray, spec: np.ndarray) -> float:
    ok = np.isfinite(spec) & (k > 0)
    if not ok.any():
        return np.nan
    return float(2 * np.pi / k[ok][np.nanargmax(spec[ok])])


def random_phase_null(field: np.ndarray, seed: int = 9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(np.nan_to_num(field, nan=0.0))
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, F.shape))
    return np.real(np.fft.ifft2(np.abs(F) * ph))


def pattern_metrics(obs: np.ndarray, pred: np.ndarray, mask: np.ndarray,
                    dx: float) -> Dict[str, float]:
    sig = max(3.0, min(obs.shape) / 8.0)
    ro = pattern_residual(obs, mask, sig)
    rp = pattern_residual(pred, mask, sig)
    k, so = radial_spectrum(ro, dx)
    _, sp = radial_spectrum(rp, dx)
    null = random_phase_null(rp)
    labels_o, no = ndimage.label((ro > 0) & mask)
    labels_p, np_ = ndimage.label((rp > 0) & mask)
    so_sizes = np.bincount(labels_o.ravel())[1:] if no else np.array([0])
    sp_sizes = np.bincount(labels_p.ravel())[1:] if np_ else np.array([0])
    return {
        "spectral_cosine": spectral_cosine(so, sp),
        "pixel_correlation": float(np.corrcoef(ro[mask], rp[mask])[0, 1]) if mask.sum() > 2 else np.nan,
        "null_pixel_correlation": float(np.corrcoef(ro[mask], null[mask])[0, 1]) if mask.sum() > 2 else np.nan,
        "peak_wavelength_m": peak_wavelength(k, sp),
        "observed_peak_wavelength_m": peak_wavelength(k, so),
        "mean_patch_px": float(sp_sizes.mean()),
        "observed_mean_patch_px": float(so_sizes.mean()),
        "ks": float(ks_2samp(rp[mask], ro[mask]).statistic),
        "moran_pred": float(ndimage.gaussian_filter(rp, 1).std() / (rp.std() + 1e-12)),
        "moran_obs": float(ndimage.gaussian_filter(ro, 1).std() / (ro.std() + 1e-12)),
    }


def evaluate(stack: SpatialStack, model: SpatioTemporalPINN, sampler: StackSampler,
             test_years: Sequence[int], outdir: Path, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, preds = [], {}
    last_train = int(np.max(sampler.tf))
    i_last = int(np.argmax(stack.years == last_train))
    train_sel = np.isin(stack.years, sampler.tf)
    for year in test_years:
        if int(year) not in set(stack.years.astype(int)):
            continue
        i = int(np.argmax(stack.years == int(year)))
        pred = predict_year(model, sampler, int(year))
        preds[int(year)] = pred
        for j, name in enumerate(("P", "T", "R")):
            obs = getattr(stack, name)[i]
            mask = stack.mask & np.isfinite(obs)
            m = _metric(obs, pred[:, :, j], mask)
            rows.append({"year": int(year), "variable": name, "model": "PINN", **m})
            for bname, base in (("persistence", getattr(stack, name)[i_last]),
                                ("climatology", np.nanmean(getattr(stack, name)[train_sel], axis=0))):
                mb = _metric(obs, base, mask)
                rows.append({"year": int(year), "variable": name, "model": bname, **mb})
            pm = pattern_metrics(obs, pred[:, :, j], mask, stack.dx)
            rows.append({"year": int(year), "variable": name, "model": "PINN_pattern", **pm})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "spatial_pinn_metrics.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(outdir / "spatial_pinn_fields.npz",
                        years=np.array(sorted(preds)),
                        predictions=np.stack([preds[y] for y in sorted(preds)]),
                        observed=np.stack([np.stack([getattr(stack, n)[int(np.argmax(stack.years == y))] for n in ("P", "T", "R")], axis=-1) for y in sorted(preds)]),
                        mask=stack.mask)
    plot_evaluation(stack, preds, metrics, outdir)
    return metrics, pd.DataFrame()


def plot_evaluation(stack: SpatialStack, preds: Dict[int, np.ndarray], metrics: pd.DataFrame,
                    outdir: Path) -> None:
    if not preds:
        return
    year = sorted(preds)[0]
    i = int(np.argmax(stack.years == year))
    names = ("P", "T", "R")
    fig, axes = plt.subplots(3, 4, figsize=(13.2, 8.8), squeeze=False)
    extent = [0, stack.P.shape[2] * stack.dx / 1000, 0, stack.P.shape[1] * stack.dy / 1000]
    for row, name in enumerate(names):
        obs = getattr(stack, name)[i]
        pred = preds[year][:, :, row]
        res = np.where(stack.mask, pred - obs, np.nan)
        vmin, vmax = np.nanpercentile(np.concatenate([obs[stack.mask], pred[stack.mask]]), [2, 98])
        for col, (data, title, cmap, lim) in enumerate([
            (obs, f"{name} observed {year}", "viridis", (vmin, vmax)),
            (pred, f"{name} PINN {year}", "viridis", (vmin, vmax)),
            (res, f"{name} residual", "RdBu_r", None),
        ]):
            ax = axes[row, col]
            if lim is None:
                lim = np.nanpercentile(np.abs(data[stack.mask]), [98])
                im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, vmin=-lim[0], vmax=lim[0])
            else:
                im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax.set_title(title)
            ax.set_xlabel("km")
            ax.set_ylabel("km")
            fig.colorbar(im, ax=ax, shrink=0.75)
        ax = axes[row, 3]
        ro = pattern_residual(obs, stack.mask, max(3.0, min(obs.shape) / 8.0))
        rp = pattern_residual(pred, stack.mask, max(3.0, min(pred.shape) / 8.0))
        k, so = radial_spectrum(ro, stack.dx)
        _, sp = radial_spectrum(rp, stack.dx)
        ax.semilogy(k[1:], so[1:] / (np.nansum(so) + 1e-30), label="observed")
        ax.semilogy(k[1:], sp[1:] / (np.nansum(sp) + 1e-30), label="PINN")
        ax.set_xlabel("radial wavenumber (1/m)")
        ax.set_ylabel("normalized power")
        ax.set_title(f"{name} radial spectrum")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "fig_spatiotemporal_pinn.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2))
    for ax, name in zip(axes, names):
        sub = metrics[(metrics["variable"] == name) & (metrics["model"] == "PINN_pattern")]
        if len(sub):
            ax.bar(["spectral\ncosine", "pixel\ncorrelation"],
                   [float(sub.iloc[0]["spectral_cosine"]), float(sub.iloc[0]["pixel_correlation"])],
                   color="#2c7fb8")
        ax.set_title(name)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "fig_pattern_validation.png", bbox_inches="tight")
    plt.close(fig)


def _jacobian_three_state(p: Dict[str, float], P: float, T: float) -> np.ndarray:
    a1, a2, a3, a4 = p["a1"], p["a2"], p["a3"], p["a4"]
    b1, b2, b3, c1 = p["b1"], p["b2"], p["b3"], p["c1"]
    return np.array([
        [a1 * (1 - 2 * P) * (1 - T) - a3 * T - a4, -a1 * P * (1 - P) - a3 * P, a2],
        [-b1, -b3, -b2],
        [0.0, 0.0, -c1],
    ])


def linear_turing_check(parameters: Dict[str, float], stack: SpatialStack) -> Dict[str, float]:
    P = float(np.nanmean(np.where(stack.mask, stack.P, np.nan)))
    T = float(np.nanmean(np.where(stack.mask, stack.T, np.nan)))
    J = _jacobian_three_state(parameters, P, T)
    D = np.diag([parameters["DP"], parameters["DT"], parameters["DR"]])
    ks = np.linspace(0, 2 * np.pi / max(stack.dx, 1e-6), 1000)
    growth = np.array([np.max(np.linalg.eigvals(J - D * (k * k)).real) for k in ks])
    i = int(np.argmax(growth))
    return {"P_mean": P, "T_mean": T, "growth_peak": float(growth[i]),
            "k_peak": float(ks[i]), "wavelength_m": float(2 * np.pi / max(ks[i], 1e-9)),
            "homogeneous_growth": float(growth[0]), "turing_unstable": bool(growth[i] > 1e-7 and growth[0] < 0)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _years(text: str) -> List[int]:
    m = re.fullmatch(r"(\d{4})\s*-\s*(\d{4})", text.strip())
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", type=str, default="")
    ap.add_argument("--window", type=str, default="")
    ap.add_argument("--water", type=str, default="data/tarim_water_official_2000_2023.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--outdir", type=str, default="results_spatial_pinn")
    ap.add_argument("--train-years", type=str, default="2015-2021")
    ap.add_argument("--val-years", type=str, default="2022")
    ap.add_argument("--test-years", type=str, default="2023")
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--n-freq", type=int, default=4)
    ap.add_argument("--arch", choices=["siren", "fourier"], default="siren")
    ap.add_argument("--batch-obs", type=int, default=4096)
    ap.add_argument("--batch-col", type=int, default=2048)
    ap.add_argument("--batch-bc", type=int, default=512)
    ap.add_argument("--batch-ic", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-weight", type=float, default=1.0)
    ap.add_argument("--pde-weight", type=float, default=0.15)
    ap.add_argument("--bc-weight", type=float, default=0.01)
    ap.add_argument("--ic-weight", type=float, default=0.20)
    ap.add_argument("--prior-weight", type=float, default=1e-5)
    ap.add_argument("--persistence-weight", type=float, default=0.0,
                    help="Anchor predictions toward the last calibration year")
    ap.add_argument("--batch-persist", type=int, default=2048)
    ap.add_argument("--persistence-anchor", action="store_true",
                    help="Use the last calibration year as an exact residual anchor")
    ap.add_argument("--residual-scale", type=float, default=0.2,
                    help="Maximum residual amplitude in the persistence-anchored model")
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--freeze-reaction", action="store_true")
    ap.add_argument("--init-params", type=str, default="")
    ap.add_argument("--preprocess-only", action="store_true")
    ap.add_argument("--synthetic-size", type=int, default=64)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_years = _years(args.train_years)
    val_years = _years(args.val_years)
    test_years = _years(args.test_years)
    if args.synthetic:
        stack = make_synthetic_stack(args.synthetic_size, (2015, 2023), args.seed)
        stack.save(outdir / "spatial_stack.npz")
        save_observation_csv(stack, outdir / "spatial_observations.csv")
        print("Synthetic smoke-test stack: this is not a field-data result.")
    else:
        if not args.stack:
            ap.error("--stack is required unless --synthetic is used")
        stack = build_stack(Path(args.stack), args.window or None, train_years,
                            Path(args.water) if args.water else None)
        stack.save(outdir / "spatial_stack.npz")
        save_observation_csv(stack, outdir / "spatial_observations.csv")
        (outdir / "provenance.json").write_text(json.dumps(stack.provenance, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Loaded {stack.window}: {stack.shape[0]} years, {stack.shape[2]}x{stack.shape[1]} pixels, {stack.dx:g} m")
    if args.preprocess_only:
        return

    model, sampler, hist = train_model(stack, train_years, args, outdir)
    torch.save({"model": model.state_dict(), "args": vars(args), "scales": sampler.scales},
               outdir / "spatial_pinn.pt")
    metrics, _ = evaluate(stack, model, sampler, val_years + test_years, outdir, args.seed)
    params = model.physics.numpy_parameters()
    turing = linear_turing_check(params, stack)
    pd.DataFrame([{**params, **turing}]).to_csv(outdir / "inverted_parameters.csv", index=False)
    (outdir / "spatial_pinn_summary.json").write_text(
        json.dumps({"parameters": params, "linear_stability": turing,
                    "persistence_anchor_year": sampler.anchor_year,
                    "persistence_weight": args.persistence_weight,
                    "persistence_anchor": bool(args.persistence_anchor),
                    "residual_scale": args.residual_scale}, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([{"epoch": int(hist.iloc[-1]["epoch"]), "loss": float(hist.iloc[-1]["loss"]),
                   "data": float(hist.iloc[-1]["data"]), "pde": float(hist.iloc[-1]["pde"])}]).to_csv(outdir / "training_summary.csv", index=False)
    print("\nInverted parameters and linear compatibility check:")
    print(json.dumps({"parameters": params, "linear_stability": turing}, indent=2))
    print(f"Results written to {outdir}")


if __name__ == "__main__":
    main()
