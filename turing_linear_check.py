# -*- coding: utf-8 -*-
"""Turing compatibility figure: dispersion, growth-rate check, mode, spectrum."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figure"
OUT = ROOT / "results_v4"

N = 128
DX = 1.0
DT = 0.05
T_END = 1600.0

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300,
})


def jacobian(p):
    u, v = p["u_eq"], p["v_eq"]
    r, a, c, q = p["r"], p["a"], p["c"], p["q"]
    return np.array([[r * u * v * (1.0 + a - 2.0 * u),
                      r * u * (u - a) * (1.0 - u)],
                     [-c * v, -(q + c * u)]]), (u, v)


def growth(J, du, dv, k):
    return np.array([np.max(np.linalg.eigvals(J - np.diag([du * kk * kk,
                                                           dv * kk * kk])).real)
                     for kk in k])


def classical(J, du, dv):
    fu, fv = J[0]
    gu, gv = J[1]
    tr, det = fu + gv, fu * gv - fv * gu
    ok_band = dv * fu + du * gv
    return (tr < 0) and (det > 0) and (ok_band > 0) and \
           (ok_band ** 2 - 4 * du * dv * det > 0)


def best_pair(J, k):
    out = None
    for du in np.logspace(-6, -2, 21):
        for dv in np.logspace(0, 3, 31):
            if not classical(J, du, dv):
                continue
            g = growth(J, du, dv, k)
            i = int(np.argmax(g))
            if g[i] <= 0:
                continue
            if out is None or g[i] > out[0]:
                out = (float(g[i]), float(du), float(dv), float(k[i]))
    return out


def main():
    row = pd.read_csv(OUT / "turing_summary.csv").iloc[0]
    p = {key: float(row[key]) for key in ("u_eq", "v_eq", "a", "m", "s", "r", "q", "c")}
    J, (u_eq, v_eq) = jacobian(p)

    k_probe = np.linspace(0.0, 60.0, 3001)
    g_peak, du, dv, k_star = best_pair(J, k_probe)
    k_axis = np.linspace(0.0, 60.0, 6001)
    sigma = growth(J, du, dv, k_axis)
    band = k_axis[sigma > 0]
    lam = 2.0 * np.pi / k_star

    # unstable eigenvector at k*
    M = J - np.diag([du * k_star ** 2, dv * k_star ** 2])
    w, V = np.linalg.eig(M)
    iv = int(np.argmax(w.real))
    vec = np.real(V[:, iv])
    vec = vec / np.abs(vec).max()

    X = np.arange(N) * DX
    u = u_eq + 1e-4 * vec[0] * np.cos(k_star * X)[None, :]
    v = v_eq + 1e-4 * vec[1] * np.cos(k_star * X)[None, :]
    u = np.repeat(u, N, axis=0)
    v = np.repeat(v, N, axis=0)

    kpos = 2.0 * np.pi * np.fft.fftfreq(N, d=DX)
    k2 = kpos[:, None] ** 2 + kpos[None, :] ** 2
    eu = np.exp(-du * k2 * DT)
    ev = np.exp(-dv * k2 * DT)
    r_, a_, m_, s_, q_, c_ = (p["r"], p["a"], p["m"], p["s"], p["q"], p["c"])

    i_star = int(np.argmin(np.abs(kpos - k_star)))
    times, amp = [], []
    steps = int(round(T_END / DT))
    for n in range(steps + 1):
        if n % 20 == 0:
            times.append(n * DT)
            amp.append(abs(np.fft.fft2(u - u_eq)[0, i_star]) / N ** 2)
        if n == steps:
            break
        u = u + DT * (r_ * u * (u - a_) * (1.0 - u) * v - m_ * u)
        v = v + DT * (s_ - q_ * v - c_ * u * v)
        u = np.real(np.fft.ifft2(np.fft.fft2(u) * eu))
        v = np.real(np.fft.ifft2(np.fft.fft2(v) * ev))

    times = np.asarray(times)
    amp = np.asarray(amp)
    sel = times > 0
    slope = float(np.polyfit(times[sel], np.log(amp[sel]), 1)[0])

    field = u - u_eq
    F = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    kxs = np.fft.fftshift(kpos)
    KX, KY = np.meshgrid(kxs, kxs, indexing="ij")
    KR = np.sqrt(KX ** 2 + KY ** 2)
    bins = np.linspace(0.0, 4.0, 200)
    ids = np.clip(np.digitize(KR.ravel(), bins) - 1, 0, len(bins) - 1)
    spec = np.array([F.ravel()[ids == i].mean() if np.any(ids == i) else np.nan
                     for i in range(len(bins))])
    k_meas = float(bins[int(np.nanargmax(spec))])

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 6.6))

    ax = axes[0, 0]
    ax.plot(k_axis, sigma, color="#1f77b4", lw=1.6)
    ax.axhline(0.0, color="gray", lw=0.9, ls=":")
    ax.axvline(k_star, color="crimson", lw=1.0, ls="--")
    ax.fill_between(k_axis, 0, sigma, where=sigma > 0, color="#1f77b4", alpha=0.15)
    ax.set_xlim(0.0, 4.0)
    ax.set_xlabel("Wavenumber k (1/model unit)")
    ax.set_ylabel("Growth rate (1/model time)")
    ax.set_title("(a) Dispersion relation")
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.semilogy(times[sel], amp[sel], "o", ms=2.5, color="#1f77b4",
                label="numerical integration")
    ax.semilogy(times[sel], amp[sel][0] * np.exp(g_peak * times[sel]),
                "-", color="crimson", lw=1.2, label="analytic prediction")
    ax.set_xlabel("Time (model units)")
    ax.set_ylabel("Amplitude of the seeded mode")
    ax.set_title("(b) Growth-rate verification")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, which="both")

    ax = axes[1, 0]
    show = field[:24, :32]
    lim = float(np.abs(field).max())
    im = ax.imshow(show, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   aspect="equal",
                   extent=(0.0, show.shape[1] * DX, 0.0, show.shape[0] * DX))
    ax.set_title("(c) Growing mode")
    ax.set_xlabel("x (model units); wavelength %.2f units" % lam)
    ax.set_ylabel("y (model units)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="U minus U*")

    ax = axes[1, 1]
    ax.semilogy(bins[1:], spec[1:], color="#2ca02c", lw=1.5)
    ax.axvline(k_star, color="crimson", lw=1.0, ls="--",
               label="linear peak k = %.2f" % k_star)
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(1e-2, None)
    ax.set_xlabel("Wavenumber k (1/model unit)")
    ax.set_ylabel("Radial power")
    ax.set_title("(d) Power spectrum of the final field")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2, which="both")

    fig.tight_layout()
    fig.savefig(FIG / "fig17_turing_verification.png")
    plt.close(fig)

    np.savez_compressed(
        OUT / "turing_linear_check.npz",
        k_axis=k_axis, sigma=sigma, du=du, dv=dv, k_star=k_star, g_peak=g_peak,
        measured_slope=slope, k_measured=k_meas, times=times, amp=amp,
        field=field, bins=bins, spectrum=spec, dx=DX, wavelength=lam,
        band_min=float(band.min()), band_max=float(band.max()),
    )
    print("k*=%.3f sigma=%.6e measured=%.6e k_meas=%.3f lambda=%.3f band=[%.2f,%.2f]"
          % (k_star, g_peak, slope, k_meas, lam, band.min(), band.max()))
    print("du=%.2e dv=%.2e field amp=%.2e" % (du, dv, np.abs(field).max()))


if __name__ == "__main__":
    main()
