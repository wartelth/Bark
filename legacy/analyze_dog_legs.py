#!/usr/bin/env python3
"""
In-depth time-series analysis of 4-leg IMU data from a dog.
Generates publication-quality plots examining inter-leg relationships.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from scipy import signal, stats
from scipy.signal import welch, coherence
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(OUT, "DATA.TXT")

LEGS = ["Front-Left (IMU1)", "Front-Right (IMU2)", "Hind-Left (IMU3)", "Hind-Right (IMU4)"]
LEG_SHORT = ["FL", "FR", "HL", "HR"]
LEG_COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]

DARK_BG = "#0D1117"
CARD_BG = "#161B22"
GRID_COLOR = "#21262D"
TEXT_COLOR = "#C9D1D9"
ACCENT = "#58A6FF"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.4,
    "font.family": "monospace",
    "font.size": 10,
    "axes.titlesize": 13,
    "figure.titlesize": 18,
})


def load_data():
    df = pd.read_csv(DATA_PATH, sep=";", header=0)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip()

    zero_mask = (df == 0).all(axis=1) | (
        (df.iloc[:, 0:6].nunique(axis=1) <= 2) &
        (df.iloc[:, 6:12].nunique(axis=1) <= 2) &
        (df.iloc[:, 12:18].nunique(axis=1) <= 2) &
        (df.iloc[:, 18:24].nunique(axis=1) <= 2) &
        (df.abs().sum(axis=1) < 1e-6)
    )

    stale_mask = pd.Series(False, index=df.index)
    for leg_idx in range(4):
        cols = df.columns[leg_idx * 6:(leg_idx + 1) * 6]
        leg_data = df[cols]
        is_constant = leg_data.diff().abs().sum(axis=1) == 0
        is_constant.iloc[0] = False
        run_lengths = is_constant.groupby((~is_constant).cumsum()).transform("sum")
        stale_mask |= (run_lengths > 10)

    dropout_mask = zero_mask | stale_mask
    print(f"Total rows: {len(df)}")
    print(f"Dropout/stale rows: {dropout_mask.sum()} ({100*dropout_mask.sum()/len(df):.1f}%)")
    print(f"Active rows: {(~dropout_mask).sum()}")

    return df, dropout_mask.values


def accel_magnitude(df, leg_idx):
    ax = df.iloc[:, leg_idx * 6 + 0]
    ay = df.iloc[:, leg_idx * 6 + 1]
    az = df.iloc[:, leg_idx * 6 + 2]
    return np.sqrt(ax**2 + ay**2 + az**2)


def gyro_magnitude(df, leg_idx):
    gx = df.iloc[:, leg_idx * 6 + 3]
    gy = df.iloc[:, leg_idx * 6 + 4]
    gz = df.iloc[:, leg_idx * 6 + 5]
    return np.sqrt(gx**2 + gy**2 + gz**2)


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> Saved {name}")


# ---------------------------------------------------------------------------
# PLOT 1 — Raw time-series overview (all 24 channels)
# ---------------------------------------------------------------------------
def plot_raw_overview(df, mask):
    fig, axes = plt.subplots(8, 1, figsize=(22, 20), sharex=True)
    fig.suptitle("RAW IMU SIGNALS — ALL 4 LEGS", fontweight="bold", y=0.98, fontsize=20)
    t = np.arange(len(df))
    chan_labels = ["AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ"]

    for ch_idx, (ax, label) in enumerate(zip(axes[:6], chan_labels)):
        for leg_idx in range(4):
            col = df.columns[leg_idx * 6 + ch_idx]
            vals = df[col].values.copy()
            vals[mask] = np.nan
            ax.plot(t, vals, color=LEG_COLORS[leg_idx], alpha=0.8, linewidth=0.6,
                    label=LEG_SHORT[leg_idx])
        ax.set_ylabel(label, fontweight="bold")
        ax.grid(True, linewidth=0.3)
        if ch_idx == 0:
            ax.legend(loc="upper right", ncol=4, fontsize=8, framealpha=0.5)

    for leg_idx in range(4):
        am = accel_magnitude(df, leg_idx).values.copy()
        am[mask] = np.nan
        axes[6].plot(t, am, color=LEG_COLORS[leg_idx], alpha=0.8, linewidth=0.6,
                     label=LEG_SHORT[leg_idx])
    axes[6].set_ylabel("|Accel|", fontweight="bold")
    axes[6].grid(True, linewidth=0.3)

    for leg_idx in range(4):
        gm = gyro_magnitude(df, leg_idx).values.copy()
        gm[mask] = np.nan
        axes[7].plot(t, gm, color=LEG_COLORS[leg_idx], alpha=0.8, linewidth=0.6,
                     label=LEG_SHORT[leg_idx])
    axes[7].set_ylabel("|Gyro|", fontweight="bold")
    axes[7].set_xlabel("Sample index", fontweight="bold")
    axes[7].grid(True, linewidth=0.3)

    for ax in axes:
        ax.set_xlim(0, len(df))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "01_raw_signals_overview.png")


# ---------------------------------------------------------------------------
# PLOT 2 — Acceleration magnitude comparison + smoothed envelope
# ---------------------------------------------------------------------------
def plot_accel_magnitude_comparison(df, mask):
    fig, axes = plt.subplots(5, 1, figsize=(22, 16), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1, 1.4]})
    fig.suptitle("ACCELERATION MAGNITUDE PER LEG — WITH SMOOTHED ENVELOPE",
                 fontweight="bold", y=0.98, fontsize=20)
    t = np.arange(len(df))
    clean = ~mask

    for leg_idx in range(4):
        am = accel_magnitude(df, leg_idx).values.copy()
        am[mask] = np.nan
        axes[leg_idx].fill_between(t, 0, am, color=LEG_COLORS[leg_idx], alpha=0.25)
        axes[leg_idx].plot(t, am, color=LEG_COLORS[leg_idx], alpha=0.7, linewidth=0.5)

        am_clean = accel_magnitude(df, leg_idx).values.copy()
        am_clean[mask] = np.nan
        smoothed = pd.Series(am_clean).rolling(50, center=True, min_periods=1).mean().values
        axes[leg_idx].plot(t, smoothed, color="white", linewidth=1.5, alpha=0.9)

        axes[leg_idx].set_ylabel(LEG_SHORT[leg_idx], fontweight="bold", fontsize=14)
        axes[leg_idx].grid(True, linewidth=0.3)
        axes[leg_idx].set_xlim(0, len(df))

    for leg_idx in range(4):
        am = accel_magnitude(df, leg_idx).values.copy()
        am[mask] = np.nan
        axes[4].plot(t, am, color=LEG_COLORS[leg_idx], alpha=0.7, linewidth=0.6,
                     label=LEGS[leg_idx])
    axes[4].set_ylabel("All Legs", fontweight="bold")
    axes[4].set_xlabel("Sample index", fontweight="bold")
    axes[4].legend(fontsize=9, framealpha=0.5)
    axes[4].grid(True, linewidth=0.3)
    axes[4].set_xlim(0, len(df))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "02_accel_magnitude_per_leg.png")


# ---------------------------------------------------------------------------
# PLOT 3 — Cross-correlation heatmap matrix (all 24 channels)
# ---------------------------------------------------------------------------
def plot_cross_correlation_matrix(df, mask):
    clean_df = df[~mask].reset_index(drop=True)
    nice_names = []
    for leg_idx in range(4):
        for ch in ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]:
            nice_names.append(f"{LEG_SHORT[leg_idx]}_{ch}")
    clean_df.columns = nice_names

    corr = clean_df.corr()

    fig, ax = plt.subplots(figsize=(18, 16))
    fig.suptitle("INTER-LEG CROSS-CORRELATION MATRIX (24 CHANNELS)",
                 fontweight="bold", y=0.98, fontsize=20)

    cmap = LinearSegmentedColormap.from_list("custom",
        ["#1B1F3B", "#457B9D", "#F1FAEE", "#E9C46A", "#E63946"])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(len(nice_names)))
    ax.set_xticklabels(nice_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(nice_names)))
    ax.set_yticklabels(nice_names, fontsize=7)

    for i in range(4):
        rect = plt.Rectangle((i * 6 - 0.5, i * 6 - 0.5), 6, 6, linewidth=2,
                              edgecolor="white", facecolor="none")
        ax.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson Correlation", fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "03_cross_correlation_matrix.png")


# ---------------------------------------------------------------------------
# PLOT 4 — Pairwise leg-to-leg correlation (accel magnitude + gyro magnitude)
# ---------------------------------------------------------------------------
def plot_pairwise_leg_correlation(df, mask):
    clean_arr = ~mask
    mask_arr = mask
    fig, axes = plt.subplots(2, 1, figsize=(18, 14))
    fig.suptitle("PAIRWISE LEG CORRELATION — ACCEL & GYRO MAGNITUDE",
                 fontweight="bold", y=0.98, fontsize=20)

    for plot_idx, (mag_func, title) in enumerate([
        (accel_magnitude, "Acceleration Magnitude"),
        (gyro_magnitude, "Gyroscope Magnitude"),
    ]):
        mags = np.column_stack([
            mag_func(df, i).values for i in range(4)
        ])
        mags[mask_arr] = np.nan
        mags_clean = mags[clean_arr]

        corr = np.corrcoef(mags_clean.T)
        labels = LEG_SHORT

        ax = axes[plot_idx]
        cmap = LinearSegmentedColormap.from_list("custom",
            ["#1B1F3B", "#457B9D", "#F1FAEE", "#E9C46A", "#E63946"])
        im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)

        for i in range(4):
            for j in range(4):
                color = "white" if abs(corr[i, j]) > 0.5 else TEXT_COLOR
                ax.text(j, i, f"{corr[i,j]:.3f}", ha="center", va="center",
                        fontsize=14, fontweight="bold", color=color)

        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_yticks(range(4))
        ax.set_yticklabels(labels, fontsize=12)
        ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
        plt.colorbar(im, ax=ax, shrink=0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "04_pairwise_leg_correlation.png")


# ---------------------------------------------------------------------------
# PLOT 5 — Phase analysis: lag cross-correlation between leg pairs
# ---------------------------------------------------------------------------
def plot_phase_lag_analysis(df, mask):
    clean = ~mask
    pairs = [(0, 1, "FL vs FR (lateral)"), (2, 3, "HL vs HR (lateral)"),
             (0, 2, "FL vs HL (ipsilateral)"), (1, 3, "FR vs HR (ipsilateral)"),
             (0, 3, "FL vs HR (diagonal)"), (1, 2, "FR vs HL (diagonal)")]

    fig, axes = plt.subplots(3, 2, figsize=(20, 16))
    fig.suptitle("PHASE-LAG CROSS-CORRELATION BETWEEN LEG PAIRS (ACCEL MAGNITUDE)",
                 fontweight="bold", y=0.98, fontsize=20)

    max_lag = 200
    for idx, (i, j, label) in enumerate(pairs):
        ax = axes.flat[idx]
        am_i = accel_magnitude(df, i).values.copy()
        am_j = accel_magnitude(df, j).values.copy()
        am_i[mask] = 0
        am_j[mask] = 0
        am_i -= np.mean(am_i[clean])
        am_j -= np.mean(am_j[clean])

        xcorr = np.correlate(am_i, am_j, mode="full")
        mid = len(xcorr) // 2
        lags = np.arange(-max_lag, max_lag + 1)
        xcorr_slice = xcorr[mid - max_lag:mid + max_lag + 1]
        xcorr_slice /= np.max(np.abs(xcorr_slice)) + 1e-12

        peak_lag = lags[np.argmax(xcorr_slice)]

        color_map = {"lateral": "#E63946", "ipsilateral": "#2A9D8F", "diagonal": "#E9C46A"}
        ptype = label.split("(")[1].rstrip(")")
        c = color_map.get(ptype, ACCENT)

        ax.fill_between(lags, 0, xcorr_slice, alpha=0.3, color=c)
        ax.plot(lags, xcorr_slice, color=c, linewidth=1.5)
        ax.axvline(peak_lag, color="white", linestyle="--", alpha=0.7, linewidth=1)
        ax.axhline(0, color=GRID_COLOR, linewidth=0.5)
        ax.set_title(f"{label}\npeak lag = {peak_lag} samples", fontweight="bold")
        ax.set_xlabel("Lag (samples)")
        ax.set_ylabel("Normalized XCorr")
        ax.grid(True, linewidth=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "05_phase_lag_cross_correlation.png")


# ---------------------------------------------------------------------------
# PLOT 6 — Spectral analysis: PSD per leg
# ---------------------------------------------------------------------------
def plot_spectral_analysis(df, mask):
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("POWER SPECTRAL DENSITY — ACCEL MAGNITUDE PER LEG",
                 fontweight="bold", y=0.98, fontsize=20)

    fs = 50  # assumed sample rate

    for leg_idx in range(4):
        ax = axes.flat[leg_idx]
        am = accel_magnitude(df, leg_idx).values.copy()
        am[mask] = np.nanmean(am[~mask])
        am -= np.mean(am)

        freqs, psd = welch(am, fs=fs, nperseg=256, noverlap=128)

        ax.semilogy(freqs, psd, color=LEG_COLORS[leg_idx], linewidth=2)
        ax.fill_between(freqs, psd, alpha=0.3, color=LEG_COLORS[leg_idx])
        ax.set_title(LEGS[leg_idx], fontweight="bold", fontsize=14)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (g²/Hz)")
        ax.grid(True, linewidth=0.3)
        ax.set_xlim(0, fs / 2)

        peak_f = freqs[np.argmax(psd[1:]) + 1]
        ax.axvline(peak_f, color="white", linestyle="--", alpha=0.6)
        ax.text(peak_f + 0.3, np.max(psd) * 0.5, f"peak: {peak_f:.1f} Hz",
                color="white", fontsize=10, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "06_spectral_psd_per_leg.png")


# ---------------------------------------------------------------------------
# PLOT 7 — Spectral coherence between leg pairs
# ---------------------------------------------------------------------------
def plot_spectral_coherence(df, mask):
    pairs = [(0, 1, "FL-FR"), (2, 3, "HL-HR"),
             (0, 2, "FL-HL"), (1, 3, "FR-HR"),
             (0, 3, "FL-HR"), (1, 2, "FR-HL")]
    fs = 50

    fig, axes = plt.subplots(3, 2, figsize=(20, 16))
    fig.suptitle("SPECTRAL COHERENCE BETWEEN LEG PAIRS",
                 fontweight="bold", y=0.98, fontsize=20)

    pair_types = ["lateral", "lateral", "ipsilateral", "ipsilateral", "diagonal", "diagonal"]
    color_map = {"lateral": "#E63946", "ipsilateral": "#2A9D8F", "diagonal": "#E9C46A"}

    for idx, (i, j, label) in enumerate(pairs):
        ax = axes.flat[idx]
        am_i = accel_magnitude(df, i).values.copy()
        am_j = accel_magnitude(df, j).values.copy()
        am_i[mask] = np.nanmean(am_i[~mask])
        am_j[mask] = np.nanmean(am_j[~mask])

        f, coh = coherence(am_i, am_j, fs=fs, nperseg=256, noverlap=128)
        c = color_map[pair_types[idx]]

        ax.fill_between(f, 0, coh, alpha=0.3, color=c)
        ax.plot(f, coh, color=c, linewidth=2)
        ax.set_title(f"{label} ({pair_types[idx]})", fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Coherence")
        ax.grid(True, linewidth=0.3)
        ax.axhline(0.5, color="white", linestyle=":", alpha=0.4)

        avg_coh = np.mean(coh)
        ax.text(0.95, 0.95, f"avg: {avg_coh:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG, alpha=0.8))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "07_spectral_coherence.png")


# ---------------------------------------------------------------------------
# PLOT 8 — PCA of all 4 legs in acceleration space
# ---------------------------------------------------------------------------
def plot_pca_analysis(df, mask):
    clean_arr = ~mask
    mags = np.column_stack([
        accel_magnitude(df, i).values for i in range(4)
    ])
    mags_clean = mags[clean_arr]

    scaler = StandardScaler()
    mags_scaled = scaler.fit_transform(mags_clean)

    pca = PCA(n_components=4)
    pcs = pca.fit_transform(mags_scaled)

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle("PCA ANALYSIS — INTER-LEG ACCELERATION PATTERNS",
                 fontweight="bold", y=0.98, fontsize=20)

    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(range(4), pca.explained_variance_ratio_ * 100,
                   color=[ACCENT, "#E9C46A", "#2A9D8F", "#E63946"], edgecolor="white", linewidth=0.5)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels([f"PC{i+1}" for i in range(4)])
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title("Variance Explained", fontweight="bold")
    ax1.grid(True, linewidth=0.3, axis="y")
    for bar, val in zip(bars, pca.explained_variance_ratio_):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val*100:.1f}%", ha="center", fontsize=11, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1:])
    for i in range(4):
        ax2.barh(np.arange(4) + i * 0.2 - 0.3, pca.components_[0] if i == 0 else pca.components_[i],
                 height=0.18, color=[ACCENT, "#E9C46A", "#2A9D8F", "#E63946"][i],
                 label=f"PC{i+1}", edgecolor="white", linewidth=0.3)
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(LEG_SHORT)
    ax2.set_xlabel("Loading")
    ax2.set_title("PCA Loadings (per leg)", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, linewidth=0.3, axis="x")

    ax3 = fig.add_subplot(gs[1, 0])
    sc = ax3.scatter(pcs[:, 0], pcs[:, 1], c=np.arange(len(pcs)), cmap="magma",
                     s=2, alpha=0.5)
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")
    ax3.set_title("PC1 vs PC2 (colored by time)", fontweight="bold")
    ax3.grid(True, linewidth=0.3)
    plt.colorbar(sc, ax=ax3, label="Sample idx", shrink=0.8)

    ax4 = fig.add_subplot(gs[1, 1])
    sc2 = ax4.scatter(pcs[:, 0], pcs[:, 2], c=np.arange(len(pcs)), cmap="magma",
                      s=2, alpha=0.5)
    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC3")
    ax4.set_title("PC1 vs PC3 (colored by time)", fontweight="bold")
    ax4.grid(True, linewidth=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    sc3 = ax5.scatter(pcs[:, 1], pcs[:, 2], c=np.arange(len(pcs)), cmap="magma",
                      s=2, alpha=0.5)
    ax5.set_xlabel("PC2")
    ax5.set_ylabel("PC3")
    ax5.set_title("PC2 vs PC3 (colored by time)", fontweight="bold")
    ax5.grid(True, linewidth=0.3)

    savefig(fig, "08_pca_analysis.png")


# ---------------------------------------------------------------------------
# PLOT 9 — Rolling correlation between diagonal and lateral pairs
# ---------------------------------------------------------------------------
def plot_rolling_correlation(df, mask):
    window = 100
    pairs = [
        (0, 3, "FL-HR (diagonal)", "#E9C46A"),
        (1, 2, "FR-HL (diagonal)", "#E63946"),
        (0, 1, "FL-FR (front lateral)", "#457B9D"),
        (2, 3, "HL-HR (hind lateral)", "#2A9D8F"),
        (0, 2, "FL-HL (left ipsilateral)", "#A8DADC"),
        (1, 3, "FR-HR (right ipsilateral)", "#F4A261"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(22, 16), sharex=True)
    fig.suptitle(f"ROLLING CORRELATION (window={window}) — LEG PAIR DYNAMICS",
                 fontweight="bold", y=0.98, fontsize=20)

    pair_groups = [
        (pairs[0:2], "Diagonal Pairs"),
        (pairs[2:4], "Lateral Pairs"),
        (pairs[4:6], "Ipsilateral Pairs"),
    ]

    t = np.arange(len(df))
    mask_arr = mask
    for ax, (group, title) in zip(axes, pair_groups):
        for (i, j, label, color) in group:
            am_i = pd.Series(accel_magnitude(df, i).values)
            am_j = pd.Series(accel_magnitude(df, j).values)
            am_i[mask_arr] = np.nan
            am_j[mask_arr] = np.nan

            roll_corr = am_i.rolling(window, center=True, min_periods=window // 2).corr(am_j)
            ax.plot(t, roll_corr.values, color=color, linewidth=1.2, alpha=0.9, label=label)

        ax.axhline(0, color="white", linestyle=":", alpha=0.3)
        ax.axhline(1, color="white", linestyle=":", alpha=0.15)
        ax.axhline(-1, color="white", linestyle=":", alpha=0.15)
        ax.set_ylabel("Correlation")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(-1.1, 1.1)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.5)
        ax.grid(True, linewidth=0.3)
        ax.set_xlim(0, len(df))

    axes[-1].set_xlabel("Sample index", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "09_rolling_correlation.png")


# ---------------------------------------------------------------------------
# PLOT 10 — Gait symmetry index over time
# ---------------------------------------------------------------------------
def plot_gait_symmetry(df, mask):
    fig, axes = plt.subplots(3, 1, figsize=(22, 14), sharex=True)
    fig.suptitle("GAIT SYMMETRY INDICES OVER TIME",
                 fontweight="bold", y=0.98, fontsize=20)

    window = 80
    t = np.arange(len(df))

    am = [accel_magnitude(df, i).values.copy() for i in range(4)]
    for a in am:
        a[mask] = np.nan

    def symmetry_index(a, b):
        s = pd.Series(a)
        o = pd.Series(b)
        sm = s.rolling(window, center=True, min_periods=1).mean()
        om = o.rolling(window, center=True, min_periods=1).mean()
        si = 2 * (sm - om) / (sm + om + 1e-9) * 100
        return si.values

    si_front = symmetry_index(am[0], am[1])
    si_hind = symmetry_index(am[2], am[3])
    si_left = symmetry_index(am[0], am[2])
    si_right = symmetry_index(am[1], am[3])
    si_diag1 = symmetry_index(am[0], am[3])
    si_diag2 = symmetry_index(am[1], am[2])

    axes[0].plot(t, si_front, color="#457B9D", linewidth=1.2, label="Front (FL vs FR)")
    axes[0].plot(t, si_hind, color="#2A9D8F", linewidth=1.2, label="Hind (HL vs HR)")
    axes[0].axhline(0, color="white", linestyle=":", alpha=0.3)
    axes[0].fill_between(t, -10, 10, color="white", alpha=0.04)
    axes[0].set_ylabel("Symmetry Index (%)")
    axes[0].set_title("Lateral Symmetry (L vs R)", fontweight="bold")
    axes[0].legend(fontsize=9, framealpha=0.5)
    axes[0].grid(True, linewidth=0.3)

    axes[1].plot(t, si_left, color="#E63946", linewidth=1.2, label="Left (FL vs HL)")
    axes[1].plot(t, si_right, color="#E9C46A", linewidth=1.2, label="Right (FR vs HR)")
    axes[1].axhline(0, color="white", linestyle=":", alpha=0.3)
    axes[1].fill_between(t, -10, 10, color="white", alpha=0.04)
    axes[1].set_ylabel("Symmetry Index (%)")
    axes[1].set_title("Fore-Hind Symmetry", fontweight="bold")
    axes[1].legend(fontsize=9, framealpha=0.5)
    axes[1].grid(True, linewidth=0.3)

    axes[2].plot(t, si_diag1, color="#F4A261", linewidth=1.2, label="Diagonal 1 (FL vs HR)")
    axes[2].plot(t, si_diag2, color="#A8DADC", linewidth=1.2, label="Diagonal 2 (FR vs HL)")
    axes[2].axhline(0, color="white", linestyle=":", alpha=0.3)
    axes[2].fill_between(t, -10, 10, color="white", alpha=0.04)
    axes[2].set_ylabel("Symmetry Index (%)")
    axes[2].set_xlabel("Sample index", fontweight="bold")
    axes[2].set_title("Diagonal Symmetry", fontweight="bold")
    axes[2].legend(fontsize=9, framealpha=0.5)
    axes[2].grid(True, linewidth=0.3)

    for ax in axes:
        ax.set_xlim(0, len(df))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "10_gait_symmetry_index.png")


# ---------------------------------------------------------------------------
# PLOT 11 — Scatter matrix: leg-vs-leg acceleration magnitude
# ---------------------------------------------------------------------------
def plot_leg_scatter_matrix(df, mask):
    clean_arr = ~mask
    mags = pd.DataFrame({
        LEG_SHORT[i]: accel_magnitude(df, i).values for i in range(4)
    })
    mags_clean = mags[clean_arr].reset_index(drop=True)

    fig, axes = plt.subplots(4, 4, figsize=(18, 18))
    fig.suptitle("LEG-vs-LEG ACCELERATION SCATTER MATRIX",
                 fontweight="bold", y=0.98, fontsize=20)

    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            if i == j:
                ax.hist(mags_clean.iloc[:, i].values, bins=60,
                        color=LEG_COLORS[i], alpha=0.7, edgecolor="none")
                ax.set_title(LEG_SHORT[i], fontweight="bold", fontsize=12)
            else:
                ax.scatter(mags_clean.iloc[:, j].values, mags_clean.iloc[:, i].values,
                           s=1, alpha=0.15, color=LEG_COLORS[i])
                r = np.corrcoef(mags_clean.iloc[:, j], mags_clean.iloc[:, i])[0, 1]
                ax.text(0.05, 0.95, f"r={r:.3f}", transform=ax.transAxes,
                        fontsize=10, fontweight="bold", va="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=DARK_BG, alpha=0.8))
            ax.grid(True, linewidth=0.2)
            if i == 3:
                ax.set_xlabel(LEG_SHORT[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(LEG_SHORT[i], fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "11_leg_scatter_matrix.png")


# ---------------------------------------------------------------------------
# PLOT 12 — Gyro energy distribution radar/polar chart
# ---------------------------------------------------------------------------
def plot_gyro_energy_polar(df, mask):
    clean_arr = ~mask
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), subplot_kw={"projection": "polar"})
    fig.suptitle("GYROSCOPE ENERGY DISTRIBUTION PER AXIS (POLAR)",
                 fontweight="bold", y=1.05, fontsize=20)

    for leg_idx in range(4):
        ax = axes[leg_idx]
        ax.set_facecolor(CARD_BG)

        gx = df.iloc[clean_arr, leg_idx * 6 + 3].values
        gy = df.iloc[clean_arr, leg_idx * 6 + 4].values
        gz = df.iloc[clean_arr, leg_idx * 6 + 5].values

        energies = [np.mean(gx**2), np.mean(gy**2), np.mean(gz**2)]
        total = sum(energies)
        fracs = [e / total for e in energies]

        angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
        fracs_closed = fracs + [fracs[0]]
        angles_closed = angles + [angles[0]]

        ax.fill(angles_closed, fracs_closed, color=LEG_COLORS[leg_idx], alpha=0.35)
        ax.plot(angles_closed, fracs_closed, color=LEG_COLORS[leg_idx], linewidth=2.5)
        ax.scatter(angles, fracs, color="white", s=60, zorder=5)

        ax.set_xticks(angles)
        ax.set_xticklabels(["GyroX", "GyroY", "GyroZ"], fontsize=9, fontweight="bold")
        ax.set_title(LEGS[leg_idx], fontweight="bold", pad=20, fontsize=11)

    fig.tight_layout()
    savefig(fig, "12_gyro_energy_polar.png")


# ---------------------------------------------------------------------------
# PLOT 13 — Zoomed-in active window: detailed gait cycle view
# ---------------------------------------------------------------------------
def plot_zoomed_gait_cycle(df, mask):
    am_total = sum(accel_magnitude(df, i).values for i in range(4))
    am_total[mask] = 0
    smoothed = uniform_filter1d(am_total, 50)
    active_regions = smoothed > np.percentile(smoothed[~mask], 70)

    starts = np.where(np.diff(active_regions.astype(int)) == 1)[0]
    if len(starts) == 0:
        starts = [0]

    best_start = starts[0]
    window_len = min(400, len(df) - best_start)
    sl = slice(best_start, best_start + window_len)

    fig, axes = plt.subplots(4, 1, figsize=(22, 16), sharex=True)
    fig.suptitle(f"ZOOMED GAIT CYCLE — SAMPLES {best_start} to {best_start+window_len}",
                 fontweight="bold", y=0.98, fontsize=20)

    t = np.arange(window_len)

    for leg_idx in range(4):
        ax = axes[leg_idx]
        am = accel_magnitude(df, leg_idx).values[sl]
        gm = gyro_magnitude(df, leg_idx).values[sl]

        ax.plot(t, am, color=LEG_COLORS[leg_idx], linewidth=1.8, alpha=0.9, label="|Accel|")
        ax2 = ax.twinx()
        ax2.plot(t, gm, color="white", linewidth=1, alpha=0.5, linestyle="--", label="|Gyro|")
        ax2.set_ylabel("|Gyro| (°/s)", fontsize=9)
        ax2.tick_params(axis="y", colors=TEXT_COLOR)

        peaks, _ = signal.find_peaks(am, distance=15, prominence=0.1)
        ax.scatter(peaks, am[peaks], color="white", s=30, zorder=5, marker="v")

        ax.set_ylabel(f"{LEGS[leg_idx]}\n|Accel| (g)", fontweight="bold", fontsize=10)
        ax.grid(True, linewidth=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if leg_idx == 0:
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Sample (relative)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "13_zoomed_gait_cycle.png")


# ---------------------------------------------------------------------------
# PLOT 14 — Statistical summary dashboard
# ---------------------------------------------------------------------------
def plot_statistics_dashboard(df, mask):
    clean = ~mask
    fig = plt.figure(figsize=(24, 18))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("STATISTICAL DASHBOARD — DOG LEG IMU ANALYSIS",
                 fontweight="bold", y=0.98, fontsize=22)

    ax_box_accel = fig.add_subplot(gs[0, 0:2])
    accel_data = [accel_magnitude(df, i).values[clean] for i in range(4)]
    bp = ax_box_accel.boxplot(accel_data, patch_artist=True, labels=LEG_SHORT,
                              widths=0.6, showfliers=False)
    for patch, color in zip(bp["boxes"], LEG_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(2)
    ax_box_accel.set_title("Accel Magnitude Distribution", fontweight="bold")
    ax_box_accel.set_ylabel("|Accel| (g)")
    ax_box_accel.grid(True, linewidth=0.3, axis="y")

    ax_box_gyro = fig.add_subplot(gs[0, 2:4])
    gyro_data = [gyro_magnitude(df, i).values[clean] for i in range(4)]
    bp2 = ax_box_gyro.boxplot(gyro_data, patch_artist=True, labels=LEG_SHORT,
                               widths=0.6, showfliers=False)
    for patch, color in zip(bp2["boxes"], LEG_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp2["medians"]:
        median.set_color("white")
        median.set_linewidth(2)
    ax_box_gyro.set_title("Gyro Magnitude Distribution", fontweight="bold")
    ax_box_gyro.set_ylabel("|Gyro| (°/s)")
    ax_box_gyro.grid(True, linewidth=0.3, axis="y")

    ax_violin = fig.add_subplot(gs[1, 0:2])
    violin_data = pd.DataFrame({
        LEG_SHORT[i]: accel_magnitude(df, i).values[clean] for i in range(4)
    })
    parts = ax_violin.violinplot([violin_data[c].values for c in LEG_SHORT],
                                  showmeans=True, showmedians=True)
    for idx, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(LEG_COLORS[idx])
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("white")
    parts["cmedians"].set_color(ACCENT)
    ax_violin.set_xticks(range(1, 5))
    ax_violin.set_xticklabels(LEG_SHORT)
    ax_violin.set_title("Accel Magnitude Violin Plot", fontweight="bold")
    ax_violin.set_ylabel("|Accel| (g)")
    ax_violin.grid(True, linewidth=0.3, axis="y")

    ax_kde = fig.add_subplot(gs[1, 2:4])
    for i in range(4):
        am = accel_magnitude(df, i).values[clean]
        kde_x = np.linspace(am.min(), np.percentile(am, 99), 300)
        kde = stats.gaussian_kde(am)
        ax_kde.plot(kde_x, kde(kde_x), color=LEG_COLORS[i], linewidth=2, label=LEGS[i])
        ax_kde.fill_between(kde_x, kde(kde_x), alpha=0.15, color=LEG_COLORS[i])
    ax_kde.set_title("Accel Magnitude KDE", fontweight="bold")
    ax_kde.set_xlabel("|Accel| (g)")
    ax_kde.set_ylabel("Density")
    ax_kde.legend(fontsize=8, framealpha=0.5)
    ax_kde.grid(True, linewidth=0.3)

    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")
    stats_rows = []
    for i in range(4):
        am = accel_magnitude(df, i).values[clean]
        gm = gyro_magnitude(df, i).values[clean]
        stats_rows.append([
            LEGS[i],
            f"{np.mean(am):.4f}", f"{np.std(am):.4f}",
            f"{np.median(am):.4f}", f"{np.max(am):.4f}",
            f"{np.mean(gm):.2f}", f"{np.std(gm):.2f}",
            f"{np.max(gm):.2f}",
            f"{stats.skew(am):.3f}", f"{stats.kurtosis(am):.3f}",
        ])
    table = ax_table.table(
        cellText=stats_rows,
        colLabels=["Leg", "Accel μ", "Accel σ", "Accel med", "Accel max",
                    "Gyro μ", "Gyro σ", "Gyro max", "Skewness", "Kurtosis"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if row == 0:
            cell.set_facecolor(ACCENT)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(CARD_BG)
            cell.set_text_props(color=TEXT_COLOR)

    savefig(fig, "14_statistics_dashboard.png")


# ---------------------------------------------------------------------------
# PLOT 15 — Spectrogram heatmap per leg
# ---------------------------------------------------------------------------
def plot_spectrogram(df, mask):
    fs = 50
    fig, axes = plt.subplots(4, 1, figsize=(22, 18), sharex=True)
    fig.suptitle("SPECTROGRAM — ACCELERATION MAGNITUDE PER LEG",
                 fontweight="bold", y=0.98, fontsize=20)

    for leg_idx in range(4):
        ax = axes[leg_idx]
        am = accel_magnitude(df, leg_idx).values.copy()
        am[mask] = np.nanmean(am[~mask])
        am -= np.mean(am)

        f, t_spec, Sxx = signal.spectrogram(am, fs=fs, nperseg=128, noverlap=96)
        ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud",
                      cmap="inferno")
        ax.set_ylabel(f"{LEG_SHORT[leg_idx]}\nFreq (Hz)", fontweight="bold")
        ax.set_ylim(0, fs / 2)

    axes[-1].set_xlabel("Time (s)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, "15_spectrogram_per_leg.png")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  DOG LEG IMU ANALYSIS — GENERATING PLOTS")
    print("=" * 60)

    df, mask = load_data()

    print("\n[1/15] Raw signals overview...")
    plot_raw_overview(df, mask)

    print("[2/15] Acceleration magnitude per leg...")
    plot_accel_magnitude_comparison(df, mask)

    print("[3/15] Cross-correlation matrix...")
    plot_cross_correlation_matrix(df, mask)

    print("[4/15] Pairwise leg correlation...")
    plot_pairwise_leg_correlation(df, mask)

    print("[5/15] Phase-lag cross-correlation...")
    plot_phase_lag_analysis(df, mask)

    print("[6/15] Power spectral density...")
    plot_spectral_analysis(df, mask)

    print("[7/15] Spectral coherence...")
    plot_spectral_coherence(df, mask)

    print("[8/15] PCA analysis...")
    plot_pca_analysis(df, mask)

    print("[9/15] Rolling correlation...")
    plot_rolling_correlation(df, mask)

    print("[10/15] Gait symmetry indices...")
    plot_gait_symmetry(df, mask)

    print("[11/15] Leg scatter matrix...")
    plot_leg_scatter_matrix(df, mask)

    print("[12/15] Gyro energy polar...")
    plot_gyro_energy_polar(df, mask)

    print("[13/15] Zoomed gait cycle...")
    plot_zoomed_gait_cycle(df, mask)

    print("[14/15] Statistics dashboard...")
    plot_statistics_dashboard(df, mask)

    print("[15/15] Spectrogram...")
    plot_spectrogram(df, mask)

    print("\n" + "=" * 60)
    print("  ALL 15 PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 60)
