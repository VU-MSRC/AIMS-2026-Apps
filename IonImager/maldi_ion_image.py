"""
MALDI Ion Image Generator — Bruker TDF-SDK v3.3.6.2
=====================================================
Generates a 2D ion image for a given m/z window from a Bruker MALDI .d dataset.
Uses the TSF (TimsSpectrum File) API, which is the correct interface for MALDI data.

Directory layout expected next to this script (or on sys.path):
    tsfdata.py          <- from timsdata/examples/py/
    libs/
        libtimsdata.so  <- Linux   (from timsdata/linux64/)
        timsdata.dll    <- Windows (from timsdata/win64/)

Dependencies:
    pip install numpy matplotlib tqdm

Usage:
    python maldi_ion_image.py \\
        --input   /path/to/dataset.d \\
        --mz_min  750.5 \\
        --mz_max  751.5 \\
        --output  ion_image.png

Optional:
    --colormap  viridis   (any matplotlib colormap; default: hot)
    --dpi       300       (output resolution; default: 150)
    --normalize           (normalise each pixel to the local TIC)
"""

import argparse
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Import TsfData from the Bruker SDK ───────────────────────────────────────
try:
    from tsfdata import TsfData
except ImportError:
    sdk_dir = Path(__file__).parent
    sys.path.insert(0, str(sdk_dir))
    try:
        from tsfdata import TsfData
    except ImportError:
        sys.exit(
            "ERROR: Cannot import 'tsfdata'.\n"
            "Place tsfdata.py (from timsdata/examples/py/) next to this script,\n"
            "and put the compiled libtimsdata.so / timsdata.dll in a 'libs/' sub-folder.\n"
            "See the script docstring for the expected directory layout."
        )

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kw):
        return iterable


# ── MALDI pixel metadata ──────────────────────────────────────────────────────

def get_maldi_frame_coordinates(tsf: TsfData) -> dict:
    """
    Query MaldiFrameInfo from analysis.tsf and return
    {frame_id: (x_pixel, y_pixel)} using 0-based XIndexPos / YIndexPos.
    """
    cur = tsf.conn.cursor()
    cur.execute(
        "SELECT Frame, XIndexPos, YIndexPos FROM MaldiFrameInfo ORDER BY Frame"
    )
    rows = cur.fetchall()
    if not rows:
        raise RuntimeError(
            "MaldiFrameInfo table is empty — is this a MALDI dataset? "
            "Non-MALDI Bruker data does not have pixel coordinates."
        )
    return {int(frame): (int(x), int(y)) for frame, x, y in rows}


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_ion_image(
    d_folder: str,
    mz_min: float,
    mz_max: float,
    normalize: bool = False,
) -> np.ndarray:
    """
    Iterate over every MALDI frame, sum intensities within [mz_min, mz_max],
    and deposit the value into a 2-D pixel array.

    Returns
    -------
    image : np.ndarray, shape (height, width), dtype float64
    """
    with TsfData(d_folder) as tsf:
        coord_map = get_maldi_frame_coordinates(tsf)

        xs = [v[0] for v in coord_map.values()]
        ys = [v[1] for v in coord_map.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        width  = x_max - x_min + 1
        height = y_max - y_min + 1
        image  = np.zeros((height, width), dtype=np.float64)

        print(f"  Grid size : {width} x {height}  ({len(coord_map)} frames)")

        for frame_id, (x, y) in tqdm(
            coord_map.items(), desc="Extracting frames", unit="frame"
        ):
            # readLineSpectrum returns (index_array, intensity_array).
            # index_array holds float detector indices, NOT m/z values directly.
            try:
                index_arr, intensity_arr = tsf.readLineSpectrum(frame_id)
            except Exception as exc:
                print(f"  Warning: frame {frame_id} failed ({exc}), skipping.")
                continue

            if len(index_arr) == 0:
                continue

            # Convert m/z window boundaries to detector indices for this frame.
            # mzToIndex() is per-frame because the calibration can vary per spot.
            idx_lo, idx_hi = tsf.mzToIndex(frame_id, [mz_min, mz_max])

            # Guard: index increases monotonically with m/z on TSF data,
            # but clamp to be safe.
            if idx_lo > idx_hi:
                idx_lo, idx_hi = idx_hi, idx_lo

            # Select peaks whose detector index falls in [idx_lo, idx_hi].
            mask = (index_arr >= idx_lo) & (index_arr <= idx_hi)
            window_intensity = float(np.sum(intensity_arr[mask]))

            if normalize:
                total = float(np.sum(intensity_arr))
                window_intensity = (window_intensity / total) if total > 0 else 0.0

            row = y - y_min
            col = x - x_min
            image[row, col] = window_intensity

    return image


# ── Plotting ──────────────────────────────────────────────────────────────────

def save_ion_image(
    image: np.ndarray,
    mz_min: float,
    mz_max: float,
    output_path: str,
    colormap: str = "hot",
    dpi: int = 150,
    normalize: bool = False,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    nonzero = image[image > 0]
    vmax = float(np.percentile(nonzero, 99)) if nonzero.size > 0 else 1.0

    im = ax.imshow(
        image,
        cmap=colormap,
        origin="upper",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )

    intensity_label = (
        "Normalised Intensity (fraction of TIC)" if normalize
        else "Summed Intensity (a.u.)"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(intensity_label, fontsize=11)

    ax.set_title(f"Ion Image  m/z {mz_min:.4f} - {mz_max:.4f}", fontsize=13)
    ax.set_xlabel("X pixel index")
    ax.set_ylabel("Y pixel index")

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate a MALDI ion image from a Bruker .d dataset (TSF format).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",     required=True,
                   help="Path to the Bruker .d folder")
    p.add_argument("--mz_min",    required=True, type=float,
                   help="Lower m/z bound")
    p.add_argument("--mz_max",    required=True, type=float,
                   help="Upper m/z bound")
    p.add_argument("--output",    default="ion_image.png",
                   help="Output file (PNG / PDF / SVG)")
    p.add_argument("--colormap",  default="hot",
                   help="Matplotlib colormap name")
    p.add_argument("--dpi",       default=150, type=int,
                   help="Output resolution (DPI)")
    p.add_argument("--normalize", action="store_true",
                   help="Normalise each pixel intensity to its frame TIC")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.input):
        sys.exit(f"ERROR: '{args.input}' is not a directory / does not exist.")
    if args.mz_min >= args.mz_max:
        sys.exit("ERROR: --mz_min must be strictly less than --mz_max.")

    print(f"Dataset   : {args.input}")
    print(f"m/z range : {args.mz_min} - {args.mz_max}")
    if args.normalize:
        print("Mode      : TIC-normalised")

    image = extract_ion_image(
        args.input, args.mz_min, args.mz_max, normalize=args.normalize
    )

    print(f"Max pixel intensity : {image.max():.4g}")

    save_ion_image(
        image,
        args.mz_min,
        args.mz_max,
        args.output,
        colormap=args.colormap,
        dpi=args.dpi,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
