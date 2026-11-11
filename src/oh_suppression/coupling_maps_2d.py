"""Utilities for calculating and saving 2D PSF-decentre fibre coupling maps.

This module contains the production functions extracted from the notebook
``test function coupling map(1).ipynb``.

It assumes the core fibre-mode and point-source coupling functions already
exist in the ``oh_suppression`` package.
"""

from __future__ import annotations

import os
import time

import joblib
import numpy as np
from joblib import Parallel, delayed
import pandas as pd

from oh_suppression.point_source_coupling import prepare_modes_2d
from oh_suppression.point_source_coupling import total_eff_2d
from oh_suppression.fibre_modes import cutoff


# __all__ = [
#     "eta_psf_2d_decentre_quadrant",
#     "mirror_quadrant_map",
#     "psf_2d_decentre_map",
#     "select_core_radius_bin",
#     "choose_bin_decentre_resolution",
#     "compute_and_save_coupling_maps_for_V_bin",
#     "load_coupling_map_results",
# ]


def eta_psf_2d_decentre_quadrant(
    prepared_modes,
    x,
    y,
    x_1d,
    y_1d,
    lam0,
    F_eff,
    alpha,
    decentre_x_quadrant,
    decentre_y_quadrant,
    E_S=1.0,
    return_mode_maps=True,
    n_jobs=1,
    backend="threading",
    batch_size="auto",
):
    """Calculate coupling efficiency over the positive PSF-decentre quadrant."""

    decentre_x_quadrant = np.asarray(decentre_x_quadrant)
    decentre_y_quadrant = np.asarray(decentre_y_quadrant)

    if (
        np.any(decentre_x_quadrant < 0)
        or np.any(decentre_y_quadrant < 0)
    ):
        raise ValueError("Quadrant decentre coordinates must be non-negative.")

    if (
        not np.isclose(decentre_x_quadrant[0], 0.0)
        or not np.isclose(decentre_y_quadrant[0], 0.0)
    ):
        raise ValueError("Quadrant decentre coordinates must begin at zero.")

    nx = len(decentre_x_quadrant)
    ny = len(decentre_y_quadrant)

    if return_mode_maps:
        mode_keys = [
            (mode["l"], mode["m"], mode["angular"])
            for mode in prepared_modes
        ]
    else:
        mode_keys = None

    def calculate_row(dy):
        eta_row = np.zeros(nx)

        if return_mode_maps:
            mode_rows = {
                key: np.zeros(nx)
                for key in mode_keys
            }
        else:
            mode_rows = None

        for ix, dx in enumerate(decentre_x_quadrant):
            eta_total, mode_results = total_eff_2d(
                prepared_modes=prepared_modes,
                x=x,
                y=y,
                x_1d=x_1d,
                y_1d=y_1d,
                lam0=lam0,
                F_eff=F_eff,
                alpha=alpha,
                decentre=(dx, dy),
                E_S=E_S,
            )

            eta_row[ix] = eta_total

            if return_mode_maps:
                for result in mode_results:
                    key = (
                        result["l"],
                        result["m"],
                        result["angular"],
                    )
                    mode_rows[key][ix] = result["eta"]

        return eta_row, mode_rows

    if n_jobs == 1:
        row_results = []

        for iy, dy in enumerate(decentre_y_quadrant):
            row_results.append(calculate_row(dy))

            print(
                f"{iy + 1}/{ny} quadrant rows complete "
                f"({100 * (iy + 1) / ny:.1f}%)"
            )
    else:
        row_results = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            batch_size=batch_size,
            verbose=10,
        )(
            delayed(calculate_row)(dy)
            for dy in decentre_y_quadrant
        )

    eta_total_quadrant = np.array([
        result[0]
        for result in row_results
    ])

    if return_mode_maps:
        mode_eta_quadrants = {
            key: np.array([
                result[1][key]
                for result in row_results
            ])
            for key in mode_keys
        }
    else:
        mode_eta_quadrants = None

    return eta_total_quadrant, mode_eta_quadrants


def mirror_quadrant_map(quadrant_map):
    """Mirror a positive-quadrant map into the full x-y plane.

    The input includes x=0 and y=0. The zero row and zero column are not
    duplicated, so an N x N quadrant becomes a (2N-1) x (2N-1) full map.
    """

    negative_x = np.flip(
        quadrant_map[:, 1:],
        axis=1,
    )

    full_x = np.concatenate(
        (negative_x, quadrant_map),
        axis=1,
    )

    negative_y = np.flip(
        full_x[1:, :],
        axis=0,
    )

    return np.concatenate(
        (negative_y, full_x),
        axis=0,
    )


def psf_2d_decentre_map(
    a,
    x,
    y,
    x_1d,
    y_1d,
    lam0,
    NA,
    l_max=10,
    m_max=5,
    F_eff=None,
    alpha=0.0,
    n_decentre_quadrant=51,
    E_S=1.0,
    mode_case="few_mode",
    n_jobs=1,
    backend="threading",
    batch_size="auto",
    expected_LP_families=None,
    expected_prepared_components=None,
):
    """Calculate full 2D coupling-efficiency maps for one fibre core radius."""

    if F_eff is None:
        F_eff = 1 / (
            2 * np.tan(np.arcsin(NA))
        )

    prepared_modes = prepare_modes_2d(
        x,
        y,
        a=a,
        NA=NA,
        lam0=lam0,
        az_sym=False,
        mode_case=mode_case,
        l_max=l_max,
        m_max=m_max,
    )

    decentre_x_quadrant = np.linspace(
        0.0,
        a,
        n_decentre_quadrant,
    )

    decentre_y_quadrant = np.linspace(
        0.0,
        a,
        n_decentre_quadrant,
    )

    (
        eta_total_quadrant,
        mode_eta_quadrants,
    ) = eta_psf_2d_decentre_quadrant(
        prepared_modes=prepared_modes,
        x=x,
        y=y,
        x_1d=x_1d,
        y_1d=y_1d,
        lam0=lam0,
        F_eff=F_eff,
        alpha=alpha,
        decentre_x_quadrant=decentre_x_quadrant,
        decentre_y_quadrant=decentre_y_quadrant,
        E_S=E_S,
        return_mode_maps=True,
        n_jobs=n_jobs,
        backend=backend,
        batch_size=batch_size,
    )

    eta_total_map = mirror_quadrant_map(
        eta_total_quadrant
    )

    mode_eta_maps = {
        key: mirror_quadrant_map(quadrant_map)
        for key, quadrant_map in mode_eta_quadrants.items()
    }

    decentre_x_full = np.concatenate(
        (
            -decentre_x_quadrant[1:][::-1],
            decentre_x_quadrant,
        )
    )

    decentre_y_full = np.concatenate(
        (
            -decentre_y_quadrant[1:][::-1],
            decentre_y_quadrant,
        )
    )

    V = 2 * np.pi * a * NA / lam0

    return {
        "a": a,
        "V": V,
        "F_eff": F_eff,
        "n_decentre_quadrant": n_decentre_quadrant,
        "decentre_x_1d": decentre_x_full,
        "decentre_y_1d": decentre_y_full,
        "eta_total_map": eta_total_map,
        "mode_eta_maps": mode_eta_maps,
    }

def make_cutoff_table(
        l_max=6,
        m_max=5,
        V_max=None
):
    rows = []

    for l in range(l_max + 1):
        for m in range(1, m_max + 1):

            Vc = cutoff(l, m)

            if V_max is not None and Vc > V_max:
                continue

            rows.append({
                "l": l,
                "m": m,
                "Mode": f"LP{l}{m}",
                "V_cutoff": Vc,
                "n_components": 1 if l == 0 else 2
            })

    df = pd.DataFrame(rows)

    df = df.sort_values(
        ["V_cutoff", "l", "m"]
    ).reset_index(drop=True)

    return df

def make_mode_count_table(
    l_max=10,
    m_max=5,
    V_limit=12.6,
    decimals=4,
):
    """
    Construct V-number bins for coupling-map calculations.

    The final bin extends to the first real LP-mode cutoff
    frequency greater than V_limit.
    """

    # ---------------------------------------------------------
    # Build a cutoff table extending beyond V_limit
    # ---------------------------------------------------------

    # Use a safety margin so that at least one cutoff above
    # V_limit is included.
    cutoff_table = make_cutoff_table(
        l_max=l_max,
        m_max=m_max,
        V_max=V_limit + 5.0,
    ).copy()

    # ---------------------------------------------------------
    # Group modes sharing the same cutoff
    # ---------------------------------------------------------

    cutoff_table["Vc_group"] = (
        cutoff_table["V_cutoff"]
        .round(decimals)
    )

    unique_Vc = np.sort(
        cutoff_table["Vc_group"].unique()
    )

    # ---------------------------------------------------------
    # Find first cutoff strictly above V_limit
    # ---------------------------------------------------------

    above_limit = unique_Vc[
        unique_Vc > V_limit
    ]

    if len(above_limit) == 0:
        raise ValueError(
            "No cutoff frequency found above V_limit. "
            "Increase the V range used to generate the cutoff table."
        )

    final_V_max = above_limit[0]

    rows = []

    cumulative_families = 0
    cumulative_components = 0

    # ---------------------------------------------------------
    # Construct bins
    # ---------------------------------------------------------

    for i in range(len(unique_Vc) - 1):

        V_min = unique_Vc[i]
        V_max_bin = unique_Vc[i + 1]

        # Stop once we have reached the first cutoff above V_limit
        if V_min >= final_V_max:
            break

        group = cutoff_table[
            cutoff_table["Vc_group"] == V_min
        ]

        modes_added = list(
            group["Mode"]
        )

        families_added = len(
            group
        )

        components_added = int(
            group["n_components"].sum()
        )

        cumulative_families += (
            families_added
        )

        cumulative_components += (
            components_added
        )

        rows.append({
            "V_min":
                float(V_min),

            "V_max":
                float(V_max_bin),

            "Modes_added":
                ", ".join(modes_added),

            "Families_added":
                families_added,

            "Components_added":
                components_added,

            "Total_LP_families":
                cumulative_families,

            "Total_prepared_components":
                cumulative_components,
        })

        # This is the final required bin
        if V_max_bin >= final_V_max:
            break

    # ---------------------------------------------------------
    # Convert to DataFrame
    # ---------------------------------------------------------

    mode_count_table = pd.DataFrame(
        rows
    )

    mode_count_table.insert(
        0,
        "file",
        np.arange(
            len(mode_count_table)
        )
    )

    return mode_count_table

def select_core_radius_bin(
    bin_id,
    core_radius_array,
    V_array,
    mode_count_table,
):
    """Select fibre core radii belonging to one V-number bin."""

    row = mode_count_table.loc[
        mode_count_table["file"] == bin_id
    ]

    if len(row) == 0:
        raise ValueError(f"Bin {bin_id} not found.")

    row = row.iloc[0]
    V_min = float(row["V_min"])
    V_max = float(row["V_max"])

    mask = (
        (V_array >= V_min)
        & (V_array < V_max)
    )

    selected_radii = np.asarray(core_radius_array)[mask]
    selected_V = np.asarray(V_array)[mask]

    return selected_radii, selected_V, V_min, V_max


def choose_bin_decentre_resolution(
    bin_id,
    mode_count_table,
    core_radius_array,
    NA,
    lam0,
    x_1d,
    n_fixed=51,
):
    """Choose one quadrant size for a complete V-number bin.

    Rules
    -----
    * Bin 0: return None because LP01 can use the radial method.
    * Bins 1-2: use ``n_fixed`` (default 51).
    * Bin 3 onward: choose the size from the largest actual radius in the
      bin so that the decentre-map spacing is no coarser than the original
      2D field-grid spacing.

    The quadrant size itself may be even or odd. Because the quadrant starts
    at zero and mirroring excludes duplicate zero rows/columns, the returned
    full map always has an odd size: ``2*n_quadrant - 1``.
    """

    if bin_id == 0:
        return None

    if bin_id in (1, 2):
        return n_fixed

    core_radius_array = np.asarray(core_radius_array)

    V_array = (
        2 * np.pi
        * core_radius_array
        * NA
        / lam0
    )

    selected_radii, _, _, _ = select_core_radius_bin(
        bin_id=bin_id,
        core_radius_array=core_radius_array,
        V_array=V_array,
        mode_count_table=mode_count_table,
    )

    if len(selected_radii) == 0:
        return None

    a_max = selected_radii.max()

    field_spacing = np.abs(
        x_1d[1] - x_1d[0]
    )

    n_quadrant = (
        int(np.ceil(a_max / field_spacing))
        + 1
    )

    return n_quadrant


def compute_and_save_coupling_maps_for_V_bin(
    bin_id,
    mode_count_table,
    core_radius_array,
    x,
    y,
    x_1d,
    y_1d,
    NA,
    lam0,
    l_max=10,
    m_max=5,
    alpha=0.0,
    F_eff=None,
    n_decentre_quadrant=None,
    mode_case="few_mode",
    E_S=1.0,
    n_jobs=1,
    backend="threading",
    batch_size="auto",
    output_dir="coupling_results",
    compress=0,
):
    """
    Calculate and save all 2D coupling maps for one V-number bin.

    Parameters
    ----------
    l_max : int, default=10
        Maximum azimuthal LP-mode order considered.

    m_max : int, default=5
        Maximum radial LP-mode order considered.

        The defaults l_max=10 and m_max=5 are intended for the
        current parameter range:
            a <= 30 microns
            NA = 0.1
            lam0 = 1.5 microns

        For larger NA, larger core radii, or shorter wavelength,
        increase these limits.

    Notes
    -----
    The expected number of LP families and prepared cos/sin
    components is taken from mode_count_table and checked against
    the modes actually prepared by prepare_modes_2d().

    If n_decentre_quadrant is None, the resolution is selected using
    choose_bin_decentre_resolution().

    Bin 0 is not calculated here because it uses the radially
    symmetric LP01 method.
    """

    core_radius_array = np.asarray(
        core_radius_array
    )

    # ---------------------------------------------------------
    # Bin 0 uses radial symmetry
    # ---------------------------------------------------------

    if bin_id == 0:
        print(
            "Bin 0 uses the radially symmetric LP01 method; "
            "no 2D maps calculated."
        )
        return None, []

    # ---------------------------------------------------------
    # Calculate V for supplied radii
    # ---------------------------------------------------------

    V_array = (
        2 * np.pi
        * core_radius_array
        * NA
        / lam0
    )

    # ---------------------------------------------------------
    # Get row corresponding to this mode-count bin
    # ---------------------------------------------------------

    row = mode_count_table.loc[
        mode_count_table["file"] == bin_id
    ]

    if len(row) == 0:
        raise ValueError(
            f"bin_id={bin_id} not found in mode_count_table."
        )

    row = row.iloc[0]

    # Expected mode numbers from the table
    expected_LP_families = int(
        row["Total_LP_families"]
    )

    expected_prepared_components = int(
        row["Total_prepared_components"]
    )

    # ---------------------------------------------------------
    # Select radii belonging to this bin
    # ---------------------------------------------------------

    (
        selected_radii,
        selected_V,
        V_min,
        V_max,
    ) = select_core_radius_bin(
        bin_id=bin_id,
        core_radius_array=core_radius_array,
        V_array=V_array,
        mode_count_table=mode_count_table,
    )

    n_total = len(
        selected_radii
    )

    if n_total == 0:

        print(
            f"No radii found in bin {bin_id}: "
            f"{V_min:.4f} <= V < {V_max:.4f}"
        )

        return None, []

    # ---------------------------------------------------------
    # Choose map resolution
    # ---------------------------------------------------------

    if n_decentre_quadrant is None:

        n_decentre_quadrant = (
            choose_bin_decentre_resolution(
                bin_id=bin_id,
                mode_count_table=mode_count_table,
                core_radius_array=core_radius_array,
                NA=NA,
                lam0=lam0,
                x_1d=x_1d,
                n_fixed=51,
            )
        )

    # ---------------------------------------------------------
    # Effective focal ratio
    # ---------------------------------------------------------

    if F_eff is None:

        F_eff = (
            1
            / (
                2
                * np.tan(
                    np.arcsin(NA)
                )
            )
        )

    # ---------------------------------------------------------
    # Print bin information
    # ---------------------------------------------------------

    print("=" * 70)

    print(
        f"Bin {bin_id}"
    )

    print(
        f"V range: "
        f"{V_min:.4f} <= V < {V_max:.4f}"
    )

    print(
        f"Number of radii: {n_total}"
    )

    print(
        f"Radius range: "
        f"{selected_radii.min() * 1e6:.3f} "
        f"to "
        f"{selected_radii.max() * 1e6:.3f} µm"
    )

    print(
        f"Mode search limits: "
        f"l_max={l_max}, "
        f"m_max={m_max}"
    )

    print(
        f"Expected LP families: "
        f"{expected_LP_families}"
    )

    print(
        f"Expected prepared components: "
        f"{expected_prepared_components}"
    )

    print(
        f"n_decentre_quadrant = "
        f"{n_decentre_quadrant}"
    )

    print(
        f"full map size = "
        f"{2 * n_decentre_quadrant - 1} x "
        f"{2 * n_decentre_quadrant - 1}"
    )

    print(
        f"F_eff = {F_eff:.4f}"
    )

    print("=" * 70)

    # ---------------------------------------------------------
    # Calculate every radius in the bin
    # ---------------------------------------------------------

    results_all_radius = []

    start_time = time.time()

    for i, (a, V) in enumerate(
        zip(
            selected_radii,
            selected_V
        )
    ):

        quadrant_spacing = (
            a
            / (
                n_decentre_quadrant - 1
            )
        )

        print(
            f"\nCore radius = "
            f"{a * 1e6:.3f} µm "
            f"({i + 1}/{n_total})"
        )

        print(
            f"V = {V:.4f}"
        )

        print(
            f"Decentre spacing = "
            f"{quadrant_spacing * 1e6:.3f} µm"
        )

        # -----------------------------------------------------
        # Calculate one radius
        # -----------------------------------------------------

        result = psf_2d_decentre_map(
            a=a,
            x=x,
            y=y,
            x_1d=x_1d,
            y_1d=y_1d,
            lam0=lam0,
            NA=NA,

            l_max=l_max,
            m_max=m_max,

            F_eff=F_eff,
            alpha=alpha,

            n_decentre_quadrant=
                n_decentre_quadrant,

            E_S=E_S,
            mode_case=mode_case,

            n_jobs=n_jobs,
            backend=backend,
            batch_size=batch_size,

            expected_LP_families=
                expected_LP_families,

            expected_prepared_components=
                expected_prepared_components,
        )

        results_all_radius.append(
            result
        )

        # -----------------------------------------------------
        # Progress
        # -----------------------------------------------------

        elapsed = (
            time.time()
            - start_time
        )

        minutes = int(
            elapsed // 60
        )

        seconds = int(
            elapsed % 60
        )

        print(
            f"{100 * (i + 1) / n_total:.1f}% complete | "
            f"elapsed: "
            f"{minutes} min "
            f"{seconds:02d} sec"
        )

    # ---------------------------------------------------------
    # Package results
    # ---------------------------------------------------------

    output = {
        "bin_id": bin_id,

        "V_min": V_min,
        "V_max": V_max,

        "core_radius_array":
            selected_radii,

        "V_array":
            selected_V,

        "NA": NA,
        "lam0": lam0,

        "l_max": l_max,
        "m_max": m_max,

        "expected_LP_families":
            expected_LP_families,

        "expected_prepared_components":
            expected_prepared_components,

        "alpha": alpha,
        "F_eff": F_eff,

        "n_decentre_quadrant":
            n_decentre_quadrant,

        "full_map_size":
            2 * n_decentre_quadrant - 1,

        "mode_case":
            mode_case,

        "E_S":
            E_S,

        "backend":
            backend,

        "batch_size":
            batch_size,

        "results":
            results_all_radius,
    }

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    filename = (
        f"coupling_maps_"
        f"bin_{bin_id:02d}_"
        f"V_{V_min:.4f}_"
        f"to_{V_max:.4f}.joblib"
    )

    filepath = os.path.join(
        output_dir,
        filename,
    )

    joblib.dump(
        output,
        filepath,
        compress=compress,
    )

    print(
        f"\nSaved bin {bin_id} to:"
    )

    print(
        filepath
    )

    return (
        filepath,
        results_all_radius
    )


def load_coupling_map_results(filepath):
    """Load one saved coupling-map V-bin file."""

    loaded = joblib.load(filepath)
    results = loaded["results"]

    print(
        f"Bin {loaded['bin_id']}: "
        f"{loaded['V_min']:.4f} <= V < "
        f"{loaded['V_max']:.4f}"
    )
    print(f"Number of radii = {len(results)}")
    print("\nIncluded radii:")

    for i, result in enumerate(results, start=1):
        print(
            f"{i}: "
            f"a = {result['a'] * 1e6:.3f} µm, "
            f"V = {result['V']:.4f}, "
            f"modes = {len(result['mode_eta_maps'])}"
        )

    return loaded, results
