import numpy as np
import scipy.integrate as integrate

from joblib import Parallel, delayed

from oh_suppression.fibre_modes import modes, find_root_XY, lp_mode_field_2d
from oh_suppression.image_plane_fields import diffraction_limited_E_field_at_fibre_2d

def coupling_efficiency_2d(E_image, E_mode, x_1d, y_1d): 
    """
    Calculate the 2D coupling efficiency between an image-plane electric field
    and a fibre mode electric field.

    Input:
    E_image: 2D electric field of the image at the fibre input plane
    E_mode: 2D electric field of the fibre LP mode
    x_1d: 1D x-coordinate array used to make the 2D grid [m]
    y_1d: 1D y-coordinate array used to make the 2D grid [m]

    Output:
    eta: coupling efficiency
    numerator = |∫∫ E_image * conj(E_mode) dx dy|^2
    denominator = ∫∫ |E_image|^2 dx dy * ∫∫ |E_mode|^2 dx dy
    """

    # The integrand inside the integral
    overlap_integrand = E_image * np.conjugate(E_mode)

    # First integrate along the direction of x: 
    overlap_x = integrate.simpson(overlap_integrand, x=x_1d, axis=1)
    overlap = integrate.simpson(overlap_x, x=y_1d) 

    # Power is the integral of the electric field amplitude intensity. 
    # The power in each electric field are normalisation factors when we take the product of 
    # power in image field and power in mode field
    P_image_x = integrate.simpson(np.abs(E_image)**2, x=x_1d, axis=1)
    P_image = integrate.simpson(P_image_x, x=y_1d)
    P_mode_x = integrate.simpson(np.abs(E_mode)**2, x=x_1d, axis=1)
    P_mode = integrate.simpson(P_mode_x, x=y_1d)

    # Calculate the coupling efficiency
    numerator = np.abs(overlap)**2
    denominator = P_image * P_mode

    eta = numerator / denominator

    return eta, numerator, denominator


def prepare_modes_2d(
    x, y,
    a, NA, lam0,
    az_sym=True,
    mode_case="few_mode"
):
    """
    Prepare fibre mode fields for a fixed core radius.

    Parameters:
    x, y: 2D coordinate arrays at the fibre input plane [m]
    a: Core radius of the fibre [m]
    NA: Numerical aperture of the fibre
    lam0: Wavelength [m]
    az_sym: If True, only include azimuthally symmetric modes (l=0). If False, 
    include all modes up to l_max.
    mode_case:
        "smf"      -> only LP01
        "few_mode" -> all supported modes, optionally restricted by az_sym
    
    Returns value: 
    prepared_modes: A list of dictionaries containing precomputed fibre mode fields and their properties.
    Includes l, m, cutoff_V, X_root, Y_root, angular, E_mode, I_mode for each mode.
    """

    V = 2 * np.pi * a * NA / lam0

    if mode_case == "smf": # note if the maximum radius leads to V > 2.405, the fibre would support more modes, the case would be impractical
        allmodes = [[0, 1, 0.0]]
        if V > 2.405:
            print(f"Warning: V={V:.3f} > 2.405, the fibre would support more modes than just LP01, the 'smf' case may be impractical.")

    elif mode_case == "few_mode":
        allmodes = modes(V, az_sym_only=az_sym) # compute all modes supported by the fibre based on the V number

        if az_sym:
            allmodes = [mode for mode in allmodes if mode[0] == 0] # only keep azimuthally symmetric modes if az_sym is True

    else:
        raise ValueError("mode_case must be 'smf' or 'few_mode'")

    # Create a list to store the prepared modes
    prepared_modes = []
    # Loop through each mode created from modes() and compute the corresponding electric field distribution
    for mode in allmodes:
        l, m, cutoff_V = mode

        try:
            X_root, Y_root = find_root_XY(l, m, V) # find X_root and Y_root for the mode with l,m, at the given V estimated from a (not Vc)
        except ValueError:
            continue

        if l == 0:
            angular_list = ["cos"] # only one polarisaton for azimuthally symmetric modes, technically for l=0 the angular part is just a constant (cos(0)=1)
        else:
            angular_list = ["cos", "sin"] # both cos and sin polarisations for asymmetric modes

        for angular in angular_list:
            E_mode, I_mode = lp_mode_field_2d(
                x, y,
                a=a,
                l=l,
                X_root=X_root,
                Y_root=Y_root,
                angular=angular
            )

            prepared_modes.append({
                "l": l,
                "m": m,
                "cutoff_V": cutoff_V,
                "X_root": X_root,
                "Y_root": Y_root,
                "angular": angular,
                "E_mode": E_mode,
                "I_mode": I_mode
            })

    return prepared_modes


def total_eff_2d(
    prepared_modes,
    x, y, x_1d, y_1d,
    lam0, F_eff, alpha,
    decentre=None,
    E_S=1.0
):
    """
    Calculates the total 2D coupling efficiency using precomputed fibre modes.
    Only E_image changes with focal ratio F.

    Parameters:
    prepared_modes: A list of dictionaries containing precomputed fibre mode fields and their properties.
    x, y: 2D coordinate arrays at the fibre input plane [m]
    x_1d, y_1d: 1D coordinate arrays used to make the 2D grid [m]
    lam0: Wavelength [m]
    F_eff: Effective focal ratio to be optimised
    alpha: Divergence angle [rad]
    decentre: Decentre of the image field [m]
    E_S: Source electric field amplitude [V/m]
    """

    E_image, I_image = diffraction_limited_E_field_at_fibre_2d(
        x, y,
        lam0=lam0,
        F_eff=F_eff,
        alpha=alpha,
        E_S=E_S,
        decentre=decentre
    )

    # begin with zero efficiency when we haven't started accumulating eta for any modes
    total = 0.0
    mode_results = [] # array of dictionaries containing mode results

    # for each existing modes in the prepared_modes list, calculate eta at that mode
    for mode in prepared_modes:
        eta, num, den = coupling_efficiency_2d(
            E_image,
            mode["E_mode"],
            x_1d,
            y_1d
        )

        total += eta # accumulate coupling efficiency at each mode to the total coupling efficiency

        mode_results.append({
            "l": mode["l"],
            "m": mode["m"],
            "cutoff_V": mode["cutoff_V"],
            "X_root": mode["X_root"],
            "Y_root": mode["Y_root"],
            "angular": mode["angular"],
            "eta": eta
        })

    return total, mode_results


def eta_psf_vs_decentre_position_2d(
        prepared_modes, 
        x, y, x_1d, y_1d, 
        lam0, F_eff, alpha, 
        decentre_array, 
        E_S = 1.0, 
        direction_angle = 0.0
):
    """
    Calculate the coupling efficiency of a point source  as a function to its decentre relative 
    to the fibre. 

    Parameters: 
    prepared_modes: the pre-computed fibre modes containing eigenfield information for ONE core radius

    decentre_array: array of image positions relative to the fibre centre

    direction_angle: the angle of offset direction in the x-y grid plane
    Assuming angle is zero for a circularly symmetric aperature
    """

    eta_psf = [] # prepare array of eta_psf for each decentre coordinate
    mode_results_all = [] # not required but returns the fields of eigenmodes

    for r in decentre_array: # r is distance to centre
        dx = r * np.cos(direction_angle) # cos(0)=1, we are varying in the x direction
        dy = r * np.sin(direction_angle) # sin(0)=0 

        eta_total, mode_results = total_eff_2d(
            prepared_modes, 
            x, y, x_1d, y_1d, 
            lam0=lam0, 
            F_eff=F_eff, 
            alpha=alpha, 
            decentre=(dx, dy),
            E_S=E_S
        )

        eta_psf.append(eta_total)
        mode_results_all.append(mode_results)

    eta_psf = np.array(eta_psf)

    return eta_psf, mode_results_all


def eta_non_opt_vs_core_radius(
    core_radius_array,
    x, y, x_1d, y_1d,
    lam0,
    NA,
    F_eff=None,
    alpha=0.0,
    decentre=None,
    E_S=1.0,
    mode_case="few_mode",
    az_sym=True,
    n_jobs=1,
    store_mode_results=True,
): 
    """
    Calculate the fixed-focal-ratio point-source coupling efficiency
    for a range of fibre core radius values.

    No focal-ratio optimisation is performed.
    The same fixed F_eff is used for every core radius.

    The coupling efficiency is calculated from the 2D electric-field
    overlap integral through total_eff_2d().

    Parameters
    ----------
    core_radius_array : array
        Fibre core radius values [m].

    x, y, x_1d, y_1d : arrays
        2D and 1D coordinate grids at the fibre input plane [m].

    lam0 : float
        Wavelength [m].

    NA : float
        Fibre numerical aperture.

    F_eff : float or None
        Fixed effective fibre focal ratio used for all core radii.
        If None, calculate from NA using:
            F_eff = 1 / (2 tan(arcsin(NA)))

    alpha : float
        Central obstruction ratio.

    decentre : tuple, list, or None
        Point-source decentre position at the fibre input plane.
        If None, the point source is centred.

    E_S : float
        Electric-field amplitude of the point source.

    mode_case : str
        Fibre mode case, for example "smf" or "few_mode".

    az_sym : bool
        Whether to include only azimuthally symmetric modes.

    n_jobs : int
        Number of parallel workers. Use n_jobs=1 for serial.

    store_mode_results : bool
        If True, store the per-mode coupling results for every core radius.
        If False, only return total coupling efficiency.

    Returns
    -------
    eta_total_non_opt_array : 1D numpy array
        Total point-source coupling efficiency for each core radius.

    mode_results_non_opt_array : list
        Per-mode coupling results for each core radius.
        If store_mode_results=False, this is None.

    V_array : 1D numpy array
        Normalised frequency values corresponding to core_radius_array.
    """

    if F_eff is None:
        F_eff = 1 / (2 * np.tan(np.arcsin(NA)))

    core_radius_array = np.asarray(core_radius_array)

    V_array = 2 * np.pi * core_radius_array * NA / lam0

    n_total = len(core_radius_array)

    def run_one_radius(a):
        prepared_modes = prepare_modes_2d(
            x, y,
            a=a,
            NA=NA,
            lam0=lam0,
            az_sym=az_sym,
            mode_case=mode_case
        )

        eta_tot, mode_results = total_eff_2d(
            prepared_modes=prepared_modes,
            x=x,
            y=y,
            x_1d=x_1d,
            y_1d=y_1d,
            lam0=lam0,
            F_eff=F_eff,
            alpha=alpha,
            decentre=decentre,
            E_S=E_S
        )

        if store_mode_results:
            return eta_tot, mode_results
        else:
            return eta_tot, None

    if n_jobs == 1:
        eta_total_non_opt_array = []
        mode_results_non_opt_array = [] if store_mode_results else None

        checkpoints = {
            max(1, int(np.ceil(n_total * p / 10)))
            for p in range(1, 11)
        }

        for i, a in enumerate(core_radius_array, start=1):
            eta_tot, mode_results = run_one_radius(a)

            eta_total_non_opt_array.append(eta_tot)

            if store_mode_results:
                mode_results_non_opt_array.append(mode_results)

            if i in checkpoints:
                percent = int(round(100 * i / n_total))
                print(f"{percent}% complete ({i}/{n_total})")

    else:
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_one_radius)(a)
            for a in core_radius_array
        )

        eta_total_non_opt_array = [row[0] for row in results]

        if store_mode_results:
            mode_results_non_opt_array = [row[1] for row in results]
        else:
            mode_results_non_opt_array = None

    eta_total_non_opt_array = np.asarray(eta_total_non_opt_array)

    return eta_total_non_opt_array, mode_results_non_opt_array, V_array