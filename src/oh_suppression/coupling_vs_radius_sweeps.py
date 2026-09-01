import numpy as np
import scipy.integrate as integrate
from joblib import Parallel, delayed

from oh_suppression.image_plane_fields import diffraction_limited_E_field_at_fibre_2d
from oh_suppression.point_source_coupling import prepare_modes_2d, total_eff_2d, eta_psf_vs_decentre_position_2d

def eta_background_from_psf_weights(decentre_array, eta_psf_array): 
    """
    Calculate the background coupling efficiency (eta_background) from the PSF coupling efficiency 
    (eta_psf(r)) as a function of decentre r. 

    The background coupling efficiency is calculated as the weighted average of the PSF coupling 
    efficiency over the image field, where the weights are given by the intensity distribution 
    of the image field (I_ext(r/f)). 

    eta_ext(background) = ∫ eta_psf(r) * I_ext(r/f) dA / ∫ I_ext(r/f) dA, where A is the area of the image field.

    For the background, the image field would not be optimised via focal ratio, and the image field 
    of a decentred psf here would use the optimal F of the centred case. 

    Parameters: 
    decentre_array: 1D array of decentre values (r) in meters
    eta_psf_array: 1D array of PSF coupling efficiency values corresponding to each decentre value in decentre_array


    """
    # numerator = ∫ eta_psf(r) * I_ext(r/f) dA, where A=pi * r^2, so dA = 2 * pi * r dr  
    # I_ext(r^arrow) = I_0 for a completely uniform field, where I_0 is a constant that cancels out in the ratio, so we can ignore it in the calculation. 
    I_0=1.0
    numerator = integrate.simpson(eta_psf_array * I_0 * 2 * np.pi * decentre_array, x=decentre_array)
    # denominator = ∫ I_ext(r/f) dA, where A=pi * r^2, so dA = 2 * pi * r dr
    denominator = integrate.simpson(I_0 * 2 * np.pi * decentre_array, x=decentre_array)

    eta_background = numerator / denominator

    return eta_background


def background_no_opt_efficiency_at_one_radius_2d(
        a,
        x, y, x_1d, y_1d, 
        NA, lam0, 
        alpha=0.0, 
        az_sym=True, 
        mode_case="smf", 
        decentre_max=None, 
        n_decentre=50, E_S=1.0, 
        n_F_grid=60, 
        # F_opt_input=None
        F_eff=None
):
    """
    For one fibre core radius a: 
        1. Prepare fibre modes
        2. Optimise F_eff for centred point source
        3. Calculate eta_psf(d)
        4. Integrate eta_psf(d) to get flat-background efficiency

    Parameters: 
    decentre_max: the maximum decentre radius, set to None, but equals to the cladding radius
    n_decentre: the number of decentre points to consider
    E_S: the electric field amplitude of the point source
    n_F_grid: the number of grid points for the F_eff calculation
    F_opt_input: 

    """

    if F_eff is None:
        F_eff = 1 / (2 * np.tan(np.arcsin(NA)))

    V = 2 * np.pi * a * NA / lam0

    prepared_modes = prepare_modes_2d(
        x, y, 
        a=a, 
        NA=NA, 
        lam0=lam0,
        az_sym=az_sym, 
        mode_case=mode_case
    )

    eta_centre, mode_results_centred = total_eff_2d(
        prepared_modes,
        x=x, y=y, x_1d=x_1d, y_1d=y_1d,
        lam0=lam0,
        F_eff=F_eff,
        alpha=alpha,
        decentre=None,
        E_S=E_S
    )

    E_image_opt, I_image_opt = diffraction_limited_E_field_at_fibre_2d(
        x, y,
        lam0=lam0,
        F_eff=F_eff,
        alpha=alpha,
        E_S=E_S,
        decentre=None
    )

    # Choose decentre_max to be the cladding radius if not specified
    if decentre_max is None: 
        r_max = 125e-6 # in meters, this is the maximum radius of the grid we prepared for the field calculations, so we can't go beyond this
        decentre_max = r_max # cladding radius 

    elif decentre_max == "core_radius": 
        decentre_max = a # core radius

    decentre_array = np.linspace(0, decentre_max, n_decentre)

    # Calculate eta_psf(d) for the range of decentre values
    eta_psf,  mode_results_decentre = eta_psf_vs_decentre_position_2d(
        prepared_modes, 
        x, y, x_1d, y_1d,
        lam0=lam0, F_eff=F_eff, alpha=alpha,
        decentre_array=decentre_array,
        E_S=E_S
    )

    # Integrate over the flat background, assume I_background is unitorm so I_ext(r/f) = I_0, 
    # where I_0 is a constant that cancels out in the ratio, so we can ignore it in the calculation.
    eta_background = eta_background_from_psf_weights(decentre_array, eta_psf)
    
    return {
        "a": a,
        "V": V,
        "F_eff": F_eff,
        "eta_centre": eta_centre,
        "eta_background": eta_background,
        "decentre_array": decentre_array,
        "eta_psf": eta_psf,
        "mode_results_centred": mode_results_centred,
        "mode_results_decentre": mode_results_decentre,
        "E_image_opt": E_image_opt,
        "I_image_opt": I_image_opt
    }


def background_efficiency_vs_core_radius_2d(
        core_radius_array, 
        x, y, x_1d, y_1d,
        NA, lam0,
        alpha=0.0,
        az_sym=True,
        mode_case="smf",
        decentre_max=None,
        n_decentre=50, 
        E_S=1.0, 
        n_F_grid=60,
        n_jobs=1, 
        F_eff=None
        # F_opt_array = None
):
    """Loop over core radius and calculate: 
        - centred point source efficiency at optimal F
        - background efficiency by the weighted average of a number of decentred point source efficiencies

    Parameters:
    core_radius_array: array of core radius values to process
    x, y, x_1d, y_1d: grid points for the field calculations
    NA: numerical aperture
    lam0: wavelength
    alpha: polarization angle
    az_sym: whether to use azimuthal symmetry
    mode_case: case for mode calculation
    decentre_max: maximum decentre radius
    n_decentre: number of decentre points
    E_S: electric field amplitude of the point source
    n_F_grid: number of grid points for F_eff calculation
    n_jobs: number of parallel workers. Use 1 for serial, >1 for parallel.
    F_eff: pre-computed F_eff values (optional)
    Return: 
    results: list of dictionaries containing the background efficiency and other related 
    information for each core radius in the range of core_radius_array
    """
    # Count how many individual core radiuses are computed
    n_total = len(core_radius_array)
    if F_eff is None:
        F_eff = 1 / (2 * np.tan(np.arcsin(NA)))
    # if F_opt_array is not None and len(F_opt_array) == n_total:
    #     F_opt_array = np.asarray(F_opt_array)

    def run_one_radius(i, a):
        # if F_opt_array is None: 
        #     F_opt_input = None 
        # else: 
        #     F_opt_input = F_opt_array[i]

        row = background_no_opt_efficiency_at_one_radius_2d(
            a=a, 
            x=x, y=y, x_1d=x_1d, y_1d=y_1d,
            NA=NA, 
            lam0=lam0,
            alpha=alpha,
            az_sym=az_sym, 
            mode_case=mode_case,
            decentre_max=decentre_max, 
            n_decentre=n_decentre, 
            E_S=E_S,
            n_F_grid=n_F_grid, 
            F_eff=F_eff # remember to pass the F_opt_input for each radius if provided
        )
        return row
    



    if n_jobs == 1:
        results = []

        checkpoints = {
            max(1, int(np.ceil(n_total * p / 10))) 
            for p in range(1, 11)
        }

        for i, a in enumerate(core_radius_array, start=0):
            results.append(run_one_radius(i,a))

            step=i + 1 # step is the number of core radiuses we have processed so far, starting from 1

            if i in checkpoints:
                percent = int(round(100 * step / n_total))
                print(f"{percent}% complete ({step}/{n_total})")

    else:
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_one_radius)(i,a)
            for i, a in enumerate(core_radius_array, start=0)
        )

    return results

