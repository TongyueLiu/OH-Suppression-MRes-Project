import numpy as np
import scipy.integrate as integrate
from joblib import Parallel, delayed

from oh_suppression.image_plane_fields import diffraction_limited_E_field_at_fibre_2d
from oh_suppression.point_source_coupling import prepare_modes_2d, total_eff_2d, eta_psf_vs_decentre_position_2d

def radial_efficiency_to_2d(decentre_array, eta_point_source, x, y, fibre_x=0.0, fibre_y=0.0):
    """
    Convert a radial efficiency profile to a 2D efficiency map.

    Parameters:
    decentre_array: 1D array of radial distances from the fibre centre [m]
    eta_point_source: 1D array of coupling efficiencies corresponding to decentre_array
    x, y: 2D coordinate arrays at the fibre input plane [m]
    fibre_x, fibre_y: coordinates of the fibre centre in the x-y grid [m]

    Returns:
    eta_2d: 2D array of coupling efficiencies at each point in the x-y grid
    """

    # Calculate the radial distance from the fibre centre for each point in the grid
    R = np.sqrt((x - fibre_x)**2 + (y - fibre_y)**2)

    # Interpolate the radial efficiency profile onto the 2D grid
    eta_map_2d = np.interp(R.ravel(), decentre_array, eta_point_source, left=eta_point_source[0], right=eta_point_source[-1]).reshape(R.shape)

    return eta_map_2d


def eta_extended_source_2d(
    eta_map_2d,
    galaxy_brightness_2d,
    x_1d,
    y_1d,
):
    """
    Calculate the coupling efficiency of an incoherent extended source
    into a fibre.

    Parameters
    ----------
    eta_map_2d : 2D array
        Point-source coupling efficiency evaluated across the x-y grid.

    galaxy_brightness_2d : 2D array
        Linear galaxy brightness distribution evaluated on the same grid.

    x_1d, y_1d : 1D arrays
        Coordinates of the x and y grid axes.

    Returns
    -------
    eta_extended : float
        Brightness-weighted coupling efficiency of the extended source.
    """

    weighted_function = (
        eta_map_2d
        * galaxy_brightness_2d
    )

    # First integrate along x, then along y
    weighted_sum = integrate.simpson(
        integrate.simpson(
            weighted_function,
            x=x_1d,
            axis=1,
        ),
        x=y_1d,
    )

    total_brightness = integrate.simpson(
        integrate.simpson(
            galaxy_brightness_2d,
            x=x_1d,
            axis=1,
        ),
        x=y_1d,
    )

    eta_extended = weighted_sum / total_brightness

    return eta_extended


def extended_source_no_opt_efficiency_at_one_radius_2d(
        a,
        x, y, x_1d, y_1d, 
        NA, lam0, I_source, 
        alpha=0.0, 
        az_sym=True, 
        mode_case="smf", 
        decentre_max=None, 
        n_decentre=50, E_S=1.0, 
        n_F_grid=60, 
        F_eff=None,
):
    """
    For one fibre core radius a: 
        1. Prepare fibre modes
        2. Calculate eta_psf(d)
        3. Integrate eta_psf(d) to get coupling effieciency for an extended source with a given brightness distribution

    Parameters:
    decentre_max: the maximum decentre radius, set to None, but equals to the cladding radius
    n_decentre: the number of decentre points to consider
    E_S: the electric field amplitude of the point source
    n_F_grid: the number of grid points for the F_eff calculation
    I_source: the intensity distribution of the extended source, 2d numpy array, same shape as x and y grids

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

    # Integrate over the extended source brightness distribution to get the coupling efficiency for the extended source
    eta_map_2d=radial_efficiency_to_2d(decentre_array, eta_psf, x, y)

    eta_extended_source = eta_extended_source_2d(eta_map_2d=eta_map_2d,
        galaxy_brightness_2d=I_source, x_1d=x_1d, y_1d=y_1d)
    
    return {
        "a": a,
        "V": V,
        "F_eff": F_eff,
        "eta_centre": eta_centre,
        "eta_extended_source": eta_extended_source,
        "decentre_array": decentre_array,
        "eta_psf": eta_psf,
        "mode_results_centred": mode_results_centred,
        "mode_results_decentre": mode_results_decentre,
        "E_image_opt": E_image_opt,
        "I_image_opt": I_image_opt
    }


def extended_source_efficiency_vs_core_radius_2d(
        core_radius_array, 
        x, y, x_1d, y_1d,
        NA, lam0, I_source,
        alpha=0.0,
        az_sym=True,
        mode_case="smf",
        decentre_max=None,
        n_decentre=50, 
        E_S=1.0, 
        n_F_grid=60,
        F_eff=None,
        n_jobs=1
):
    """
    Loop over core radius and calculate: 
        - centred point source efficiency at optimal F
        - extended source efficiency by the weighted average of a number of decentred point source efficiencies
    
    Parameters: 

    core_radius_array: array of core radius values to process
    x, y, x_1d, y_1d: grid points for the field calculations
    NA: numerical aperture 
    lam0: wavelength
    alpha: central obstruction ratio
    az_sym: whether to use azimuthal symmetry
    mode_case: case for mode calculation
    decentre_max: maximum decentre radius
    n_decentre: number of decentre points
    E_S: electric field amplitude of the point source
    n_F_grid: number of grid points for F_eff calculation
    n_jobs: number of parallel workers. Use 1 for serial, >1 for parallel.

    """

    if F_eff is None:
        F_eff = 1 / (2 * np.tan(np.arcsin(NA)))

    n_total = len(core_radius_array)

    # if decentre_max is None:
    #     angular_extent = 1.30/2 # arcsec, which is half of the FOV of a 125e-6m light bucket at the Coudé focus, based on the plate scale calculated above.
    #     model_cent = Sersic2D(amplitude=1, r_eff = Re_arcsec, n = n_sersic, x_0=0, y_0=0, ellip=ellip, theta=np.radians(PA_deg))

    #     x_arcsec_1d = np.linspace(-angular_extent, angular_extent, n_grid)
    #     y_arcsec_1d = np.linspace(-angular_extent, angular_extent, n_grid)

    #     x_arcsec, y_arcsec = np.meshgrid(x_arcsec_1d, y_arcsec_1d)
        
    #     I_source = model_cent(x_arcsec, y_arcsec) # 2D array of the Sersic profile evaluated at each point in the 2D grid defined by x_arcsec and y_arcsec.
    
    # elif decentre_max == "core_radius":
    #     FOV_test_smf = plate_scale_demag * 2 * a
    #     angular_extent = FOV_test_smf / 2 # arcsec, which is half of the FOV of a 125e-6m light bucket at the Coudé focus, based on the plate scale calculated above.
    #     model_cent = Sersic2D(amplitude=1, r_eff = Re_arcsec, n = n_sersic, x_0=0, y_0=0, ellip=ellip, theta=np.radians(PA_deg))

    #     x_arcsec_1d = np.linspace(-angular_extent, angular_extent, n_grid)
    #     y_arcsec_1d = np.linspace(-angular_extent, angular_extent, n_grid)

    #     x_arcsec, y_arcsec = np.meshgrid(x_arcsec_1d, y_arcsec_1d)
        
    #     I_source = model_cent(x_arcsec, y_arcsec) # 2D array of the Sersic profile evaluated at each point in the 2D grid defined by x_arcsec and y_arcsec.

    def run_one_radius(i, a):
        row = extended_source_no_opt_efficiency_at_one_radius_2d(
            a=a, 
            x=x, y=y, x_1d=x_1d, y_1d=y_1d,
            NA=NA, 
            lam0=lam0,
            I_source=I_source,
            alpha=alpha,
            az_sym=az_sym, 
            mode_case=mode_case,
            decentre_max=decentre_max, 
            n_decentre=n_decentre, 
            E_S=E_S,
            n_F_grid=n_F_grid, 
            F_eff=F_eff
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

