import numpy as np
from matplotlib import pyplot as plt 

from ..fd.grids import FDLocalPolarGrid

def amplitude_error_polar(
    computed_vals: np.ndarray,
    act_vals: np.ndarray,
    grid: FDLocalPolarGrid
) -> dict[str, float]:
    """Compute amplitude error between computed values and actual values.
    
    Returns the following error estimates:
    
    1. L2 Error on the entire domain ('L2_domain')
    2. Max Error on the entire domain ('max_domain')
    3. L2 Error on the artificial boundary ('L2_art_bndry')
    4. Relative L2 Error on the artificial boundary ('L2rel_art_bndry')
    5. Max Error on the artificial boundary ('max_art_bndry')
    6. Relative Max Error on the artificial boundary ('maxrel_art_bndry')

    NOTE: All L2 errors are computed using a left-point
    Reimann-sum approximation for the double integral over the domain
    of interest.

    Args:
        computed_vals (np.ndarray): A 2d array representing the
            computed values of the quantity of interest at each
            gridpoint 
        act_vals (np.ndarray): A 2d array representing the
            actual values of the quantity of interest at each
            gridpoint
        grid (FDLocalPolarGrid): A polar grid that these quantities
            of interest are defined on
    
    Returns:
        dict[str, float]: A dictionary 
        float: The error between the computed and actual values
    """
    # Get differences in amplitudes/complex moduli of the
    # computed values versus the actual values
    abs_diffs_domain = np.abs(np.abs(computed_vals)-np.abs(act_vals))
    abs_act_domain = np.abs(act_vals)
    abs_diffs_bndry = abs_diffs_domain[:,-1]
    abs_act_bndry = abs_act_domain[:,-1]

    # Parse other needed constants
    r_domain = grid.r_local
    r_bndry = r_domain[:,-1] 
    dr = grid.dr 
    dtheta = grid.dtheta

    # Initialize output error dictionary
    out_errs = dict()

    ## L2 ERROR
    # Get L2 (and relative L2) error over entire domain 
    L2_err_domain = np.sqrt(
        np.sum(abs_diffs_domain**2 * r_domain * dr * dtheta)
    )
    L2_norm_act_domain = np.sqrt(
        np.sum(abs_act_domain**2 * r_domain * dr * dtheta)
    )       # TODO: Are these constants correct?
    out_errs['L2_domain'] = L2_err_domain
    out_errs['L2rel_domain'] = L2_err_domain/L2_norm_act_domain

    # Get L2 (and relative L2) error over artificial boundary
    L2_err_bndry = np.sqrt(
        np.sum(abs_diffs_bndry**2 * r_bndry * dr * dtheta)
    )
    L2_norm_act_bndry = np.sqrt(
        np.sum(abs_act_bndry**2 * r_bndry * dr * dtheta)
    )
    out_errs['L2_art_bndry'] = L2_err_bndry
    out_errs['L2rel_art_bndry'] = L2_err_bndry/L2_norm_act_bndry

    ## MAX/L-INFINITY ERROR
    # Get maximum (and relative maximum/L-infinity) error over entire domain
    max_err_domain = np.max(abs_diffs_domain)
    max_norm_act_domain = np.max(abs_act_domain)
    out_errs['max_domain'] = max_err_domain
    out_errs['maxrel_domain'] = max_err_domain / max_norm_act_domain

    # Get maximum (and relative maximum/L-infinity) error over artificial boundary
    max_err_bndry = np.max(abs_diffs_bndry)
    max_norm_act_bndry = np.max(abs_act_bndry)
    out_errs['max_art_bndry'] = max_err_bndry
    out_errs['maxrel_art_bndry'] = max_err_bndry / max_norm_act_bndry

    # Return this output dictionary of errors
    return out_errs


### Define function for examining order of convergence
def display_error_convergence(
    step_vals: np.ndarray,
    errs: np.ndarray,
    err_name: str
):
    """Prints out an error table of the order of convergence in the 
    given norm. Computes a least-squares fit to determine the 
    average order of convergence. 
    Also plots a log-log graph of the error versus the step 
    size.

    Parameters:
        step_vals ((n,) ndarray): Array of step sizes
        errs ((n,) ndarray): The corresponding errors at 
            each iteration 
        err_name (str): The name of the error (for printing/formatting)
    """
    ### Compute the observed order of the L2-norm and Max-Norms at each iteration
    print(f"Step Size    {err_name}-Error   Observed Order")
    for i in range(len(step_vals)):
        err = errs[i]
        h = step_vals[i]

        if i == 0:          # Do not calculate error order at first iteration
            print(f"{h:0.3e}      {err:.3e}")
        else:
            past_err = errs[i-1]
            past_h = step_vals[i-1]
            ord = np.log(np.abs(err/past_err)) / np.log(np.abs(h/past_h))
            print(f"{h:0.3e}      {err:.3e}         {ord:.3e}")

    ### Compute the least-squares fit for the order of the error at each iteration 
    lsq_ord_err = np.polyfit(np.log(step_vals), np.log(errs), 1)[0]
    print('')
    print('Least Squares fit gives that:')
    print('  Error Order:  ', lsq_ord_err)

    plt.loglog(step_vals, errs, '.-', label=f'{err_name}-Norm Error')
    plt.title("Error of Finite Difference Approximation")
    plt.xlabel('Step Size')
    plt.ylabel('Error')
    plt.legend()
    plt.show()