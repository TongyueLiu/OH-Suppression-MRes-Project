from scipy import *
from Numeric import *
from pylab import *
from math import *

def lp(r, phi, l, a, u, w, cosine=True):
    """
    Returns the electric field of the fibre LP mode for a given r, phi
    coordinate when given the azimuthal order and transverse propagation
    constants.
    """
    if r > 0.0:
        if l==0:
            fphi = 1.0
        else:
            if cosine:
                fphi = cos(l*phi)
            else:
                fphi = sin(l*phi)
    else:
        if l==0:
            fphi = 1.0
        else:
            fphi = 0.0

    if r <= a:
        return fphi * special.jn(l,u*r/a) / special.jn(l,u)
    else:
        return fphi * special.kn(l,w*r/a) / special.kn(l,w)

def image(r, phi, F, alpha, wave, decentre=None):
    """
    Returns the electric field of the diffraction limited image at a given
    r, phi coordinate when given the F/#, central obstruction size and
    wavelength.
    """
    if decentre:
        dx,dy = decentre
        new_x = r*cos(phi) - dx
        new_y = r*sin(phi) - dy
        r = (new_x**2 + new_y**2)**0.5
        phi = atan2(new_y,new_x)
    rho = pi * r / (wave * F)
    if rho > 0.0:
        if alpha > 0.0:
            return special.j1(rho)/rho - alpha**2 * \
                   special.j1(alpha*rho)/(alpha*rho)
        else:
            return special.j1(rho)/rho
    else:
        return 0.5*(1.0 - alpha**2)

def Fvsd(A,NA,wave,alpha):
    lower = 1
    upper = 15
    data = []
    for a in A:
        data.append(best_eff(a,NA,wave,alpha,True,False,lower,upper))
        lower = data[-1][0] * 0.7 
        upper = data[-1][0] * 1.05
    return data

def best_eff(a, NA, wave, alpha, az_sym=False, pupil=False, lower=1, upper=30):
    """
    Optimises the total coupling efficiency over F ratio and
    returns the optimal F ratio and the resulting coupling efficiency

    If the image is azimuthally symmetric then the optional parameter
    az_sym can be set to True to speed up the calculationby only
    considering the l=0 modes.
    """
    #opt_F = optimize.brent( \
    #    lambda x: 1 - total_eff(a, NA, wave, x, alpha, az_sym)[0], \
    #    brack=(1.0,20.0))
    #opt_F = abs(optimize.anneal( \
    #    lambda x: 1 - total_eff(a, NA, wave, abs(x), alpha, az_sym)[0], \
    #    5, lower = 1, upper = 30)[0][0])
    opt_F = optimize.fminbound( \
        lambda x: 1 - total_eff(a, NA, wave, x, alpha, az_sym, None, pupil)[0], \
        lower, upper)
    max_eff = total_eff(a, NA, wave, opt_F, alpha, az_sym, None, pupil)
    print "a=%f NA=%f wave=%f alpha=%f opt_F=%f best_eff=%f" % \
          (a,NA,wave,alpha,opt_F,max_eff[0])
    return [opt_F] + max_eff

def total_eff(a, NA, wave, F, alpha, az_sym=False, decentre=None, pupil=False):
    """
    Calculates the total coupling efficiency included all LP modes
    above their cutoff.

    If the image is azimuthally symmetric then the optional parameter
    az_sym can be set to True to speed up the calculationby only
    considering the l=0 modes.
    """
    V = 2 * pi * a * NA / wave
    allmodes = modes(V)
    if az_sym:
        allmodes = [mode for mode in allmodes if mode[0]==0]
    if pupil:
        eff_func = eff2
    else:
        eff_func = eff
    total = 0.0
    for mode in allmodes:
        l,m = mode[0:2]
        u,w = uw(l,m,V)
        if l==0:
            mode.append(eff_func(l,m,a,NA,wave,F,alpha,decentre)[0])
            total += mode[-1]
        else:
            mode.append(eff_func(l,m,a,NA,wave,F,alpha,decentre,cosine=True)[0])
            mode.append(eff_func(l,m,a,NA,wave,F,alpha,decentre,cosine=False)[0])
            total += mode[-2] + mode[-1]
    print "a=%f NA=%f wave=%f F=%f alpha=%f total_eff=%f" % \
          (a,NA,wave,F,alpha,total)
    return [total] + allmodes

def eff(l, m, a, NA, wave, F, alpha, decentre=None, cosine=True):
    """
    Calculates the coupling efficiency between the diffraction limited image
    and a given LP mode of the fibre.
    """
    V = 2 * pi * a * NA / wave
    u,w = uw(l, m, V)
    overlap = integrate.dblquad( \
        lambda phi,r: image(r,phi,F,alpha,wave,decentre) * \
                        lp(r,phi,l,a,u,w,cosine) * r, \
        0, Inf, lambda x: 0, lambda x: 2*pi)
    if alpha > 0.0:
        im_int = integrate.quad( \
            lambda rho: special.j1(rho)*special.j1(alpha*rho)/(alpha*rho), \
            0, Inf)
        norm_im = [ \
            (wave**2 * F**2 / pi) * (1 + alpha**2 - 4 * alpha**2 * im_int[0]), \
            (wave**2 * F**2 * alpha**2 * im_int[1] / pi)]
    else:
        norm_im = [wave**2 * F**2 / pi, 0.0]
    norm_lp = integrate.dblquad( \
        lambda phi,r: lp(r,phi,l,a,u,w,cosine)**2 * r, \
        0, Inf, lambda x: 0, lambda x: 2*pi)
    efficiency = overlap[0]**2 / (norm_im[0] * norm_lp[0])
    error = efficiency * (2*(overlap[1]/overlap[0])**2 + \
            (norm_im[1]/norm_im[0])**2 + (norm_lp[1]/norm_lp[0])**2)**0.5
    print "l=%i m=%i a=%f NA=%f wave=%f F=%f alpha=%f cosine=%i eff=%f" % \
          (l,m,a,NA,wave,F,alpha,cosine,efficiency)
    return efficiency,error

def eff2(l, m, a, NA, wave, R, alpha, decentre=None, cosine=True):
    """
    Calculates the coupling efficiency between a perfect pupil image
    and a given LP mode of the fibre.
    """
    V = 2 * pi * a * NA / wave
    u,w = uw(l, m, V)
    overlap = integrate.dblquad( \
        lambda phi,r: lp(r,phi,l,a,u,w,cosine) * r, \
        alpha*R, R, lambda x: 0, lambda x: 2*pi)
    norm_im = [pi * R**2 * (1-alpha**2)]
    norm_lp = integrate.dblquad( \
        lambda phi,r: lp(r,phi,l,a,u,w,cosine)**2 * r, \
        0, Inf, lambda x: 0, lambda x: 2*pi)
    efficiency = overlap[0]**2 / (norm_im[0] * norm_lp[0])
    error = efficiency * (2*(overlap[1]/overlap[0])**2 + \
            (norm_lp[1]/norm_lp[0])**2)**0.5
    print "l=%i m=%i a=%f NA=%f wave=%f R=%f alpha=%f cosine=%i eff=%f" % \
          (l,m,a,NA,wave,R,alpha,cosine,efficiency)
    return efficiency,error

def pupil(r,phi,R,alpha,decentre):
    if decentre:
        dx,dy = decentre
        new_x = r*cos(phi) - dx
        new_y = r*sin(phi) - dy
        r = (new_x**2 + new_y**2)**0.5
    if r >= alpha*R and r < R:
        return 1.0
    else:
        return 0.0

def modes(V):
    # For each value of l, calculate the allowed values of m (if any).
    modelist = []
    l = 0
    while True:
        m = 1
        if l >= 1 and special.jn_zeros(l-1, 1)[0] > V:
            modelist.sort(lambda x,y: cmp(x[2],y[2]))
            return modelist
        cutoff = 0
        while V > cutoff:
            if l >= 1:
                cutoff = special.jn_zeros(l-1, m)[m-1]
            elif l == 0:
                if m >= 2:
                    cutoff = special.jn_zeros(1, m-1)[m-2]
                elif m == 1:
                    cutoff = 0
            if cutoff <= V: modelist.append([l,m,cutoff])
            m += 1
        l += 1
        
def uw(l, m, V):
    """
    Calculates the two transverse propagation constants u & w for a given
    normalised frequency and l,m.

    Arguments:

    V: Normalised frequency, float >= cutoff(l,m)
    l: Azimuthal order of the LP mode, integer >= 0
    m: Radial order of the LP mode, integer >= 1

    Returned values:

    u: Transverse propagation constant for the core, float
    w: Transverse propagation constant for the cladding, float
    """
    # First check that the normalised frequency is above the cutoff for
    # this mode.
    if V <= cutoff(l,m):
        raise ValueError( \
            "The given V (%f) is below the cutoff (%f) for this mode (%i,%i)" \
            % (V,cutoff(l,m),l,m) \
            )

    u = optimize.zeros.brentq(fu, cutoff(l,m)*(1+limits.double_epsilon), \
                              min(cutoff(l+1,m)*(1-limits.double_epsilon),V), \
                               args=(V,l))
    w = (V**2 - u**2)**0.5
    return u,w
    
def fu(ut,V,l):
    """
    Function whose zeros determine the transverse propagation contants
    u and w.

    Arguments:

    ut: Trial value of the tranverse propagation constant u, float >= 0
    V:  Normalised frequency, float >= 0
    l:  Azimuthal order of the LP mode, integer >= 0
    """
    wt = (V**2 - ut**2)**0.5
    if l > 0:
        return ut*special.jn(l-1,ut)/special.jn(l,ut) + \
               wt*special.kn(l-1,wt)/special.kn(l,wt)
    elif l == 0:
        return ut*special.jn(1,ut)/special.jn(0,ut) - \
               wt*special.kn(1,wt)/special.kn(0,wt)
    else:
        raise ValueError("l must be an integer >= 0")

def dfu(ut, V, l):
    """
    Calculates the derivative of the function calculated by fu.
    """
    wt = (V**2 - u**2)**0.5
    if l > 0:
        return ut*( special.jn(l+1,ut)*special.jn(l-1,ut)/jn(l,ut)**2 - \
                    special.kn(l+1,wt)*special.kn(l-1,wt)/kn(l,wt)**2 )
    elif l == 0:
        return ut*((special.j1(ut)/special.j0(ut))**2 + \
                   (special.k1(wt)/special.k0(wt))**2);
    else:
        raise ValueError("l must be an integer >= 0")

def cutoff(l,m):
    """
    Utility function which returns the normalised frequency (V) at cutoff for
    a given mode l,m.

    Arguments:

    l: Azimuthal order of the LP mode, integer >=0
    m: Radial order of the LP mode, integer >=1

    Return value:

    Vc: Normalised frequency at cutoff for the given mode
    """
    if l>=1:
        # Vc is the mth zero of the J_(l-1) Bessel function.
        Vc = special.jn_zeros(l-1, m)[-1]
    elif l == 0:
        if m >= 2:
            # Special case of l==0, use m-1th zero of J_1 as
            # special.jn_zeros(1,x) excludes the zero at 0 for some reason. 
            Vc = special.jn_zeros(1, m-1)[-1]
        elif m == 1:
            # This is the fundamental mode, cutoff frequency is zero.
            Vc = 0.0
    else:
        raise ValueError("l must be an integer >= 0")

    return Vc
