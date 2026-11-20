"""
Benjamin Courtney-Barrer
Step index multimode fibre coupling to coherent/incoherent extended sources.
scalar LP modes, ideal lens, one 1rad rms frozen filtered noise phase screen.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import jv, kv
from scipy.ndimage import gaussian_filter

# ---------------- Physical parameters ----------------
lam = 633e-9
n_core, n_clad = 1.460, 1.455
core_d = 25e-6
D = 8.2                       # telescope pupil diameter [m]
F_number = 15.0                  # one fixed injection F/#
phase_rms = 1.0                  # pupil phase RMS [rad]
phase_corr =  D / 7.1            # smoothing scale of random phase [m]
source_fwhm_mas = 30.0 #milliarcsec
source_sigma_rad = source_fwhm_mas * np.pi / (180 * 3600 * 1000) / 2.355
source_sigma = source_sigma_rad / (lam / D) # Gaussian source sigma in lambda/D units
source_halfwidth = 2.0           # sampled source half-width in lambda/D units

source_N = 9                     # source grid per axis
seed = 3
N, dx = 768, 0.8e-6             # fibre face sampling (pixel count, metres)

# ---------------- Guided LP modes at fibre face ----------------
a = core_d/2
V = 2*np.pi*a/lam*np.sqrt(n_core**2-n_clad**2)

def eigen_eq(u, ell):
    w = np.sqrt(max(V*V-u*u, 0.0))
    return u*jv(ell-1, u)*kv(ell, w) + w*jv(ell, u)*kv(ell-1, w)

roots = []
trial = np.linspace(1e-5, V-1e-5, 5000)
for ell in range(int(np.ceil(V))+1):
    values = np.array([eigen_eq(u, ell) for u in trial])
    for i in np.where(values[:-1]*values[1:] < 0)[0]:
        u = brentq(eigen_eq, trial[i], trial[i+1], args=(ell,))
        if not any(l == ell and abs(u-v) < 1e-5 for l,v in roots):
            roots.append((ell, u))

# Modes are confined near fibre core, so store only a small central region.
x = (np.arange(N)-N//2)*dx
h = int(np.ceil(100e-6/dx))
s = slice(N//2-h, N//2+h+1)
X, Y = np.meshgrid(x[s], x[s])
r, phi = np.hypot(X, Y), np.arctan2(Y, X)
modes, labels = [], []
for ell, u in roots:
    w = np.sqrt(V*V-u*u)
    radial = np.where(r <= a, jv(ell, u*r/a),
                      jv(ell, u)*kv(ell, w*np.maximum(r, a)/a)/kv(ell, w))
    for tag, angular in ([('', np.ones_like(phi))] if ell == 0 else
                         [('c', np.cos(ell*phi)), ('s', np.sin(ell*phi))]):
        mode = radial*angular
        mode /= np.linalg.norm(mode)  # entire mode, not pupil-truncated
        modes.append(mode.ravel())
        labels.append(f'LP({ell},{tag})')
modes = np.asarray(modes)
print(f'V={V:.2f}; {len(modes)} scalar spatial modes')
print(f'Max orthogonality error: {np.max(np.abs(modes@modes.T-np.eye(len(modes)))):.2e}')

# ---------------- Same physical coordinates for pupil and fibre modes ----------------
f = F_number*D
p = np.fft.fftshift(np.fft.fftfreq(N, dx))*lam*f
PX, PY = np.meshgrid(p, p)
pupil = PX**2+PY**2 <= (D/2)**2
rng = np.random.default_rng(seed)
phase = gaussian_filter(rng.standard_normal((N, N)), phase_corr/(p[1]-p[0]))
phase -= phase[pupil].mean()
phase *= phase_rms/phase[pupil].std()
base_field = pupil*np.exp(1j*phase)
base_field /= np.linalg.norm(base_field)

# Fourier transform fibre's LP01 mode to show its back propagated pupil intensity.
face_01 = np.zeros((N, N), complex)
face_01[s, s] = modes[0].reshape(X.shape)
pupil_01 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(face_01), norm='ortho'))

# ---------------- Incoherent extended source ----------------
# Each sky point (theta_x,theta_y) adds a pupil tilt exp(2pi i theta.x / lambda).
# Angles expressed in lambda/D units => tilt exp(2pi i (q_x X+q_y Y)/D).
q = np.linspace(-source_halfwidth, source_halfwidth, source_N)
QX, QY = np.meshgrid(q, q)
brightness = np.exp(-(QX**2+QY**2)/(2*source_sigma**2))
brightness /= brightness.sum()   # fractional source flux at each sky position

source_coupling = np.zeros_like(brightness)
modal_powers = np.zeros(len(modes))    # incoherent: weighted modal powers
coherent_amplitudes = np.zeros(len(modes), complex)
coherent_field = np.zeros_like(base_field)  # coherent: sum tilted fields
point_powers = None
for iy in range(source_N):
    for ix in range(source_N):
        tilt = np.exp(2j*np.pi*(QX[iy,ix]*PX+QY[iy,ix]*PY)/D)
        field = base_field*tilt
        # Unitary Parseval: overlap in pupil equals overlap at fibre face.
        face = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field), norm='ortho'))
        amplitudes = modes.conj() @ face[s, s].ravel()
        powers = np.abs(amplitudes)**2
        weight = np.sqrt(brightness[iy, ix])  # zero relative source phase
        coherent_amplitudes += weight*amplitudes
        coherent_field += weight*field
        source_coupling[iy,ix] = powers.sum()
        modal_powers += brightness[iy,ix]*powers  # add POWERS, not amplitudes
        if iy == source_N//2 and ix == source_N//2:
            point_powers = powers.copy()

eta_extended = modal_powers.sum()
# Coherent modes interfere; normalise by ACTUAL total pupil-plane power.
coherent_pupil_power = np.vdot(coherent_field, coherent_field).real
coherent_powers = np.abs(coherent_amplitudes)**2 / coherent_pupil_power
eta_coherent = coherent_powers.sum()
print(f'Point-source on-axis coupling: {point_powers.sum():.4f}')
print(f'Extended-source total coupling (incoherent): {eta_extended:.4f}')
print(f'Extended-source LP01 coupling (incoherent): {modal_powers[0]:.4f}')
print(f'Extended-source total coupling (coherent):   {eta_coherent:.4f}')
print(f'Extended-source LP01 coupling (coherent):   {coherent_powers[0]:.4f}')
print(f'Coherent / incoherent coupling ratio:       {eta_coherent/eta_extended:.3f}')

# ---------------- Plots ----------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
zoom = (-1.3*D/2*1e3, 1.3*D/2*1e3)*2
im = ax[0,0].imshow(np.ma.masked_where(~pupil, phase), origin='lower',
                    extent=[p[0]*1e3,p[-1]*1e3,p[0]*1e3,p[-1]*1e3], cmap='RdBu_r')
ax[0,0].set(xlim=zoom[:2], ylim=zoom[2:], title='Turbulent pupil phase [rad]',
            xlabel='x [mm]', ylabel='y [mm]')
fig.colorbar(im, ax=ax[0,0])
im = ax[0,1].imshow(abs(pupil_01)**2, origin='lower',
                    extent=[p[0]*1e3,p[-1]*1e3,p[0]*1e3,p[-1]*1e3], cmap='inferno')
ax[0,1].add_patch(plt.Circle((0,0), D/2*1e3, ec='cyan', fill=False, lw=1.5))
ax[0,1].set(xlim=zoom[:2], ylim=zoom[2:], title=f'back propagated LP01 intensity, F/{F_number:g}',
            xlabel='x [mm]', ylabel='y [mm]')
fig.colorbar(im, ax=ax[0,1])
im = ax[1,0].imshow(brightness, origin='lower', cmap='magma',
                    extent=[q[0],q[-1],q[0],q[-1]])
ax[1,0].set(title='Source brightness profile', xlabel=r'$\theta_x\ [\lambda/D]$',
            ylabel=r'$\theta_y\ [\lambda/D]$')
fig.colorbar(im, ax=ax[1,0])
ax[1,1].plot(point_powers, '.', ms=5, label=f'On-axis point: {point_powers.sum():.3f}')
ax[1,1].plot(modal_powers, '.', ms=5, label=f'Extended source: {eta_extended:.3f}')
ax[1,1].set(title='Modal coupling (fraction of incident flux)', xlabel='Mode index',
            ylabel='Coupled power fraction', ylim=(0,None))
ax[1,1].legend()

# ---------------- Additional coherent/incoherent diagnostics ----------------
fig2, bx = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
# Same angular brightness envelope; compare combined coherent pupil amplitude.
amp = np.abs(coherent_field) / np.max(np.abs(coherent_field))
im = bx[0].imshow(np.ma.masked_where(~pupil, amp), origin='lower',
                  extent=[p[0]*1e3,p[-1]*1e3,p[0]*1e3,p[-1]*1e3], cmap='viridis',
                  vmin=0, vmax=1)
bx[0].set(xlim=zoom[:2], ylim=zoom[2:], title='Coherent source: pupil amplitude',
          xlabel='x [mm]', ylabel='y [mm]')
fig2.colorbar(im, ax=bx[0], label='Relative amplitude')
bx[1].plot(modal_powers, '.', ms=5, label=f'Incoherent: {eta_extended:.3f}')
bx[1].plot(coherent_powers, '.', ms=5, label=f'Coherent: {eta_coherent:.3f}')
bx[1].set(title='Extended-source modal coupling', xlabel='Mode index',
          ylabel='Coupled power / incident pupil power', ylim=(0, None))
bx[1].legend()
plt.show()
