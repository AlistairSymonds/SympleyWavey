#from here: https://www.reddit.com/r/Optics/comments/xccrsk/options_for_free_optical_simulation/io83hq6/
# Big thanks to Brandon Dube for both prysm and giving me this example :)
from prysm import geometry, coordinates, polynomials
from prysm.propagation import Wavefront as WF
from prysm.segmented import _local_window
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from astropy.io import fits

def shack_hartmann_phase_screen(x, y, pitch, n, efl, wavelength, aperture=geometry.rectangle):
    if not hasattr(n, '__iter__'):
        n = (n,n)

    dx = x[0,1] - x[0,0]
    samples_per_lenslet = int(pitch / dx + 1) # ensure safe rounding

    xc, yc = coordinates.make_xy_grid(n, dx=pitch, grid=False)
# if you use prysm's plotting utilities, they shear things a bit to be "FFT aligned"
# and doing this will make the center of the beam not contain a lenslet
# uncomment this if you want symmetric lenslets across the origin, instead
# of having a lenslet at the origin

#     if not is_odd(n[0]):
#         # even number of lenslets, FFT-aligned make_xy_grid needs positive shift
#         xc += (pitch/2)
#     if not is_odd(n[1]):
#         yc += (pitch/2)

    cx = ceil(x.shape[1]/2)
    cy = ceil(y.shape[0]/2)
    lenslet_rsq = (pitch/2)**2
    total_phase = np.zeros_like(x)
    for j, yy in enumerate(yc):
        for i, xx in enumerate(xc):
            iy, ix = _local_window(cy, cx, (xx,yy), dx, samples_per_lenslet, x, y)
            lx = x[iy, ix] - xx
            ly = y[iy, ix] - yy
            rsq = lx * lx + ly * ly
            phase = rsq / (2*efl)
            phase *= aperture(pitch/2, lx, ly)
# swap the aperture here if you fancy
#             phase *= geometry.circle(lenslet_rsq, rsq)
            total_phase[iy, ix] += phase

    prefix = -1j * 2 * np.pi/(wavelength/1e3)
    return np.exp(prefix*total_phase)


# above this is your "library" code

# diameter here is the beam in mm, need to assume some sort of magnification between your telescope
# and the SH-WFS
#
# also, if you fiddle with the SH-WFS lenslet parameters, they are quite sensitive and easy to drive the model
# into an undersampled regime (need more samples)
# or have the lenslet PSFs overlap because the F/# is so big.  Delicate balancing act.
x, y = coordinates.make_xy_grid(4096, diameter=10)
r, t = coordinates.cart_to_polar(x, y)
dx = x[0,1]-x[0,0]

efl = 5
rmax = 5
wvl = 0.550
mask = shack_hartmann_phase_screen(x, y, pitch=.3, n=40, efl=efl, wavelength=wvl)
amp = geometry.circle(rmax, r)
phs = polynomials.zernike_nm(4,0, r/rmax, t) # Zernike flavor spherical aberration
phs += (polynomials.zernike_nm(2,2, r/rmax, t)) # Zernike flavor astig
#show phase of wavefront
if True:
    phi2 = phs.copy()
    phi2[amp!=1]=np.nan
    plt.imshow(phi2)
    plt.show()
nwaves_aber = 20
phs = phs * (wvl*1e3 * nwaves_aber)
wf = WF.from_amp_and_phase(amp, phs, .550, dx)

#fig, ax = plt.subplots(figsize=(12,12))
#wf.plot2d(power=1/2,  interpolation='lanczos',  fig=fig, ax=ax)
#plt.show()

wf = wf * mask
wf2 = wf.free_space(dz=efl, Q=1)
i = wf2.intensity
i.data /= i.data.max()

hdu = fits.PrimaryHDU(i.data)
hdu.writeto('prysm_sh_capture.fits', overwrite=True)
fig, ax = plt.subplots(figsize=(12,12))
i.plot2d(power=1/2,  interpolation='lanczos',  fig=fig, ax=ax)
plt.show()