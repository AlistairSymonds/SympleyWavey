#from here: https://www.reddit.com/r/Optics/comments/xccrsk/options_for_free_optical_simulation/io83hq6/
# Big thanks to Brandon Dube for both prysm and giving me this example :)
from prysm import geometry, coordinates, polynomials
from prysm.propagation import Wavefront as WF
from prysm.segmented import _local_window
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from astropy.io import fits
import wavey





def shack_hartmann_phase_screen(x, y, pitch_x, pitch_y, n_x, n_y, efl, wavelength, aperture=geometry.rectangle):

    dx = x[0,1] - x[0,0]
    min_pitch = min(pitch_x, pitch_y)
    samples_per_lenslet = int(min_pitch / dx + 1) # ensure safe rounding

    ulens_centres_x = np.linspace(-((pitch_x*(n_x-1))/2), ((pitch_x*(n_x-1))/2), n_x)
    ulens_centres_y = np.linspace(-((pitch_y*(n_y-1))/2), ((pitch_y*(n_y-1))/2), n_y)
    print(ulens_centres_x)
    print(ulens_centres_y)
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
    #lenslet_rsq = (pitch/2)**2
    total_phase = np.zeros_like(x)
    for j, yy in enumerate(ulens_centres_y):
        for i, xx in enumerate(ulens_centres_x):
            iy, ix = _local_window(cy, cx, (xx,yy), dx, samples_per_lenslet, x, y)
            lx = x[iy, ix] - xx
            ly = y[iy, ix] - yy
            rsq = lx * lx + ly * ly
            phase = rsq / (2*efl)
            phase *= aperture(width=pitch_x/2, height=pitch_y/2, x=lx, y=ly)
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
x, y = coordinates.make_xy_grid(8192, diameter=12)
r, t = coordinates.cart_to_polar(x, y)
dx = x[0,1]-x[0,0]

efl = 4.7
rmax = 5
wvl = 0.550
mask = shack_hartmann_phase_screen(x, y, pitch_x=1, pitch_y=1.4, n_x=10, n_y=7, efl=efl, wavelength=wvl)
amp = geometry.circle(rmax, r) #^ geometry.circle(0.47*rmax, r)

znorm = True
phs = polynomials.zernike_nm(4,0, r/rmax, t, norm=znorm) # Zernike flavor spherical aberration
phs += (polynomials.zernike_nm(2,2, r/rmax, t, norm=znorm)) # Zernike flavor astig
phs += (polynomials.zernike_nm(2,-2, r/rmax, t, norm=znorm))/2 # Zernike flavor astig
phs += ((polynomials.zernike_nm(1,-1, r/rmax, t, norm=znorm))) # Zernike flavor tilt in y
##
phs += ((polynomials.zernike_nm(3,1, r/rmax, t, norm=znorm))/2) # Coma
phs += ((polynomials.zernike_nm(3,-3, r/rmax, t, norm=znorm))/6) # Trefoil, we won't try to fit this and instead have it as 'noise'
nwaves_aber = 5
phs = phs * (wvl*1e3 * nwaves_aber)
wf = WF.from_amp_and_phase(amp, phs, .550, dx)
wf = wf * mask
wf2 = wf.free_space(dz=efl, Q=1)
i = wf2.intensity
i.data /= i.data.max()

c = wavey.imx174()
capture174 = c.expose2(i, x, y, 500)

hdu174 = fits.PrimaryHDU(capture174)
hdu174.writeto('imx174_mla1.fits', overwrite=True)


c = wavey.imx533()
capture533 = c.expose2(i, x, y, 50)

hdu533 = fits.PrimaryHDU(capture533)
hdu533.writeto('imx533_mla1.fits', overwrite=True)
