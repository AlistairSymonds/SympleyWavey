import argparse as ap
from wavey import microlens_array, sensor
from astropy.io import fits
from pathlib import Path
from prysm import geometry, coordinates
from prysm.polynomials import zernike_nm, zernike_nm_der_sequence, lstsq, nm_to_name, fringe_to_nm, zernike_nm_der
from prysm.propagation import Wavefront as WF
import numpy as np
import random
import matplotlib.pyplot as plt
if __name__ == "__main__":

    parser = ap.ArgumentParser("Shack-Hartmann analyser")
    parser.add_argument("--sensor", help="Sensor type", choices=["imx174", "im533"])
    parser.add_argument("--wavefront_radius", help="mm", type=float)
    parser.add_argument("--wavelength", help="in micron", type=float)
    parser.add_argument("--microlens", help="Which type of microlens array was used", choices=["mla300", "mla1"])
    parser.add_argument("--aperture", choices=["circle", "RC8", "RC12", "Hubble", "JWST"])
    
    args = parser.parse_args()

    wf_r = args.wavefront_radius

    x, y = coordinates.make_xy_grid(8192, diameter=12)
    r, t = coordinates.cart_to_polar(x, y)
    dx = x[0,1]-x[0,0]

    ulens_arr = None
    if args.microlens == "mla300":
        ulens_arr = microlens_array.ThorlabsMLA300()
    elif args.microlens == "mla1": 
        ulens_arr = microlens_array.ThorlabsMLA1()

   
    s = None
    if args.sensor == "imx533":
        s = sensor.imx533()
    elif args.sensor == "imx174": 
        s = sensor.imx174()
    
    amp = None
    if args.aperture == "circle":
        amp = geometry.circle(wf_r, r)

    

    microlens_positions = ulens_arr.get_lens_centres()
    micro_lens_corners = ulens_arr.get_lens_cell_corners()

    ulens_phase = ulens_arr.shack_hartmann_phase_screen(x=x, y=y, wavelength=args.wavelength)
    plt.imshow(np.angle(ulens_phase))
    plt.show()

    phs = np.zeros((8192,8192))
    fringe_indices = range(2,10)
    for fi in fringe_indices:
        n, m = fringe_to_nm(fi)
        weight = random.random()
        phs += (weight * zernike_nm(n, m, r/wf_r, t))
        print(str((n, m)) + " " + nm_to_name(n, m) + ": " + str(weight))


    nwaves_aber = 5
    phs = phs * (args.wavelength*1e3 * nwaves_aber)
    if True:
        phi2 = phs.copy()
        phi2[amp!=1]=np.nan
        plt.imshow(phi2)
        plt.show()

    wf = WF.from_amp_and_phase(amp, phs, args.wavelength, dx)
    wf = wf * ulens_phase
    wf2 = wf.free_space(dz=ulens_arr.get_fl(), Q=1)
    i = wf2.intensity
    i.data /= i.data.max()

    c = sensor.imx174()
    capture174 = c.expose2(i, x, y, 500)

    hdu174 = fits.PrimaryHDU(capture174)
    hdu174.writeto('imx174_mla1.fits', overwrite=True)

