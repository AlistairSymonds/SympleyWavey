import argparse as ap
from wavey import microlens_array, sensor, wavefronts
from astropy.io import fits
from pathlib import Path
from prysm import geometry, coordinates
from prysm.polynomials import zernike_nm, zernike_nm_der_sequence, lstsq, nm_to_name, fringe_to_nm, zernike_nm_der
from prysm.propagation import Wavefront as WF
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
if __name__ == "__main__":

    parser = ap.ArgumentParser("Shack-Hartmann analyser")
    parser.add_argument("--seed", help="Seed value for python's rand to be used during wavefront generation", type=int, default=1)
    parser.add_argument("--sensor", help="Sensor type", choices=["imx174", "im533"])
    parser.add_argument("--wavefront_radius", help="mm", type=float)
    parser.add_argument("--wavelength", help="in micron", type=float)
    parser.add_argument("--microlens", help="Which type of microlens array was used", choices=["mla300", "mla1"])
    parser.add_argument("--aperture", choices=["circle", "RC8", "RC12", "Hubble", "JWST"])
    parser.add_argument("--output_dir", default="")
    
    args = parser.parse_args()

    random.seed(args.seed)

    wf_r = args.wavefront_radius
    samples = 8192
    x, y = coordinates.make_xy_grid(samples, diameter=12)
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

    

    phs = np.zeros((samples,samples))
    max_waves_aber = 1
    fringe_indices = range(1,10)
    fringe_indices = [7]
    zernike_values = {}
    for fi in fringe_indices:
        if fi != 4:
            n, m = fringe_to_nm(fi)
            weight = random.random() * max_waves_aber
            phs += (weight * zernike_nm(n, m, r/wf_r, t))
            zernike_values[(n, m)] = weight
            print(str((n, m)) + " " + nm_to_name(n, m) + ": " + str(weight))

    
    wf_data = wavefronts.WavefrontAnalysis("generated", args.wavelength, zernike_values)
    #(dr, dt) = wf_data.gen_der_wf(r/wf_r, t)
    #wavefronts.plot_zernike_der(dr, dt, r/wf_r, t)

    #z = wf_data.gen_wf(r/wf_r, t)
    #wavefronts.plot_zernike(z, r/wf_r, t)

    run_path = Path("runs")
    run_path = run_path / str(args.seed)
    run_path.mkdir(parents=True, exist_ok=True)

    #TODO
    #for ulens array in ulens types
    #    for sensor in sensors
    #        all code below but loopified
    microlens_positions = ulens_arr.get_lens_centres()
    micro_lens_corners = ulens_arr.get_lens_cell_corners()

    ulens_phase = ulens_arr.shack_hartmann_phase_screen(x=x, y=y, wavelength=args.wavelength)

    path_error_nm = phs * (args.wavelength*1e3)
    wf = WF.from_amp_and_phase(amp, path_error_nm, args.wavelength, dx)
    wf = wf * ulens_phase
    print("propagating wf")
    wf2 = wf.free_space(dz=ulens_arr.get_fl(), Q=1)
    i = wf2.intensity
    i.data /= i.data.max()

    c = sensor.imx174()
    capture174 = c.expose2(i, x, y, 500)

    hdu174 = fits.PrimaryHDU(capture174)
    hdu174.writeto(run_path / 'imx174_mla1.fits', overwrite=True)
