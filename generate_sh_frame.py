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
import json
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

    ulens_arrays = [microlens_array.ThorlabsMLA300(), microlens_array.ThorlabsMLA1()]
   
    image_sensors = [sensor.imx533(), sensor.imx174()]
    
    amp = None
    if args.aperture == "circle":
        amp = geometry.circle(wf_r, r)

    

    phs = np.zeros((samples,samples))
    max_waves_aber = 1
    num_aberations = random.randint(1, 4)
    fringe_indices = []
    for i in range(0, num_aberations):
        fringe_indices.append(random.randint(3, 10))
    
    fringe_indices.sort()

    zernike_values = {}
    for fi in fringe_indices:
        n, m = fringe_to_nm(fi)
        weight = random.random() * max_waves_aber
        phs += (weight * zernike_nm(n, m, r/wf_r, t))
        zernike_values[fi] = weight
        print(str((n, m)) + " " + nm_to_name(n, m) + ": " + str(weight))

    path_error_nm = phs * (args.wavelength*1e3)

    wf_data = wavefronts.WavefrontAnalysis("generated", args.wavelength, path_error_nm, x, y)
    fig, ax1 = plt.subplots()

    wf_data.plot_wavefront(ax=ax1)
    plt.show()

    run_path = Path("runs")
    run_path = run_path / str(args.seed)
    run_path.mkdir(parents=True, exist_ok=True)

    
    with open(run_path / "generated_wave_zernike.json", "w") as f:
        json.dump(zernike_values, f)


    for ulens_arr in ulens_arrays:
        microlens_positions = ulens_arr.get_lens_centres()
        micro_lens_corners = ulens_arr.get_lens_cell_corners()

        ulens_phase = ulens_arr.shack_hartmann_phase_screen(x=x, y=y, wavelength=args.wavelength)

        
        wf = WF.from_amp_and_phase(amp, path_error_nm, args.wavelength, dx)
        wf = wf * ulens_phase
        print("propagating wf")
        wf2 = wf.free_space(dz=ulens_arr.get_fl(), Q=1)
        i = wf2.intensity
        i.data /= i.data.max()

        for s in image_sensors:
            capture = s.expose2(i, x, y, 500)
            hdu = fits.PrimaryHDU(capture)
            hdu.header['XPIXSZ'] = s.pixel_width * 1000
            hdu.header['YPIXSZ'] = s.pixel_height * 1000
            hdu.writeto(run_path / f"{s.__class__.__name__}_{ulens_arr.__class__.__name__}.fits", overwrite=True)
