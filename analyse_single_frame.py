import argparse as ap
from wavey import microlens_array, analyser, wavefronts
from prysm import coordinates
from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
import json
if __name__ == "__main__":

    parser = ap.ArgumentParser("Shack-Hartmann analyser")
    parser.add_argument("fits_path")
    parser.add_argument("--pixel_size", help="Pixel size in um", type=float)
    parser.add_argument("--wavefront_radius", help="mm", type=float)
    parser.add_argument("--microlens", help="Which type of microlens array was used", choices=["mla300", "mla1"])
    
    args = parser.parse_args()

    fits_path = Path(args.fits_path)
    fits_file = fits.open(fits_path)
    ulens_arr = None
    if args.microlens == "mla300":
        ulens_arr = microlens_array.ThorlabsMLA300()
    elif args.microlens == "mla1": 
        ulens_arr = microlens_array.ThorlabsMLA1()

    microlens_positions = ulens_arr.get_lens_centres()
    micro_lens_corners = ulens_arr.get_lens_cell_corners()

    print("analysing...")
    a = analyser.ShackHartmannAnalyser(microlens_positions, microlens_cell_corners=micro_lens_corners, wavefront_radius=5, wvl=0.550, px_x_size=args.pixel_size/1000, px_y_size=args.pixel_size/1000, 
                    sensor_y_px_count=fits_file[0].data.shape[0], sensor_x_px_count=fits_file[0].data.shape[1], debug=True)

    r = a.analyse(fits_file[0].data)
    
    #also load in refernce wave data if it exists
    ref_json_path = fits_path.parent / "generated_wave_zernike.json"
    if ref_json_path.exists:
        with open(ref_json_path, "r") as f:
            ref_zernikes = json.load(f)

        x, y = coordinates.make_xy_grid(2048, diameter=10)
        zerns = {}
        for j in ref_zernikes:
            zerns[int(j)] = float(ref_zernikes[j])
            

        ref_wave = wavefronts.ZernikeWavefrontAnalysis("reference", wavelength=0.550, zernikes=zerns, wf_radius=5, x=x, y=y)
        
        ref_wave_comparator = wavefronts.WavefrontComparator(ref_wave)
        ref_wave_comparator.compare_waves(r)
    else:
        r.plot_wavefront()

    plt.show()