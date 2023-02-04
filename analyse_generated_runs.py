import argparse as ap
from wavey import microlens_array, analyser, wavefronts
from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
from prysm import coordinates
import json

def analyse_single_fits(fits_path: Path) -> wavefronts.WavefrontAnalysis:

    fits_file = fits.open(fits_path)
    ulens_arr = None
    
    print(f"Analysing: {fits_path}")

    if "ThorlabsMLA300" in fits_path.stem:
        ulens_arr = microlens_array.ThorlabsMLA300()
    elif "ThorlabsMLA1" in fits_path.stem: 
        ulens_arr = microlens_array.ThorlabsMLA1()
    else: 
        raise LookupError("Couldn't determine ulens array from fits filename")

    microlens_positions = ulens_arr.get_lens_centres()
    micro_lens_corners = ulens_arr.get_lens_cell_corners()
    
    xpixsz = fits_file[0].header['XPIXSZ']
    ypixsz = fits_file[0].header['YPIXSZ']

    print("analysing...")
    a = analyser.ShackHartmannAnalyser(microlens_positions, microlens_cell_corners=micro_lens_corners, wavefront_radius=5, wvl=0.550, px_x_size=xpixsz/1000, px_y_size=ypixsz/1000, 
                    sensor_y_px_count=fits_file[0].data.shape[0], sensor_x_px_count=fits_file[0].data.shape[1])

    r = a.analyse(fits_file[0].data)
    return r

def analyse_single_run(run_path: Path):
    
    ref_json_path = run_path / "generated_wave_zernike.json"
    with open(ref_json_path, "r") as f:
        ref_zernikes = json.load(f)

    x, y = coordinates.make_xy_grid(2048, diameter=10)
    zerns = {}
    for j in ref_zernikes:
        zerns[int(j)] = float(ref_zernikes[j])
        

    ref_wave = wavefronts.ZernikeWavefrontAnalysis("reference", wavelength=0.550, zernikes=zerns, wf_radius=5, x=x, y=y)
    
    ref_wave_comparator = wavefronts.WavefrontComparator(ref_wave)
    for f in run_path.glob("*.fits"):
        r = analyse_single_fits(f)
        ref_wave_comparator.compare_waves(r)
    

if __name__ == "__main__":

    parser = ap.ArgumentParser("Shack-Hartmann analyser")
    parser.add_argument("--runs_path", default="runs", type=Path)
    args = parser.parse_args()

    for dir in args.runs_path.glob("*"):
        print(dir)
        analyse_single_run(dir)