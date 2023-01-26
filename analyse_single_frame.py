import argparse as ap
from wavey import microlens_array, analyser
from astropy.io import fits
from pathlib import Path
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
                    sensor_y_px_count=fits_file[0].data.shape[0], sensor_x_px_count=fits_file[0].data.shape[1])

    r = a.analyse(fits_file[0].data)