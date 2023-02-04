import numpy as np
from prysm.polynomials import zernike_nm, zernike_nm_der_sequence, lstsq, nm_to_name, fringe_to_nm, zernike_nm_der
from prysm import geometry, coordinates
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path

def plot_frame(fringes, title, file_path):
    print(f"Making frame: {file_path} with {fringes}")
    x, y = coordinates.make_xy_grid(2048, diameter=2)
    r, t = coordinates.cart_to_polar(x, y)

    z = np.zeros((2048, 2048))
    dr = np.zeros((2048, 2048))
    dt = np.zeros((2048, 2048))
    for J in fringes:
        

        (n, m) = fringe_to_nm(J)

        _z = zernike_nm(n, m, r, t)
        (_dr, _dt) = zernike_nm_der(n, m, r, t)

        z  += (fringes[J] * _z)
        dr += (fringes[J] * _dr)
        dt += (fringes[J] * _dt)

    # Mask out a circle for plotting
    mask = geometry.circle(1, r)

    z[mask!=1]=np.nan
    dr[mask!=1]=np.nan
    dt[mask!=1]=np.nan

    #  Plotting
    width = 20
    fig = plt.figure(figsize=(width,width/3))
    ticks = [0, 1024, 2047]
    labels = [-1, 0, 1]

    color_norm=colors.TwoSlopeNorm(vcenter=0.0)
    z_ax = fig.add_subplot(1, 3, 1)
    z_ax.imshow(z, norm=color_norm, cmap="RdBu")
    z_ax.set_title("Phase")
    fig.colorbar(z_ax.get_images()[0])


    color_norm=colors.TwoSlopeNorm(vcenter=0.0)
    dr_ax = fig.add_subplot(1, 3, 2)
    dr_ax.imshow(dr, norm=color_norm, cmap="RdBu")
    dr_ax.set_title("Deriv Phase/radial")
    fig.colorbar(dr_ax.get_images()[0])


    color_norm=colors.TwoSlopeNorm(vcenter=0.0)
    dt_ax = fig.add_subplot(1, 3, 3)
    if m == 0:
        color_norm=colors.TwoSlopeNorm(vcenter=0.0)

    dt_ax.imshow(dt, norm=color_norm, cmap="RdBu")
    dt_ax.set_title("Deriv Phase/theta")
    fig.colorbar(dt_ax.get_images()[0])

    fig.suptitle(title)
    plt.savefig(file_path)
    #plt.show()

zernikes = range(1, 10)

stable_frames = 1
lerp_frames = 30 
frames_per_zernike = lerp_frames + stable_frames
frame_count = 0

frames_dir = Path("anim")
frames_dir.mkdir(exist_ok=True, parents=True)
for idx, j in enumerate(zernikes):
    for f in range(0, frames_per_zernike):
        title = ""
        z = {}
        if(f < stable_frames):
            n, m = fringe_to_nm(j)
            z_name = nm_to_name(n, m)
            title = f"(n: {n}, m: {m}) - {z_name}"
            z[j] = 1
        else:
            lerp_frame = f - stable_frames
            lerp_percent = float(lerp_frame)/float(lerp_frames)
            z[j] = 1 - lerp_percent
            z[zernikes[idx+1]] = lerp_percent


        frame_path = frames_dir / f"frame__{frame_count:08}.png"
        plot_frame(z, title, frame_path)
        frame_count += 1
        
        