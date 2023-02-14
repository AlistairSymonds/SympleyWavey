import numpy as np
from prysm.polynomials.qpoly import compute_z_zprime_Q2d, Q2d_nm_c_to_a_b, Q2d
from prysm.polynomials import fringe_to_nm, nm_to_name
from prysm import geometry, coordinates
import matplotlib.pyplot as plt
from matplotlib import colors
import random

def plot_q2d(fringes, coeffs):
    
    x, y = coordinates.make_xy_grid(2048, diameter=2)
    r, t = coordinates.cart_to_polar(x, y)

    z = np.zeros((2048, 2048))
    dr = np.zeros((2048, 2048))
    dt = np.zeros((2048, 2048))
    
    nms = []
    for J, C in zip(fringes, coeffs):
        (n, m) = fringe_to_nm(J)
        if C != 0:
            print(f"(n: {n}, m: {m}) - {C}")
        nms.append((n, m))


    (cms, ams, bms )= Q2d_nm_c_to_a_b(nms, coeffs)
    (_z, _dr, _dt) = compute_z_zprime_Q2d(cms, ams, bms, r, t)

    z  = (_z)
    dr = (_dr)
    dt = (_dt)

    # Mask out a circle for plotting
    mask = geometry.circle(1, r)

    z[mask!=1]=np.nan
    dr[mask!=1]=np.nan
    dt[mask!=1]=np.nan

    #  Plotting
    width = 20
    fig = plt.figure(figsize=(width,width/3))

    color_norm=colors.TwoSlopeNorm(vcenter=0.0)
    z_ax = fig.add_subplot(1, 3, 1)
    z_ax.imshow(z, norm=color_norm, cmap="RdBu", extent=[-1, 1, -1, 1])
    z_ax.set_title("Phase")
    fig.colorbar(z_ax.get_images()[0])
    #plt.axis('off')


    color_norm=colors.TwoSlopeNorm(vcenter=0.0)
    dr_ax = fig.add_subplot(1, 3, 2)
    dr_ax.imshow(dr, norm=color_norm, cmap="RdBu", extent=[-1, 1, -1, 1])
    dr_ax.set_title("Radial derivative of phase")
    fig.colorbar(dr_ax.get_images()[0])
    #plt.axis('off')


    color_norm=colors.TwoSlopeNorm(vcenter=0.0)
    dt_ax = fig.add_subplot(1, 3, 3)

    dt_ax.imshow(dt, norm=color_norm, cmap="RdBu", extent=[-1, 1, -1, 1])
    dt_ax.set_title("Tangential derivative of phase")
    fig.colorbar(dt_ax.get_images()[0])

    #plt.axis('off')
    #plt.tight_layout()
    fig.suptitle("Q2D stuff")
    #plt.savefig(file_path)
    plt.show()
    plt.close(fig)



zernikes = np.arange(1, 36)
coeffs = np.zeros(zernikes.shape)
for J, C in zip(zernikes, coeffs):
    (n, m) = fringe_to_nm(J)
    name = nm_to_name(n, m)
    print(f"(n: {n}, m: {m}) - {name}")
    

for num in range(3):
    idx = random.randint(0, len(coeffs)-1)

plot_q2d(zernikes, coeffs)
        
        