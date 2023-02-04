import numpy as np
from prysm.polynomials import zernike_nm, zernike_nm_der_sequence, lstsq, nm_to_name, fringe_to_nm, zernike_nm_der
from prysm import geometry, coordinates
import matplotlib.pyplot as plt
from matplotlib import colors

x, y = coordinates.make_xy_grid(2048, diameter=2)
r, t = coordinates.cart_to_polar(x, y)


Z_j = 9

(n, m) = fringe_to_nm(Z_j)

z = zernike_nm(n, m, r, t)
(dr, dt) = zernike_nm_der(n, m, r, t)

# Mask out a circle for plotting
mask = geometry.circle(1, r)

z[mask!=1]=np.nan
dr[mask!=1]=np.nan
dt[mask!=1]=np.nan

#  Plotting
fig = plt.figure(figsize=(15,5))
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
    color_norm=colors.TwoSlopeNorm(vcenter=0.0, vmax=0.001, vmin=-0.001)

dt_ax.imshow(dt, norm=color_norm, cmap="RdBu")
dt_ax.set_title("Deriv Phase/theta")
fig.colorbar(dt_ax.get_images()[0])


plt.show()