import numpy as np
from prysm.polynomials import zernike_nm, zernike_nm_der_sequence, lstsq, nm_to_name, fringe_to_nm, zernike_nm_der
from prysm import geometry
import matplotlib.pyplot as plt

class WavefrontAnalysis:
    def __init__(self, name, wavelength, zern_waves) -> None:
        self.name = name
        self.wavelength = wavelength
        self.z_wv = zern_waves
    

    def gen_wf(self, r, t):
        Z = np.zeros_like(r)
        for nm in (self.z_wv):
            Z += (self.z_wv[nm] * zernike_nm(nm[0], nm[1], r, t))

    def gen_der_wf(self, r, t):
        dZdr = np.zeros_like(r)
        dZdt = np.zeros_like(r)
        for nm in (self.z_wv):
            (dr, dt) = (zernike_nm_der(nm[0], nm[1], r, t))
            dZdr += (self.z_wv[nm] * dr)
            dZdt += (self.z_wv[nm] * dt)

        return (dZdr, dZdt)

def plot_zernike(z, r, t):
     
    Z_img = z.copy()
    zernike_mask = geometry.circle(1, r)
    Z_img[zernike_mask!=1]=np.nan


    fig = plt.figure()  
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Reconstructed wavefront error (in waves)")
    ax.imshow(Z_img)
    plt.show()

def plot_zernike_der(dr, dt, r, t):
     
    dZdr_img = dr.copy()
    dZdt_img = dt.copy()
    zernike_mask = geometry.circle(1, r)
    dZdr_img[zernike_mask!=1]=np.nan
    dZdt_img[zernike_mask!=1]=np.nan


    fig = plt.figure()  
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("dZ/dr")
    ax.imshow(dZdr_img)
    
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title("dZ/dt")
    ax.imshow(dZdt_img)
    plt.show()

    