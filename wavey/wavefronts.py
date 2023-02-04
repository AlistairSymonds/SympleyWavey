import numpy as np
from prysm.polynomials import zernike_nm, zernike_nm_der_sequence, lstsq, nm_to_name, fringe_to_nm, zernike_nm_der
from prysm import geometry, coordinates
import matplotlib.pyplot as plt
from matplotlib import colors


def plot_wavefront(z, x, y, zernike_mask=None, ax=None, **plt_kwargs):

    r, t = coordinates.cart_to_polar(x, y)
    
    Z_img = z.copy()
    if zernike_mask is None:
        rmax = np.max(r)
        zernike_mask = geometry.circle(rmax, r)


    Z_img[zernike_mask!=1]=np.nan

    if ax == None:
        ax = plt.gca()
    ax.set_title("Wavefront deviation")
    color_norm=colors.TwoSlopeNorm(vcenter=0.0)

    ax.imshow(Z_img, cmap='Spectral', norm=color_norm)
    x_tick = x[0]
    y_tick = y[:,0]

    #ax.set_xticks(x_tick)
    #ax.set_yticks(y_tick)
    return ax


class WavefrontAnalysis:
    def __init__(self, name, wavelength, wavefront, x, y, aperture_mask=None) -> None:
        self.name = name
        self.wavelength = wavelength
        
        self.wf = wavefront
        self.x = x
        self.y = y

        if aperture_mask is None:
            r, t = coordinates.cart_to_polar(x, y)
            rmax = np.max(x)
            aperture_mask = geometry.circle(rmax, r)
        self.aperture = aperture_mask

    def plot_wavefront(self, ax=None,  **plt_kwargs):
        return plot_wavefront(self.wf, self.x, self.y, zernike_mask=self.aperture, ax=ax, plt_kwargs=plt_kwargs)


class ZernikeWavefrontAnalysis(WavefrontAnalysis):
    def __init__(self, name, wavelength, fringe_indicies, weights, radius, aperture_mask=None) -> None:
        self.fi = fringe_indicies
        self.fringe_weights = weights

        wf = np.zeros_like(r)
        for j in (fringe_indicies):
            n, m = fringe_to_nm(j)
            wf += (weights[j] * zernike_nm(n, m, r, t))
            
        wf *= wavelength
        super().__init__(name, wavelength, wf, aperture_mask)

    def gen_der_wf(self, r, t):
        dZdr = np.zeros_like(r)
        dZdt = np.zeros_like(r)
        for j in (self.fringe_indicies):
            n, m = fringe_to_nm(j)
            (dr, dt) = (zernike_nm_der(n, m, r, t))
            dZdr += (self.z_wv[j] * dr)
            dZdt += (self.z_wv[j] * dt)

        return (dZdr, dZdt)

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

class WavefrontComparator:
    def __init__(self, reference_wave: WavefrontAnalysis, ref_x, ref_y) -> None:
        self.ref = reference_wave
        self.ref_x = ref_x
        self.ref_y = ref_y
    
    def compare_waves(self, test_wavefront):
        assert(self.ref.shape == test_wavefront.shape)

        delta_wf = self.ref - test_wavefront

        fig = plt.figure()  
        ref_ax = fig.add_subplot(1, 3, 1)
        plot_wavefront(self.ref, ax=ref_ax)

        test_ax = fig.add_subplot(1, 3, 2)
        plot_wavefront(test_wavefront, ax=test_ax)
        
        delta_ax = fig.add_subplot(1, 3, 3)
        plot_wavefront(delta_wf, ax=delta_ax)

        plt.show()