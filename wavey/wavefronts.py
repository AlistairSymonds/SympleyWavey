import numpy as np
from prysm.polynomials import zernike_nm, zernike_nm_der_sequence, lstsq, nm_to_name, fringe_to_nm, zernike_nm_der
from prysm import geometry, coordinates
import matplotlib.pyplot as plt
from matplotlib import colors
  

def plot_wavefront(z, x, y, mask=None, ax=None, vmin=None, vmax=None, **plt_kwargs):

    r, t = coordinates.cart_to_polar(x, y)
    
    Z_img = z.copy()
    Z_stats = z.copy()
    if mask is None:
        rmax = np.max(x)
        mask = geometry.circle(rmax, r)


    Z_img[mask!=1]=np.nan
    Z_stats[mask!=1]=0

    if ax == None:
        ax = plt.gca()
    ax.set_title("Wavefront deviation")
    color_norm=colors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

    var = np.var(Z_stats)
    rms = np.sqrt(var)
    z_min = np.min(Z_stats)
    z_max = np.max(Z_stats)
    peak_to_valley = z_max - z_min

    stats_str = f"PtV: {peak_to_valley:.03f}\nRMS: {rms:.03f}"
    x_ticks = x[0]
    y_ticks = y[:,0]
    xmin = np.min(x_ticks)
    xmax = np.max(x_ticks)
    ymin = np.min(y_ticks)
    ymax = np.max(y_ticks)
    ax.imshow(Z_img, cmap='Spectral', norm=color_norm, extent=[xmin, xmax, ymax, ymin], origin='lower')
    plt.text(0,0, stats_str, transform=ax.transAxes)
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
    def __init__(self, name, wavelength, zernikes, wf_radius, x, y, aperture_mask=None) -> None:
        self.zernikes = zernikes
        
        r, t = coordinates.cart_to_polar(x, y)
        phs = np.zeros_like(r)

        for j in (zernikes):
            n, m = fringe_to_nm(j)
            phs += (zernikes[j] * zernike_nm(n, m, r/wf_radius, t))
    
        opd_nanometres = phs *  wavelength * 1e3

        if aperture_mask is None:
            aperture_mask = geometry.circle(wf_radius, r)

        super().__init__(name, wavelength, opd_nanometres, x, y, aperture_mask=aperture_mask)

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
    def __init__(self, reference_wave: WavefrontAnalysis) -> None:
        self.ref = reference_wave
    
    def compare_waves(self, test_wavefront: WavefrontAnalysis, use_lambda_units=False):
        assert(self.ref.wf.shape == test_wavefront.wf.shape)

        ref_wf_masked = self.ref.wf.copy()
        test_wf_masked = test_wavefront.wf.copy()
        
        ref_wf_masked[self.ref.aperture!=1]=np.nan
        test_wf_masked[self.ref.aperture!=1]=np.nan

        delta_wf = ref_wf_masked - test_wf_masked
        
        units_str = "nm"
        if use_lambda_units:
            assert(self.ref.wavelength == test_wavefront.wavelength)
            delta_wf       *= (1.0/(self.ref.wavelength * 1e3)) 
            ref_wf_masked  *= (1.0/(self.ref.wavelength * 1e3))
            test_wf_masked *= (1.0/(self.ref.wavelength * 1e3))
            units_str = "waves, lambda = " + str(self.ref.wavelength*1e3) + "nm"

        vmin = np.nanmin(np.array([delta_wf, ref_wf_masked, test_wf_masked]))
        vmax = np.nanmax(np.array([delta_wf, ref_wf_masked, test_wf_masked]))

        fig = plt.figure(figsize=(15,5))  
        fig.suptitle(f"Wavefront OPD errors [{units_str}]")
        ref_ax = fig.add_subplot(1, 3, 1)
        plot_wavefront(ref_wf_masked, mask=self.ref.aperture,  x=self.ref.x, y=self.ref.y, ax=ref_ax, vmax=vmax, vmin=vmin)
        ref_ax.set_title(self.ref.name)

        test_ax = fig.add_subplot(1, 3, 2)
        test_ax = plot_wavefront(test_wf_masked, mask=self.ref.aperture, x=self.ref.x, y=self.ref.y, ax=test_ax, vmax=vmax, vmin=vmin)
        test_ax.set_title(test_wavefront.name)

        delta_ax = fig.add_subplot(1, 3, 3)
        plot_wavefront(delta_wf, mask=self.ref.aperture,  x=self.ref.x, y=self.ref.y, ax=delta_ax, vmax=vmax, vmin=vmin)
        delta_ax.set_title("delta between ref and test")
        
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])

        fig.colorbar(test_ax.get_images()[0], cax=cbar_ax)
        plt.show()