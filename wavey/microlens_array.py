from prysm import detector, geometry
from prysm.segmented import _local_window
from math import ceil
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



#right now it only supports rectangular grid microlens arrays
#with some cleanup in the future should be able to stuff like hexagonal packing with small changes
class MicrolensArray:
    def __init__(self, pitch_x, pitch_y, lens_centres_xy, lens_cell_corners, focal_len):
        self.pitch_x = pitch_x #TODO: come up with a better way pitch handling for forward pass sims
        self.pitch_y = pitch_y
        self.lens_centres_xy = lens_centres_xy
        self.lens_cell_corners = lens_cell_corners
        self.fl = focal_len

    def get_lens_centres(self):
        return self.lens_centres_xy
        
    def get_lens_cell_corners(self):
        return self.lens_cell_corners

    def shack_hartmann_phase_screen(self, x, y, wavelength, aperture=geometry.rectangle):

        dx = x[0,1] - x[0,0]
        min_pitch = min(self.pitch_x, self.pitch_y)
        samples_per_lenslet = int(min_pitch / dx + 1) # ensure safe rounding

       
        cx = ceil(x.shape[1]/2)
        cy = ceil(y.shape[0]/2)
        total_phase = np.zeros_like(x)
        for lenslet in self.lens_centres_xy:
            xx = lenslet[0]
            yy = lenslet[1]
            iy, ix = _local_window(cy, cx, (xx,yy), dx, samples_per_lenslet, x, y)
            lx = x[iy, ix] - xx
            ly = y[iy, ix] - yy
            rsq = lx * lx + ly * ly
            phase = rsq / (2*self.fl)
            phase *= aperture(width=self.pitch_x, height=self.pitch_y, x=lx, y=ly)
            total_phase[iy, ix] += phase

        prefix = -1j * 2 * np.pi/(wavelength/1e3)
        return np.exp(prefix*total_phase)


class ThorlabsMLA300(MicrolensArray):
    def __init__(self):
        pitch_x = 0.3
        pitch_y = 0.3

        n_x = 33
        n_y = 33

        ulens_centres_x = np.linspace(-((pitch_x*(n_x-1))/2), ((pitch_x*(n_x-1))/2), n_x)
        ulens_centres_y = np.linspace(-((pitch_y*(n_y-1))/2), ((pitch_y*(n_y-1))/2), n_y)

        micro_lens_positions_x, micro_lens_positions_y = np.meshgrid(ulens_centres_x, ulens_centres_y)

        micro_lens_positions_x = micro_lens_positions_x.flatten()
        micro_lens_positions_y = micro_lens_positions_y.flatten()
        lens_centres_xy = []
        for i in range(micro_lens_positions_x.size):
            lens_centres_xy.append((micro_lens_positions_x[i], micro_lens_positions_y[i]))

        lens_cell_corners = [[-pitch_x/2,pitch_y/2],[pitch_x/2,pitch_y/2],[pitch_x/2,-pitch_y/2],[-pitch_x/2,-pitch_y/2]]
        super().__init__(pitch_x, pitch_y, lens_centres_xy, lens_cell_corners, 14.7)

class ThorlabsMLA1(MicrolensArray):
    def __init__(self):
        pitch_x = 1.0
        pitch_y = 1.4

        n_x = 10
        n_y = 7

        ulens_centres_x = np.linspace(-((pitch_x*(n_x-1))/2), ((pitch_x*(n_x-1))/2), n_x)
        ulens_centres_y = np.linspace(-((pitch_y*(n_y-1))/2), ((pitch_y*(n_y-1))/2), n_y)

        micro_lens_positions_x, micro_lens_positions_y = np.meshgrid(ulens_centres_x, ulens_centres_y)

        micro_lens_positions_x = micro_lens_positions_x.flatten()
        micro_lens_positions_y = micro_lens_positions_y.flatten()
        lens_centres_xy = []
        for i in range(micro_lens_positions_x.size):
            lens_centres_xy.append((micro_lens_positions_x[i], micro_lens_positions_y[i]))

        lens_cell_corners = [[-pitch_x/2,pitch_y/2],[pitch_x/2,pitch_y/2],[pitch_x/2,-pitch_y/2],[-pitch_x/2,-pitch_y/2]]
        super().__init__(pitch_x, pitch_y, lens_centres_xy, lens_cell_corners, 4.7)