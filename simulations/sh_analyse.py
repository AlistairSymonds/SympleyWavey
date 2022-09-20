import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits
import scipy.ndimage
import cv2 as cv


#assuming 0,0 is top left....
def optical_coords_to_pixel(x,y,sensor_x_pitch, sensor_y_pitch, sensor_x_px_count, sensor_y_px_count):
    xpx = x /sensor_x_pitch
    ypx = y /sensor_y_pitch
    centre_offset_x = sensor_x_px_count / 2
    centre_offset_y = sensor_y_px_count / 2
    return (xpx+centre_offset_x, ypx+centre_offset_y)

sensor_x = 10
sensor_y = 10

pitch_x = 0.3
pitch_y = pitch_x

n_x = 3
n_y = 1

ulens_centres_x = np.linspace(-((pitch_x*(n_x-1))/2), ((pitch_x*(n_x-1))/2), n_x)
ulens_centres_y = np.linspace(-((pitch_y*(n_y-1))/2), ((pitch_y*(n_y-1))/2), n_y)

ulens_centres_x = [-4.5, -3.0, -1.5, 0]
ulens_centres_y = 0

ulens_cell_corners = [[-0.5,0.5],[0.5,0.5],[0.5,-0.5],[-0.5,-0.5]]
micro_lens_positions_x, micro_lens_positions_y = np.meshgrid(ulens_centres_y, ulens_centres_x)

micro_lens_positions_x = micro_lens_positions_x.flatten()
micro_lens_positions_y = micro_lens_positions_y.flatten()

microlens_positions = []
for i in range(micro_lens_positions_x.size):
    microlens_positions.append((micro_lens_positions_x[i], micro_lens_positions_y[i]))
    #if i % 4:
    #    plt.scatter(microlens_positions[i][0], microlens_positions[i][1])

#plt.show()

sh_capture = astropy.io.fits.open("C:/Users/alist/Documents/GitHub/SympleyWavey/prysm_sh_capture.fits")



x_res = sh_capture[0].data.shape[0]
y_res = sh_capture[0].data.shape[1]
pixel_pitch_x = sensor_x / x_res
pixel_pitch_y = sensor_y / y_res
#cell_size_px_x = pixel_pitch_x // pitch_x
#cell_size_px_y = pixel_pitch_y // pitch_y
ulens_aabb = []
for ulens in microlens_positions:
    lens_corners = np.empty(shape=(len(ulens_cell_corners),2))
    for cnr_idx in range(len(ulens_cell_corners)):
        cnr = ulens_cell_corners[cnr_idx]
        cnr_x = (cnr[0] * pitch_x) + ulens[0]
        cnr_y = (cnr[1] * pitch_y) + ulens[1]
        lens_corners[cnr_idx][0] = cnr_x
        lens_corners[cnr_idx][1] = cnr_y
    print(lens_corners)
    maxs = np.max(lens_corners, axis=0)     
    mins = np.min(lens_corners, axis=0)
    ulens_aabb.append([mins[0], mins[1], maxs[0], maxs[1]]) #min_x, min_y, max_x, max_y

cell_dbg_d = sh_capture[0].data
for ulens in ulens_aabb:
    print(ulens)
    min_cnr_pixels = np.array(optical_coords_to_pixel(ulens[0],ulens[1],pixel_pitch_x, pixel_pitch_y, x_res, y_res)).astype(int)
    max_cnr_pixels = np.array(optical_coords_to_pixel(ulens[2],ulens[3],pixel_pitch_x, pixel_pitch_y, x_res, y_res)).astype(int)
    print(min_cnr_pixels)
    print(max_cnr_pixels)
    cell = cell_dbg_d[min_cnr_pixels[0]:max_cnr_pixels[0], min_cnr_pixels[1]:max_cnr_pixels[1]]
    plt.imshow(cell)
    plt.show()   
    

img_mean = np.mean(sh_capture[0].data)
img_max = np.max(sh_capture[0].data)
img_var = np.var(sh_capture[0].data)

max_to_mean = img_max - img_mean

source_threshold = img_mean + max_to_mean * 0.1
