from tkinter.messagebox import NO
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits
import scipy.ndimage


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

n_x = 33
n_y = 33

ulens_centres_x = np.linspace(-((pitch_x*(n_x-1))/2), ((pitch_x*(n_x-1))/2), n_x)
ulens_centres_y = np.linspace(-((pitch_y*(n_y-1))/2), ((pitch_y*(n_y-1))/2), n_y)

#ulens_centres_x = [-4.5, -3.0, -1.5, 0, 4.5, 3.0, 1.5]
#ulens_centres_y = 1.5

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
    maxs = np.max(lens_corners, axis=0)     
    mins = np.min(lens_corners, axis=0)
    ulens_aabb.append([mins[0], mins[1], maxs[0], maxs[1]]) #min_x, min_y, max_x, max_y

cell_dbg_d = sh_capture[0].data
cell_infos = []
for ulens in ulens_aabb:
    min_cnr_pixels = np.array(optical_coords_to_pixel(ulens[0],ulens[1],pixel_pitch_x, pixel_pitch_y, x_res, y_res)).astype(int)
    max_cnr_pixels = np.array(optical_coords_to_pixel(ulens[2],ulens[3],pixel_pitch_x, pixel_pitch_y, x_res, y_res)).astype(int)
    cell = cell_dbg_d[min_cnr_pixels[0]:max_cnr_pixels[0], min_cnr_pixels[1]:max_cnr_pixels[1]]

    #temporary noise and constant offset
    cell = cell + np.random.normal(loc=0, scale=0.0001, size=cell.shape) + 0.05

    img_mean = np.mean(cell)
    img_max = np.max(cell)
    img_var = np.var(cell)

    max_to_mean = img_max - img_mean

    source_threshold = img_mean + max_to_mean * 0.1

    cell_thresholded = (cell > source_threshold)
    cell = cell_thresholded * cell
    #plt.imshow(cell)
    #plt.show()
    centroid_px = np.array(scipy.ndimage.center_of_mass(cell))
    centroid = centroid_px * pixel_pitch_x
    intensities = np.sort(cell, axis=None)[-8:]

    cell_info = {}
    cell_info["centroid_xy"] = np.array([centroid[1], centroid[0]])
    cell_info["corner_xy"] = np.array([ulens[1],ulens[0]])
    cell_info["size_xy"] = np.array([ulens[2]-ulens[0], ulens[3]-ulens[1]])
    cell_info["corner_px_xy"] = (min_cnr_pixels[0], min_cnr_pixels[1])
    cell_info["size_px_xy"] = (max_cnr_pixels[0]-min_cnr_pixels[0], max_cnr_pixels[1]-min_cnr_pixels[1])
    cell_info["intensities"] = np.sum(intensities)
    cell_infos.append(cell_info)


min_int = np.Infinity
max_int = 0
for c in cell_infos:
    if c['intensities'] < min_int:
        min_int = c['intensities']
    if c['intensities'] > max_int:
        max_int = c['intensities']


cell_int_thresh = ((max_int-min_int) * 0.1) + min_int
cells_for_calc = []
for c in cell_infos:
     if c['intensities'] > cell_int_thresh:
        cells_for_calc.append(c)


final_img = sh_capture[0].data
plt.imshow(final_img)
expected_pos_img_px = np.zeros((len(cells_for_calc),2))
deviation_img_px    = np.zeros((len(cells_for_calc),2))
for cell_idx in range(len(cells_for_calc)):
    c = cells_for_calc[cell_idx]
    expected_pos = c["size_xy"]/2
    deviation = c["centroid_xy"] - expected_pos

    expected_pos_image = expected_pos + c["corner_xy"]
    centroid_img = c["centroid_xy"] + c["corner_xy"]
    expected_pos_img_px[cell_idx] = np.array(optical_coords_to_pixel(expected_pos_image[0],expected_pos_image[1],pixel_pitch_x, pixel_pitch_y, x_res, y_res))
    deviation_img_px[cell_idx]    = np.array(optical_coords_to_pixel(centroid_img[0],centroid_img[1],pixel_pitch_x, pixel_pitch_y, x_res, y_res))
   
    #plt.quiver(expected_pos_img_px[0], expected_pos_img_px[1], deviation_img_px[0], deviation_img_px[1], alpha=0.5, color='white')

plt.plot(expected_pos_img_px[:,0], expected_pos_img_px[:,1], marker ='o', color='blue', alpha=0.5,linestyle='None', zorder=4)
plt.plot(deviation_img_px[:,0], deviation_img_px[:,1], marker ='+', color='red', alpha=0.5, linestyle='None',zorder=6)
plt.show()