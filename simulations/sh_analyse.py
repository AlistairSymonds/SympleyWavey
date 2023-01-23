from tkinter.messagebox import NO
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits
import scipy.ndimage
from prysm import geometry, coordinates

class sh_analyser:
    #assuming 0,0 is top left....
    #never converting back from pixels to world space :)
    def optical_coords_to_pixel(x,y,sensor_x_pitch, sensor_y_pitch, sensor_x_px_count, sensor_y_px_count):
        xpx = x /sensor_x_pitch
        ypx = y /sensor_y_pitch
        centre_offset_x = sensor_x_px_count / 2
        centre_offset_y = sensor_y_px_count / 2
        return (xpx+centre_offset_x, centre_offset_y-ypx)

    def cartesian_grad_to_polar_grad(x, y, dx, dy):
        r, t = coordinates.cart_to_polar(x, y, vec_to_grid=False)
        grad_end_pts_x = x + dx 
        grad_end_pts_y = y + dy 
        grad_end_pts_r, grad_end_pts_t = coordinates.cart_to_polar(grad_end_pts_x, grad_end_pts_y, vec_to_grid=False)

        dr = r - grad_end_pts_r
        dt = t - grad_end_pts_t
        
        dt = (dt + np.pi) % (2 * np.pi) - np.pi
        return dr, dt

    def polar_gradient_to_cartesian_grad(r, t, dr, dt):
        x, y = coordinates.polar_to_cart(r, t)
        grad_end_pts_r = r + dr
        grad_end_pts_t = t + dt 
        grad_end_pts_x, grad_end_pts_y = coordinates.polar_to_cart(grad_end_pts_r, grad_end_pts_t)

        dx = x - grad_end_pts_x
        dy = y - grad_end_pts_y
        return dx, dy

    def analyse():
        sensor_x = 10
        sensor_y = 10

        pitch_x = 0.3
        pitch_y = pitch_x

        n_x = 33
        n_y = 33

        ulens_centres_x = np.linspace(-((pitch_x*(n_x-1))/2), ((pitch_x*(n_x-1))/2), n_x)
        ulens_centres_y = np.linspace(-((pitch_y*(n_y-1))/2), ((pitch_y*(n_y-1))/2), n_y)

        #ulens_centres_x = [ -3.0, -2.7]
        #ulens_centres_y = [0]

        ulens_cell_corners = [[-0.5,0.5],[0.5,0.5],[0.5,-0.5],[-0.5,-0.5]]
        micro_lens_positions_x, micro_lens_positions_y = np.meshgrid(ulens_centres_y, ulens_centres_x)


        #micro_lens_positions_x = np.array([0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.2, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.2])
        #micro_lens_positions_y = np.array([0,   0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,    0,  0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.2])

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
            ulens_aabb.append([mins[0], mins[1], maxs[0], maxs[1], ulens[0], ulens[1]]) #min_x, min_y, max_x, max_y, centre x, centre y

        cell_dbg_d = sh_capture[0].data
        cell_infos = []
        for ulens in ulens_aabb:
            min_cnr_pixels = np.array(optical_coords_to_pixel(ulens[0],ulens[3],pixel_pitch_x, pixel_pitch_y, x_res, y_res)).astype(int)
            max_cnr_pixels = np.array(optical_coords_to_pixel(ulens[2],ulens[1],pixel_pitch_x, pixel_pitch_y, x_res, y_res)).astype(int)
            cell = cell_dbg_d[min_cnr_pixels[1]:max_cnr_pixels[1], min_cnr_pixels[0]:max_cnr_pixels[0]]

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
            centroid = (centroid_px-0.5) * pixel_pitch_x

            #need to flip y to go from pixel (Top left, pos down and right) to optical coords (0,0 centre)
            cell_size_y = ulens[3] - ulens[1]
            cell_size_x = ulens[2] - ulens[0]
            centroid[0] = cell_size_y - centroid[0]
            centroid[0] = centroid[0] - (cell_size_y/2)
            centroid[1] = centroid[1] - (cell_size_x/2)
            intensities = np.sort(cell, axis=None)[-8:]

            cell_info = {}
            cell_info["centroid_xy"] = np.array([centroid[1], centroid[0]])
            cell_info["lens_centre_xy"] = np.array([ulens[4],ulens[5]])
            cell_info["size_xy"] = np.array([ulens[2]-ulens[0], ulens[3]-ulens[1]])
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
            cr, ct = coordinates.cart_to_polar(c["lens_centre_xy"][0],c["lens_centre_xy"][1] )
            if cr <= 4.1: #c['intensities'] > cell_int_thresh:
                cells_for_calc.append(c)


        final_img = sh_capture[0].data
        expected_pos_img_px = np.zeros((len(cells_for_calc),2))
        deviation_img_px    = np.zeros((len(cells_for_calc),2))
        for cell_idx in range(len(cells_for_calc)):
            c = cells_for_calc[cell_idx]

            expected_pos_image = c["lens_centre_xy"] 
            centroid_img = c["centroid_xy"] + c["lens_centre_xy"]
            expected_pos_img_px[cell_idx] = np.array(optical_coords_to_pixel(expected_pos_image[0],expected_pos_image[1],pixel_pitch_x, pixel_pitch_y, x_res, y_res))
            deviation_img_px[cell_idx]    = np.array(optical_coords_to_pixel(centroid_img[0],centroid_img[1],pixel_pitch_x, pixel_pitch_y, x_res, y_res))
        
            #plt.quiver(expected_pos_img_px[0], expected_pos_img_px[1], deviation_img_px[0], deviation_img_px[1], alpha=0.5, color='white')


        plt.imshow(final_img)
        plt.plot(expected_pos_img_px[:,0], expected_pos_img_px[:,1], marker ='o', color='blue', alpha=0.5,linestyle='None', zorder=4)
        plt.plot(deviation_img_px[:,0], deviation_img_px[:,1], marker ='+', color='red', alpha=0.5, linestyle='None',zorder=6)
        plt.show()

        #now actually plot a wavefront

        x = np.zeros((len(cells_for_calc)))
        y = np.zeros((len(cells_for_calc)))
        r = np.zeros((len(cells_for_calc)))
        t = np.zeros((len(cells_for_calc)))
        slopes_xy = np.zeros( (len(cells_for_calc),2))





        for cell_idx in range(len(cells_for_calc)):
            x[cell_idx] = cells_for_calc[cell_idx]["lens_centre_xy"][0]
            y[cell_idx] = cells_for_calc[cell_idx]["lens_centre_xy"][1]

            #slope_xy  = (cells_for_calc[cell_idx]["centroid_xy"]+ cells_for_calc[cell_idx]["corner_xy"]) - cells_for_calc[cell_idx]["lens_centre_xy"]
            r[cell_idx], t[cell_idx] = coordinates.cart_to_polar(x[cell_idx], y[cell_idx])

            centroid_in_img = cells_for_calc[cell_idx]["centroid_xy"]  + cells_for_calc[cell_idx]["lens_centre_xy"]
            expected_in_img = cells_for_calc[cell_idx]["lens_centre_xy"]
            slopes_xy[cell_idx] = centroid_in_img - expected_in_img



        slopes_xy = slopes_xy.T

        print(slopes_xy)
        dr, dt = cartesian_grad_to_polar_grad(x/5, y/5, slopes_xy[0], slopes_xy[1])

        from prysm.polynomials import (
            fringe_to_nm,
            zernike_nm_der_sequence,
            zernike_nm_sequence,
            lstsq,
            sum_of_2d_modes,
            nm_to_name
        )
        from prysm.polynomials.zernike import barplot_magnitudes, zernikes_to_magnitude_angle

        from prysm.util import rms


        normalization_radius = 5 #np.max(r)
        r = r / normalization_radius

        fringe_indices = range(1,10)
        nms = [fringe_to_nm(j) for j in fringe_indices]
        #nms = [[1,1],[4,0]]

        print("Fitting the following zernikes:")
        for zpol in nms:
            print(zpol)
            print(nm_to_name(zpol[0], zpol[1]))

        modes = list(zernike_nm_der_sequence(nms, r, t))
        for m in modes:
            if False:
                fig = plt.figure(figsize=plt.figaspect(1))
                mxy = polar_gradient_to_cartesian_grad(r,t,m[0],m[1])
                ax = fig.add_subplot(2, 2, 1, projection='3d')
                ax.plot(r,t,m[0], linestyle='None', marker = '+')
                
                ax = fig.add_subplot(2, 2, 2, projection='3d')
                ax.plot(r,t, dr, linestyle='None', marker = 'o')

                ax = fig.add_subplot(2, 2, 3, projection='3d')
                ax.plot(r,t,m[1], linestyle='None', marker = '+')

                ax = fig.add_subplot(2, 2, 4, projection='3d')
                ax.plot(r,t, dt, linestyle='None', marker = 'o')

                plt.show()
                m_np = np.array(m).T
                measured_polar_slopes = np.array([dr, dt]).T
                fit = np.linalg.lstsq(m_np, measured_polar_slopes)
                print(fit[0])

        fit = lstsq(modes, np.array([dr, dt]))
        print(fit)
        pak = [[*nm, c] for nm, c in zip(nms, fit)]
        magnitudes = zernikes_to_magnitude_angle(pak)
        barplot_pak = {k: v[0] for k, v in magnitudes.items()}
        barplot_magnitudes(barplot_pak)
        plt.show()

if __name__ == "__main__":
    import argparse as ap
    
