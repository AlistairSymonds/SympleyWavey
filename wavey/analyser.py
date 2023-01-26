import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from prysm import geometry, coordinates
from prysm.polynomials import zernike_nm, zernike_nm_der_sequence, lstsq, nm_to_name, fringe_to_nm

from prysm.polynomials.zernike import barplot_magnitudes, zernikes_to_magnitude_angle

from prysm.util import rms

#todo: aperture masks and rework the silly pitch_x/pitch_y of lens array into something more sensible

class ShackHartmannAnalyser:
    def __init__(self, microlens_positions, microlens_cell_corners, wavefront_radius, wvl, px_x_size, px_y_size, sensor_x_px_count, sensor_y_px_count):
        self.microlens_positions = microlens_positions
        self.microlens_cell_corners = microlens_cell_corners
        self.sensor_x_pitch = px_x_size
        self.sensor_y_pitch = px_y_size
        self.sensor_x_px_count = sensor_x_px_count
        self.sensor_y_px_count = sensor_y_px_count
        self.wf_r = wavefront_radius
        self.wvl = wvl

        
    #assuming 0,0 is top left....
    #never converting back from pixels to world space :)
    def optical_coords_to_pixel(self, x,y):
        xpx = x / self.sensor_x_pitch
        ypx = y / self.sensor_y_pitch
        centre_offset_x = self.sensor_x_px_count / 2
        centre_offset_y = self.sensor_y_px_count / 2
        return (xpx+centre_offset_x, centre_offset_y-ypx)

    def cartesian_grad_to_polar_grad(self, x, y, dx, dy):
        r, t = coordinates.cart_to_polar(x, y, vec_to_grid=False)
        grad_end_pts_x = x + dx 
        grad_end_pts_y = y + dy 
        grad_end_pts_r, grad_end_pts_t = coordinates.cart_to_polar(grad_end_pts_x, grad_end_pts_y, vec_to_grid=False)

        dr = r - grad_end_pts_r
        dt = t - grad_end_pts_t
        
        dt = (dt + np.pi) % (2 * np.pi) - np.pi
        return dr, dt

    def polar_gradient_to_cartesian_grad(self, r, t, dr, dt):
        x, y = coordinates.polar_to_cart(r, t)
        grad_end_pts_r = r + dr
        grad_end_pts_t = t + dt 
        grad_end_pts_x, grad_end_pts_y = coordinates.polar_to_cart(grad_end_pts_r, grad_end_pts_t)

        dx = x - grad_end_pts_x
        dy = y - grad_end_pts_y
        return dx, dy

    def is_point_on_sensor(self, x,y):
        sensor_width = self.sensor_x_pitch * self.sensor_x_px_count
        sensor_height = self.sensor_y_pitch * self.sensor_y_px_count
        xmin = (-1*(sensor_width/2))
        xmax = (+1*(sensor_width/2))
        ymin = (-1*(sensor_height/2))
        ymax = (+1*(sensor_height/2))

        if (x > xmin and x < xmax and y > ymin and y < ymax):
            return True
        else:
            return False


    def analyse(self, img):
        
        ulens_aabb = []
        for ulens in self.microlens_positions:
            lens_corners = np.empty(shape=(len(self.microlens_cell_corners),2))
            for cnr_idx in range(len(self.microlens_cell_corners)):
                cnr = self.microlens_cell_corners[cnr_idx]
                cnr_x = cnr[0] + ulens[0]
                cnr_y = cnr[1] + ulens[1]
                lens_corners[cnr_idx][0] = cnr_x
                lens_corners[cnr_idx][1] = cnr_y
            maxs = np.max(lens_corners, axis=0)     
            mins = np.min(lens_corners, axis=0)

            if (self.is_point_on_sensor(mins[0], mins[1]) and self.is_point_on_sensor(maxs[0], maxs[1])):
                ulens_aabb.append([mins[0], mins[1], maxs[0], maxs[1], ulens[0], ulens[1]]) #min_x, min_y, max_x, max_y, centre x, centre y

        cell_dbg_d = img
        cell_infos = []
        for ulens in ulens_aabb:
            min_cnr_pixels = np.array(self.optical_coords_to_pixel(ulens[0],ulens[3])).astype(int)
            max_cnr_pixels = np.array(self.optical_coords_to_pixel(ulens[2],ulens[1])).astype(int)
            cell = cell_dbg_d[min_cnr_pixels[1]:max_cnr_pixels[1], min_cnr_pixels[0]:max_cnr_pixels[0]]

            
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
            centroid = (centroid_px-0.5) * self.sensor_x_pitch

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
            if cr <= 4.7 or cr < self.wf_r: #c['intensities'] > cell_int_thresh:
                cells_for_calc.append(c)


        final_img = img
        expected_pos_img_px = np.zeros((len(cells_for_calc),2))
        deviation_img_px    = np.zeros((len(cells_for_calc),2))
        for cell_idx in range(len(cells_for_calc)):
            c = cells_for_calc[cell_idx]

            expected_pos_image = c["lens_centre_xy"] 
            centroid_img = c["centroid_xy"] + c["lens_centre_xy"]
            expected_pos_img_px[cell_idx] = np.array(self.optical_coords_to_pixel(expected_pos_image[0],expected_pos_image[1]))
            deviation_img_px[cell_idx]    = np.array(self.optical_coords_to_pixel(centroid_img[0],centroid_img[1]))
        
            #plt.quiver(expected_pos_img_px[0], expected_pos_img_px[1], deviation_img_px[0], deviation_img_px[1], alpha=0.5, color='white')


        #plt.imshow(final_img)
        #plt.plot(expected_pos_img_px[:,0], expected_pos_img_px[:,1], marker ='o', color='blue', alpha=0.5,linestyle='None', zorder=4)
        #plt.plot(deviation_img_px[:,0], deviation_img_px[:,1], marker ='+', color='red', alpha=0.5, linestyle='None',zorder=6)
        #plt.show()

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

        dr, dt = self.cartesian_grad_to_polar_grad(x/self.wf_r, y/self.wf_r, slopes_xy[0], slopes_xy[1])

        r = r / self.wf_r #go from mm to unit circle to allow for fitting

        fringe_indices = range(1,10)
        nms = [fringe_to_nm(j) for j in fringe_indices]


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
        print("Fits of the following zernikes:")
        for z in range(0, len(nms)):
            print(nms[z])
            print(nm_to_name(nms[z][0], nms[z][1]) + ": " + str(fit[z]))

       
        pak = [[*nm, c] for nm, c in zip(nms, fit)]
        magnitudes = zernikes_to_magnitude_angle(pak)
        barplot_pak = {k: v[0] for k, v in magnitudes.items()}
        barplot_magnitudes(barplot_pak)
        plt.show()


        fig, ax = plt.subplots(figsize=(7,7), subplot_kw={'projection': 'polar'})
        for z in magnitudes:
            ax.plot(magnitudes[z][1], magnitudes[z][0], label=z, marker='+')

        plt.legend()
        ax.set_rlabel_position(-22.5)
        ax.grid(True)

        ax.set_title("Angle/mag of Zernikes", va='bottom')
        plt.show()

        #reconstruct full wavefront
        recon_x, recon_y = coordinates.make_xy_grid(2048, diameter=2)
        recon_r, recon_t = coordinates.cart_to_polar(recon_x, recon_y)
        

        reconstructed_phase = np.zeros(recon_x.shape)
        for f, nm in zip(fit, nms):
            opd_for_z = f*zernike_nm(nm[0],nm[1],recon_r,recon_t, norm=False)
            
            reconstructed_phase += opd_for_z


        rp_img = reconstructed_phase.copy()
        #go back from normalised unit circle zernike land to physical wavefront beam size
        recon_r *= self.wf_r
        zernike_mask = geometry.circle(self.wf_r, recon_r)
        rp_img[zernike_mask!=1]=np.nan

        plt.figure(figsize=(7,7))
        plt.imshow(rp_img)
        plt.show()
