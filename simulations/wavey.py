from prysm import detector
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
class camera(detector.Detector):

    def __init__(self, dark_current, read_noise, bias, fwc, conversion_gain, bits, exposure_time, pixel_width, pixel_height, sensor_width_npx, sensor_height_npx):
        super().__init__(dark_current, read_noise, bias, fwc, conversion_gain, bits, exposure_time)

        self.sensor_width_npx = sensor_width_npx
        self.sensor_height_npx = sensor_height_npx
        self.sensor_width = pixel_width * sensor_width_npx
        self.sensor_height = pixel_height * sensor_height_npx

    def expose2(self, sim_image, sim_x, sim_y, peak_eminus):
    
        #warning warning warning to future me: simulation plane needs to be larger than sensor plane

        #now we need to take the area the the sensor will observe out of the simulation coordinate space and convert to pixels
        sensor_area_x = (self.sensor_width/2  >= sim_x) & (-1*self.sensor_width/2  <= sim_x)
        sensor_area_y = (self.sensor_height/2 >= sim_y) & (-1*self.sensor_height/2  <= sim_y)
        sensor_area = sensor_area_x & sensor_area_y
        i_sens = sim_image.data.copy()
        i_sens[sensor_area!=True]=np.nan 
        plt.imshow(i_sens)
        plt.title("Area of simulation plane captured by image sensor")

        #now actually grab the relevant sensor indicies out of the full simulation xy space
        sensor_field_indicies = np.where(sensor_area == True)
        sensor_field = sim_image.data[sensor_field_indicies[0][0]:sensor_field_indicies[0][-1],sensor_field_indicies[1][0]:sensor_field_indicies[1][-1]]

        #resample to the simulation plane to pixels
        img2 = cv.resize(sensor_field, dsize=(self.sensor_width_npx, self.sensor_height_npx))

        sh_capture = self.expose(img2*peak_eminus)

        plt.figure(figsize=(12,12))
        plt.imshow(sh_capture, cmap='gray')
        plt.title("Image captured by sensor")
        return sh_capture


class imx533(camera):
    def __init__(self):
        super().__init__(dark_current=0.06250, 
        read_noise=1.5, 
        bias=800, 
        fwc=50_000, 
        conversion_gain=1, 
        bits=14, 
        exposure_time=14, 
        pixel_width=3.76/1000.0, 
        pixel_height=3.76/1000.0,
        sensor_width_npx=3008, 
        sensor_height_npx=3008)

class imx174(camera):
    def __init__(self):
        super().__init__(dark_current=0.1, #dunno lol?
        read_noise=3.5, 
        bias=800, 
        fwc=32000, 
        conversion_gain=1, 
        bits=12,
        exposure_time=14, 
        pixel_width=5.86/1000.0, 
        pixel_height=5.86/1000.0, 
        sensor_width_npx=1936, 
        sensor_height_npx=1216)

class microlens_array:
    def __init__(self, lens_centres_xy, lens_cell_corners):
        self.lens_centres_xy = lens_centres_xy
        self.lens_cell_corners = lens_cell_corners

    def get_lens_centres(self):
        return self.lens_centres_xy
        
    def get_lens_cell_corners(self):
        return self.lens_cell_corners

class thorlabs_mla300(microlens_array):
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
        super().__init__(lens_centres_xy, lens_cell_corners)

class thorlabs_mla1(microlens_array):
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
        super().__init__(lens_centres_xy, lens_cell_corners)