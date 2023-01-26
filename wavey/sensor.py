from prysm import detector
import cv2 as cv
import numpy as np
class Sensor(detector.Detector):
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

        #now actually grab the relevant sensor indicies out of the full simulation xy space
        sensor_field_indicies = np.where(sensor_area == True)
        sensor_field = sim_image.data[sensor_field_indicies[0][0]:sensor_field_indicies[0][-1],sensor_field_indicies[1][0]:sensor_field_indicies[1][-1]]

        #resample to the simulation plane to pixels
        img2 = cv.resize(sensor_field, dsize=(self.sensor_width_npx, self.sensor_height_npx))

        sh_capture = self.expose(img2*peak_eminus)
        return sh_capture


class imx533(Sensor):
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

class imx174(Sensor):
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