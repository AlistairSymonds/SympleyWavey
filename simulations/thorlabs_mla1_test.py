from raypier.api import RayTraceModel, GeneralLens, CollimatedGaussletSource, SphericalFace, CircleShape, PlanarFace, RectangleShape, OpticalMaterial, EFieldPlane, GaussletCapturePlane, IntensitySurface
from raypier.intensity_image import IntensityImageView

import numpy as np
#all units in mm

ulens_pitch = [1, 1.4]
shape = RectangleShape(height=ulens_pitch[1], width=ulens_pitch[0])

num_ulens = [10, 7]
print( ((ulens_pitch[0]*(num_ulens[0]-1))/2))
ulens_centres_x = np.linspace(-((ulens_pitch[0]*(num_ulens[0]-1))/2), ((ulens_pitch[0]*(num_ulens[0]-1))/2), num_ulens[0])
ulens_centres_y = np.linspace(-((ulens_pitch[1]*(num_ulens[1]-1))/2), ((ulens_pitch[1]*(num_ulens[1]-1))/2), num_ulens[1])

f1 = SphericalFace(curvature=-2.7, z_height=0.0)
f2 = PlanarFace(z_height=1)
#
pmma_material = OpticalMaterial(refractive_index=1.4906)

unlens_objs = []
print(ulens_centres_x)
print(ulens_centres_y)
for xx in ulens_centres_x:
    for yy in ulens_centres_y:
        ulens = GeneralLens(centre=(xx,yy,0),
                            direction=(0,0,1),
                            shape=shape,
                            surfaces=[f1,f2],
                            materials=[pmma_material])
        
        unlens_objs.append(ulens)

test_lens = False
if test_lens:
    shape = CircleShape(radius=14)
    f1 = SphericalFace(curvature=-50.0, z_height=0.0)
    f2 = SphericalFace(curvature=50.0, z_height=5.0)
    m = OpticalMaterial(glass_name="N-BK7")
    lens1 = GeneralLens(centre=(0,0,-20),
                        direction=(0,0,1),
                        shape=shape,
                        surfaces=[f1,f2],
                        materials=[m])
    unlens_objs.append(lens1)

src=CollimatedGaussletSource(origin=(0,0,-100),
                             direction=(0,0,1),
                             radius=5.0,
                             beam_waist=10.0,
                             resolution=10.0,
                             E_vector=(0,1,0),
                             wavelength=1.0,
                             display="wires",
                             opacity=0.5,
                             max_ray_len=150.0)


cap = GaussletCapturePlane(centre=(0,0,5.7),
                           direction=(0,0,1))

field = EFieldPlane(centre=(0,0,5.7),
                    direction=(0,0,1),
                    detector=cap,
                    align_detector=True,
                    width=10.0,
                    height=10.0,
                    size=500)



image = IntensityImageView(field_probe=field)
surf = IntensitySurface(field_probe=field)

model = RayTraceModel(optics=unlens_objs,
                        sources=[src],
                         probes=[field, cap], results=[image,surf])

###Now open the GUI###
model.configure_traits()