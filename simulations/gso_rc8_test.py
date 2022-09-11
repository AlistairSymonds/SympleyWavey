from raypier.api import RayTraceModel, GeneralLens, CollimatedGaussletSource, CircularAperture, ConicFace, CircleShape, PlanarFace, RectangleShape, OpticalMaterial, EFieldPlane, GaussletCapturePlane, IntensitySurface
from raypier.intensity_image import IntensityImageView

import numpy as np
#all units in mm
optics_objs = []
srcs = []

F = 1600
D = 354.6
B = D + 300

M = (F - B)/D

R_1 = (-2*F)/M
R_2 = (-2*B)/(M-1)

K_1 = -1 - ((2/(M**3)) * (B/D))

K_2 = -1 - ((2/((M-1)**3)) * ((M*(2*M-1)) + (B/D)))

print(f"Optical paramters: \n F: {F:.4f}\n D: {D:.4f}\n B: {B:.4f}\n M: {M:.4f}\n R_1: {R_1:.4f}\n K_1: {K_1:.4f}\n R_2: {R_2:.4f}\n K_2: {K_2:.4f}")
mat = OpticalMaterial(from_database=False, refractive_index=1.5)

primary_mirror_shape = CircleShape(radius=200.0/2) ^ CircleShape(radius=66.5/2)

pm_optic_face = ConicFace(conic_const=K_1, curvature=R_1, mirror=True, z_height=0)


primary_mirror = GeneralLens(centre=(0,0,354.6),
                   direction=(0,0,-1),
                   shape=primary_mirror_shape,
                   surfaces=[pm_optic_face])

optics_objs.append(primary_mirror)

secondary_mirror_shape = CircleShape(radius=94.6/2)

sm_optic_face = ConicFace(conic_const=K_2, curvature=R_2, mirror=True, z_height=0)


secondary_mirror = GeneralLens(centre=(0,0,0),
                   direction=(0,0,-1),
                   shape=secondary_mirror_shape,
                   surfaces=[sm_optic_face])

optics_objs.append(secondary_mirror)

sm_back = CircularAperture(centre=(0,0,-12),outer_diameter=94.6, inner_diameter=0)
optics_objs.append(sm_back)

src=CollimatedGaussletSource(origin=(0,0,-100), 
                             direction=(0,0,1),
                             radius=100,
                             blending=1,
                             beam_waist=10.0,
                             resolution=3,
                             E_vector=(0,1,0),
                             wavelength=1.0,
                             display="wires",
                             opacity=0.5,
                             max_ray_len=2000.0)
srcs.append(src)

if False:
    src2=CollimatedGaussletSource(origin=(0,10,-100), 
                                direction=(0,0.005,1),
                                color=(0,0,1),
                                radius=100,
                                blending=1,
                                beam_waist=10.0,
                                resolution=3,
                                E_vector=(0,1,0),
                                wavelength=1.0,
                                display="wires",
                                opacity=0.5,
                                max_ray_len=2000.0)
    srcs.append(src2)


cap = GaussletCapturePlane(centre=(0,0,B),
                           direction=(0,0,1))

field = EFieldPlane(centre=(0,0,B),
                    direction=(0,0,1),
                    detector=cap,
                    align_detector=True,
                    width=36,
                    height=24,
                    size=500)



image = IntensityImageView(field_probe=field)
surf = IntensitySurface(field_probe=field)

model = RayTraceModel(optics=optics_objs,
                        sources=srcs,
                         probes=[field, cap], results=[image,surf])

###Now open the GUI###
model.configure_traits()