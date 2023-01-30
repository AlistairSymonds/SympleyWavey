import numpy as np
"""
Taken from: Determination of phase mode components in terms of
local wave-front slopes: an analytical approach. (Acosta, 1995)

There is a patten here. I'm ignoring it for now for in the interest of just having a working v1
"""
def zernike_nm_der_acosta(n, m, r, t, norm=True):
    dr = np.zeros_like(r)
    dt = np.zeros_like(t)

    if n == 1 and m == -1:
        dt = (r ** 2) * np.sin(t)
    elif n == 1 and m == 1:
        dt = (r ** 2) * (-1*np.cos(t))

        
    elif n == 2 and m == 0:
        dr = (0.5) * ((r ** 3) - r)
    elif n == 2 and m == -2:
        dt = ((r ** 3)/2) * np.sin(2*t)
    elif n == 2 and m == 2:
        dt = ((r ** 3)/2) * (-1*np.cos(2*t))


    elif n == 3 and m == -1:
        dr = ((3 * (r ** 4)) - (2 * (r ** 2)))  * np.sin(t)
        
    elif n == 3 and m == 1:
        dr = ((3 * (r ** 4)) - (2 * (r ** 2)))  * (-1*np.cos(t))
        
    elif n == 3 and m == -3:
        dr = ((r ** 4)/3) * np.sin(3*t)
        
    elif n == 3 and m == 3:
        dr = ((r ** 4)/3) * (-1*np.cos(3*t))
        

    
    elif n == 4 and m == 0:
        
        dt = (r ** 5) - ((3/2) * (r ** 3)) + (r/2)
    elif n == 4 and m == -2:
        dr = (1/2) * ((4 * (r ** 5)) - (3 * (r ** 3)))  * (np.sin(2*t))
        
    elif n == 4 and m == 2:
        dr = (1/2) * ((4 * (r ** 5)) - (3 * (r ** 3)))  * (-1*np.cos(2*t))
        
    elif n == 4 and m == -4:
        dr = ((r ** 5)/4) * (np.sin(4*t))
        
    elif n == 4 and m == 4:
        dr = ((r ** 5)/4) * (-1*np.cos(4*t))
        
    
    return dr, dt

def zernike_nm_der_acosta_sequence(nms, r, t, norm=True):
    # blatantly ripping of prysm here for consistency
    out = []
    for n, m in nms:
        out.append(zernike_nm_der_acosta(n, m, r, t, norm=norm))

    return out
