import numpy as np

# Direction w.r.t the north axis --> alpha
# Direction w.r.t the east axis --> theta

def Polar2CartesianComponents(v):
    result = []
    result.append(v[0]*np.cos(np.deg2rad(v[1])))
    result.append(v[0] * np.sin(np.deg2rad(v[1])))
    return result

def Cartesian2PolarComponents(v):
    magnitude = np.linalg.norm(v)
    dirn = np.rad2deg(np.arctan2(v[1], v[0]))
    dirn = (dirn + 360) % 360
    return [magnitude, dirn]

def Cartesian2PolarComponentsAlpha(v):
    return Cartesian2PolarComponents(v[::-1])

def Main():

    amp = 1
    thetas = [30, 90, 120, 240, 330, 360]
    exp_alpha = [60, 0, 330, 210, 120, 90]
    est_theta = []
    est_alpha = []

    for theta in thetas:
        v_cart = Polar2CartesianComponents([amp, theta])
        est_theta.append(Cartesian2PolarComponents(v_cart)[1])
        est_alpha.append(Cartesian2PolarComponentsAlpha(v_cart)[1])

    print('exp_theta: ', thetas)
    print('est_theta: ', est_theta)
    print('exp_alpha: ', exp_alpha)
    print('est_alpha: ', est_alpha)

Main()