from math import acos
from math import sqrt
from math import pi
import numpy as np

def length(v):
    return sqrt(v[0]**2+v[1]**2)

def dot_product(v,w):
    return v[0]*w[0]+v[1]*w[1]

def determinant(v,w):
    return v[0]*w[1]-v[1]*w[0]

def inner_angle(v,w):
    cosx=dot_product(v,w)/(length(v)*length(w))
    rad=acos(cosx) # in radians
    return rad*180/pi # returns degrees

def angle_clockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner
    
def within_angles(xs, ys, angle1 = 148, angle2 = 215):
    '''
    xs and ys must be corresponding coordinates in matplotlib-space (y-axis 0 starts from the top).
    '''
    
    # Matplotlib y-axis -- 0 starts from the top, so we need to adjust. 
    v0 = np.array([xs[0] - 400, 800 - ys[0] - 400])
    v1 = np.array([xs[1] - 400, 800 - ys[1] - 400])
    v2 = np.array([xs[2] - 400, 800 - ys[2] - 400])
    v3 = np.array([xs[3] - 400, 800 - ys[3] - 400])
    v_ref = np.array([2, 0])
    
    condition = (angle_clockwise(v_ref, v0) >= angle1 and angle_clockwise(v_ref, v0) <= angle2) or \
                (angle_clockwise(v_ref, v1) >= angle1 and angle_clockwise(v_ref, v1) <= angle2) or \
                (angle_clockwise(v_ref, v2) >= angle1 and angle_clockwise(v_ref, v2) <= angle2) or \
                (angle_clockwise(v_ref, v3) >= angle1 and angle_clockwise(v_ref, v3) <= angle2)
    return condition