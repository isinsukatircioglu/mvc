import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
#from Carbon.QuickDraw import pHiliteBit

from math import cos,sin,atan2,acos
import pdb

def applyHomoTransform(T, points):
    # make homogeneous
#    points_h = np.append(points, np.matlib.repmat(np.array(1),1,points.size()[2]), 0)
    points_h = np.append(points, np.ones([1,points.shape[1]]), 0)
    
    # apply
    points_T_h = np.mat(T) * points_h
    
    # convert back
    points_T_h = points_T_h / points_T_h[-1,:]
    points_T = points_T_h[0:-1, :]
    return points_T

def para_getShearDecomposition(center_3d_w):
    center_2d_w = center_3d_w / center_3d_w[2]

    s1 = -center_2d_w[0]
    s2 = -center_2d_w[1]

    # shear matrix (shear parallel to the x-y plane at depth z0 = center_3d_w[2])
    S_small = np.mat([[1, 0, s1],
                      [0, 1, s2],
                      [0, 0,  1]])
    
    # weak perspective projection matrix at z0
    P = np.mat(np.identity(4)) / center_3d_w[2];
    P [2,:] = np.mat([0,0,0,1])
    P [3,:] = np.mat([0,0,0,1])
                      
    # p_mormalized = P(S_small * (p - p_0)) + P(p_0), where p_0 is the centroid, P is orthographic projection
    #              = P(S_small *  p - S_small * p_0 + p_0)
    
    offset = center_3d_w - S_small * center_3d_w
    #offset = np.array([-s1*center_3d_w[0], -s2*center_3d_w[1], 0])
    
    # now in homogeneous coordinates
    S = np.mat(np.zeros((4,4)))
    S[0:3,0:3] = S_small
    S[0:3, 3 ] = offset
    S[ 3 , 3 ] = 1
    
    return [P, S]    

def para_getIsometricDecomposition(center_3d_w):
    center_2d_w = center_3d_w / center_3d_w[2]
    center_2d_w = np.delete(center_2d_w,2,0)
    
    #print('center_3d_w:\n',center_3d_w)
    #print('center_2d_w:\n',center_2d_w)
       
    s1 = -center_2d_w[0]
    s2 = -center_2d_w[1]

    s2_len    = np.sqrt(s2*s2 + 1)
    s1_s2_len = np.sqrt(s1*s1 + s2*s2 + 1)
    
    # shearing part, with additional third dimension
    C_3d = np.mat([[s1_s2_len/s2_len, s1*s2/ s2_len,0],
                      [0,                s2_len,    0],
                      [0,                     0,    1]])
      
    # Debugging, disapling shear part, i.e. the compensation of rotation                
    #if True:
    #    C_3d = np.mat(np.eye(3))
    
    # weak perspective projection matrix at z0
    P = np.identity(4) / center_3d_w[2];
    P [2,:] = np.mat([0,0,0,1])
    P [3,:] = np.mat([0,0,0,1])
    
    
    # rotatoin component 3d
    R_small =  np.mat([ [s2_len/s1_s2_len, -s1*s2/(s2_len*s1_s2_len), s1/(s2_len*s1_s2_len)],
                        [0,                 1    /s2_len,             s2/s2_len],
                        [-s1   /s1_s2_len, -s2   /s1_s2_len,           1/s1_s2_len]])

    # p_mormalized = C_s* P(R_small * (p - p_0)) + P(p_0), where p_0 is the centroid, P is orthographic projection
    #              = C_s * P(R_small *  p - R_small * p_0 + C_s^-1 * p_0 + p_0 - p_0) # since the third dimension of C and, hence, C^-1 is unchanged, the order of P and C/C^-1 can be interchanged.
    #              = C_s * (P(R_small *  p - R_small * p_0 + p_0) + P(C_s^-1 * p_0 - p_0)
    #              = C_s * P(R_small *  p - R_small * p_0 + p_0) + (I-C_s)P(p_0)
    offset_R =      -R_small * center_3d_w + center_3d_w
    offset_C = applyHomoTransform(P, -C_3d    * center_3d_w + center_3d_w)
#    offset_C =  -C_3d    * center_3d_w + center_3d_w
    offset_C[2] = 0;

    C = np.mat(np.zeros((4,4)))
    C[0:3,0:3] = C_3d
    C[0:3, 3 ] = offset_C
    C[ 3 , 3 ] = 1

    R = np.mat(np.zeros((4,4)))
    R[0:3,0:3] = R_small
    R[0:3, 3 ] = offset_R
    R[ 3 , 3 ] = 1
                  
    # debugging
    #[P_ref, S_ref] = para_getShearDecomposition(points_3d_w)
    #S_dec = C * R
    #T_dec = C * P * R
    #T_ref = P_ref * S_ref
    #print('T_dec')
    #print(T_dec)
    #print('T_ref')
    #print(T_ref)
    
    return [C, P, R]

def getAizmuthAndEvaluation(direction):
    direction = direction / np.linalg.norm(direction)
    r = 1
    x=direction[0]
    y=direction[1]
    z=direction[2]
    theta = acos(y/r)-acos(0)   # polar angle (rotation towards pole)
    phi   = atan2(z,x)-atan2(1,0)  # azimuth  (rotation along equator)
    return theta, phi

def rotationMatrix(theta, phi):
    Ax = np.matrix([[1,0,0],
                    [0, cos(theta), -sin(theta)],
                    [0, sin(theta), cos(theta)]])

    Ay = np.matrix([[cos(phi),0,-sin(phi)], # Warning, flipped sign
                [0, 1, 0],
                [sin(phi), 0, cos(phi)]])

#     Az = np.matrix([[cos(theta),sin(theta),0], # Warning, flipped sign
#                     [-sin(theta), cos(theta), 0],
#                     [0, 0, 1],])
    
    R = Ay * Ax
    return R

def rotationMatrixXZY(theta, phi, psi):
    Ax = np.matrix([[1,0,0],
                    [0, cos(theta), -sin(theta)],
                    [0, sin(theta), cos(theta)]])

    Ay = np.matrix([[cos(phi),0,-sin(phi)],
                [0, 1, 0],
                [sin(phi), 0, cos(phi)]])

    Az = np.matrix([[cos(psi),-sin(psi),0],
                [sin(psi), cos(psi),0],
                [0, 0, 1],])

#     Az = np.matrix([[cos(theta),sin(theta),0],
#                     [-sin(theta), cos(theta), 0],
#                     [0, 0, 1],])
    
    R = Az * Ay * Ax
    return R

    
def persp_getIsometricDecomposition(center_3d_w):#, K_orig, K_virtual):
    theta, phi = getAizmuthAndEvaluation(center_3d_w)
    R_virtualCamToOrigCam = rotationMatrix(theta, phi)
    R_origCamToVirtualCam = inv(R_virtualCamToOrigCam)
    
    #C = K_virtual * R_origCamToVirtualCam * inv(K_orig)
    
    R = R_origCamToVirtualCam

    C = np.mat(np.zeros((4,4)))
    C[:3,:3] = R_origCamToVirtualCam
    #C[ 2 , : ] = 0
    #C[ 2 , 2 ] = 1
    C[ 3 , 3 ] = 1
    
    #pdb.set_trace()
    
    #print("R.det = ",np.linalg.det(R))
    #print("C.det = ",np.linalg.det(R))

    
    return [inv(C), None, R]    

def perspective(points_3d_w, K=np.eye(4)):
    points_2d_w = (points_3d_w / points_3d_w[2,:]);
    points_2d_w_intr = applyHomoTransform(K, points_2d_w)
    #center_3d_w = np.mean(points_3d_w,1)
    #center_2d_w = center_3d_w / center_3d_w[2]
    #points_2d_w_c = points_2d_w - center_2d_w.reshape(-1,1)
    return points_2d_w_intr
    
def weak(points_3d_w, K=np.eye(4)):
    center_3d_w = np.mean(np.mat(points_3d_w),1)
    [P, S_unused] = para_getShearDecomposition(center_3d_w)
    points_2d_w_scaled = applyHomoTransform(K*P, points_3d_w)
    return points_2d_w_scaled
    
def para(points_3d_w, K=np.eye(4)):
    center_3d_w = np.mean(np.mat(points_3d_w),1)
    [P, S] = para_getShearDecomposition(center_3d_w)
    points_2d_w_scaled = applyHomoTransform(K*P*S, points_3d_w)
    return points_2d_w_scaled
    
def para_rot(points_3d_w, K=np.eye(4)):
    center_3d_w = np.mean(np.mat(points_3d_w),1)
    [C, P, R] = para_getIsometricDecomposition(center_3d_w)
    points_2d_w_scaled = applyHomoTransform(K*C*P*R, points_3d_w)
    return points_2d_w_scaled

def getImageAnd3DCorrection(cropCenter_px, K, decomposition=persp_getIsometricDecomposition, K_cropLocal=None, K_cropLocal_orig=None):
#    assert K.shape is not np.array((4,4)).shape, [K, 'K matrix is supposed to be 4x4 (3d homogeneous)']
#    assert cropCenter_px.size is not 3, 'cropCenter_px must contain depth information (the focal length if on the image plane)'
    
    K_inv = inv(K)
    cropCenter_cam = applyHomoTransform(K_inv, cropCenter_px)
    [C, P, R] = decomposition(cropCenter_cam)
    
    R_world = R
    
    S_image = K * inv(C) * K_inv  # image -> cam -> undistorted -> image
    S_image = np.delete(S_image,2,0) # make 3D-homo to 2D-homo
    S_image = np.delete(S_image,2,1)
    
    if K_cropLocal is not None:
        C_noTrans = C
        C_noTrans[:,3] = np.array([[0,0,0,1]]).T
        #print('C_noTrans\n',C_noTrans)
        S_crop = K_cropLocal * inv(C_noTrans) * inv(K_cropLocal_orig)  # image -> cam -> undistorted -> image
        S_crop = np.delete(S_crop,2,0) # make 3D-homo to 2D-homo
        S_crop = np.delete(S_crop,2,1)
    else:
        S_crop=None
        
    
    return [R_world, S_image, S_crop]


        
        