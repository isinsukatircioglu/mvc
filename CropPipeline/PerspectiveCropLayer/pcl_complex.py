import torch
import IPython

class PerspectiveCropLayer():
    """
    Perspective Crop Layer implementation
    """
    def __init__(self, cop_pixel_size, device):
        self.device = device
        self.cop_pixel_size = cop_pixel_size
        
        # create a grid of linearly increasing indices (one for each pixel, going from 0..1)
        xs = torch.linspace(0, 1, self.cop_pixel_size[0]).to(device)
        ys = torch.linspace(0, 1, self.cop_pixel_size[1]).to(device)
        rs, cs = torch.meshgrid([xs,ys]) # going to rows and columns
        zs = torch.ones(rs.shape).to(device) # init homogeneous coordinate to 1
        pv = torch.stack([rs, cs, zs])

        # same input grid for all batch elements, expand along batch dimension
        self.grid = pv.unsqueeze(0)
        
    def bAffine2boxXXXXX(self, affine_torch):
        bbox_pos_img = affine_torch[:,:,2]
        bbox_size_img = torch.stack([affine_torch[:, 0, 0], affine_torch[:, 1, 1]], dim=-1)
        return bbox_pos_img, bbox_size_img

    def affineToNormalizedXXXXXX(self, A, image_size):
        batch_size = A.shape[0]
        scaling = 1/torch.diag(torch.cat([torch.FloatTensor(image_size).to(A.device), torch.ones([1]).to(A.device)]))
        scaling = scaling.unsqueeze(0).expand([batch_size, 3, 3])
        B = scaling @ A
        B[:, :, 2] = B[:, :, 2] * 2 - 1  # origin in center and from range -1 to 1
        return B

    def perspective_grid(self, P_virt2orig):
        batch_size = P_virt2orig.shape[0]
        bpv = self.grid.expand([batch_size, 3, self.cop_pixel_size[0], self.cop_pixel_size[1]]) # notex row,column flipped

        # linearize the 2D grid to a single dimension, to apply transformation
        bpv_lin = self.grid.view([batch_size,3, -1])

        # do the projection
        bpv_lin_orig = torch.bmm(P_virt2orig, bpv_lin)
        eps = 0.00000001
        bpv_lin_orig_p = bpv_lin_orig[:,:2,:] / (eps + bpv_lin_orig[:,2:3,:]) # projection, divide by homogeneous coordinate

        # go back from linear to two–dimensional outline of points
        bpv_orig = bpv_lin_orig_p.view(batch_size, 2, self.cop_pixel_size[0], self.cop_pixel_size[1])

        # the sampling function assumes the position information on the last dimension
        bpv_orig = bpv_orig.permute([0,3,2,1])

        #if ret_torch_convention:
        #    bpv_orig = 2 * bpv_orig - 1  # pytorch image coordinates go from -1..1 instead of 0..1
        #    bpv_orig[:, :, :, 1] *= -1  # y coordinate goes top down

        return bpv_orig
        
    def forward(self, bbox_pos_img, bbox_size_img, K):
        K_inv = torch.stack([m.inverse() for m in torch.unbind(K)])

        # get target position from image coordinates (normalized pixels)
        p_position = self.bmm_homo(K_inv, bbox_pos_img)

        # get rotation from orig to new coordinate frame
        R_virt2orig = self.virtualCameraRotationFromPosition(p_position)
        #R_orig2virt = R_virt2orig.transpose(1,2)

        # determine target frame
        K_virt = self.bK_virt(p_position, bbox_size_img, K)
        K_virt_inv = torch.stack([m.inverse() for m in torch.unbind(K_virt)])

        # projective transformation orig to virtual camera
        P_virt2orig = torch.bmm(K, torch.bmm(R_virt2orig, K_virt_inv))

        return P_virt2orig, R_virt2orig, K_virt

    def bK_virt(self, p_position, bbox_size_img, K, focal_at_image_plane=True, slant_compensation=False):
        batch_size = bbox_size_img.shape[0]
        p_length = torch.norm(p_position, dim=1, keepdim=True)
        focal_length_factor = 1
        if focal_at_image_plane:
            focal_length_factor *= p_length
        if slant_compensation:
            sx = torch.sqrt(p_position[:, 0] ** 2 + p_position[:, 2] ** 2) / p_position[:, 2]
            sy = torch.sqrt(p_position[:, 1] ** 2 + p_position[:, 2] ** 2) / p_position[:, 2]
            focal_length_factor = focal_length_factor * torch.stack([sx, sy], dim=1)
        f_orig = torch.stack([K[:,0,0], K[:,1,1]], dim=1)
        f_compensated = focal_length_factor * f_orig / bbox_size_img
        K_virt        = torch.zeros([batch_size,3,3], dtype=torch.float).to(f_compensated.device)
        K_virt[:,2,2] = 1

        if 1:  # pytorch convention for grid sample (image corner in the top left, unit coordinates)
            K_virt[:, 0, 0] = f_compensated[:, 0]
            K_virt[:, 1, 1] = -torch.abs(f_compensated[:, 1]) # points downwards in pytorch, abs to make it compatible with K in pytorch coords (when f_y is already negative)
            K_virt[:,:2, 2] = 0.5
        else: # pixel convention (image corner in bottom left, self.cop_pixel_size in px, K in px
            K_virt[:, 0, 0] = f_compensated[:, 0]
            K_virt[:, 1, 1] = f_compensated[:, 1]
            K_virt[:,:2, 2] = 0.5*self.cop_pixel_size

        return K_virt

    def virtualCameraRotationFromPosition(self, position):
        x, y, z = position[:, 0], position[:, 1], position[:, 2]
        n1x = torch.sqrt(1 + x ** 2)
        d1x = 1 / n1x
        d1y = 1 / torch.sqrt(1 + y ** 2)
        d1xy = 1 / torch.sqrt(1 + x ** 2 + y ** 2)
        d1xy1x = 1 / torch.sqrt((1 + x ** 2 + y ** 2) * (1 + x ** 2))
        nd = torch.sqrt(((1 + x ** 2)) / (1 + x ** 2 + y ** 2))
        # rotation transformation from orig to virtual camera
        R_virt2orig = torch.stack([
            d1x, -x * y * d1xy1x, x * d1xy,
                 0 * x, n1x * d1xy, y * d1xy,
                 -x * d1x, -y * d1xy1x, 1 * d1xy
        ], dim=1).reshape([-1, 3, 3])

        # Check for orthogonality
        if 1:
            RX = R_virt2orig[0] # HACK
            import numpy as np
            rows = np.array([sum(RX[0]*RX[1]),sum(RX[0]*RX[2]),sum(RX[1]*RX[2])])
            cols = np.array([sum(RX[0]*RX[1]),sum(RX[0]*RX[2]),sum(RX[1]*RX[2])])
            if np.any(rows>0.00001) or np.any(cols>0.00001):
                print("rows",rows)
                print("cols",cols)
        return R_virt2orig


    def point_px2unit(self, points, img_w_h):
        positions_unit = points / img_w_h
        positions_unit[:,1] = 1-positions_unit[:,1]
        return positions_unit

    def scale_px2unit(self, points, img_w_h):
        positions_unit = points / img_w_h
        return positions_unit

    def K_px2K_unit(self, K_px, img_w_h):
        K_unit = K_px.clone()
        K_unit[:,0,0] = K_px[:,0,0] / img_w_h[0] # scale from 0..w to 0..1
        K_unit[:,1,1] = K_px[:,1,1] / img_w_h[1] # scale from 0..h to 0..1
        K_unit[:,0,2] = K_px[:,0,2] / img_w_h[0] # scale offset from 0..w to 0..1
        K_unit[:,1,2] = K_px[:,1,2] / img_w_h[1] - 1 # scale offset from 0..h to 0..1

        K_unit[:,1,1] *= -1 # point y coordinates downwards (image coordinates start in top-left corner in pytorch)
        K_unit[:,1,2] *= -1 # point y coordinates downwards (image coordinates start in top-left corner in pytorch)

        return K_unit

    def point_px2torch(self, points, img_w_h):
        points_torch = 2 * points / img_w_h - 1
        points_torch[:, 1] *= -1
        return points_torch

    def scale_px2torch(self, points, img_w_h):
        points_torch = points / img_w_h
        return points_torch

    def K_px2K_torch(self, K_px, img_w_h):
        K_torch = K_px.clone()
        K_torch[:,0,0] = K_px[:,0,0]*2 / img_w_h[0] # spread out from 0..w to -1..1
        K_torch[:,1,1] = K_px[:,1,1]*2 / img_w_h[1] # spread out from 0..h to -1..1
        K_torch[:,0,2] = K_px[:,0,2]*2 / img_w_h[0] - 1 # move image origin bottom left corner to to 1/2 image width
        K_torch[:,1,2] = K_px[:,1,2]*2 / img_w_h[1] - 1 # move image origin to 1/2 image width

        K_torch[:,1,1] *= -1 # point y coordinates downwards (image coordinates start in top-left corner in pytorch)
        K_torch[:,1,2] *= -1 # point y coordinates downwards (image coordinates start in top-left corner in pytorch)

        return K_torch

    def bmm_homo(self, K_inv, bbox_center_img):
        batch_size = bbox_center_img.shape[0]
        ones = torch.ones([batch_size, 1], dtype=torch.float).to(bbox_center_img.device)
        bbox_center_px_homo = torch.cat([bbox_center_img, ones],dim=1).reshape([batch_size,3,1])
        cam_pos = torch.bmm(K_inv, bbox_center_px_homo).view(batch_size,-1)
        return cam_pos

    def fromNormalizedPositionXXXXX(self, position_m1to1):
        return (position_m1to1+1)/2

    def fromTorchAffineXXXXX(self, X_torch):
        X = torch.new_zeros(X_torch.shape, dtype=torch.float)
        # scales and translation
        s = torch.stack([X_torch[:, 0, 0], X_torch[:, 1,1]], dim=-1)
        t_torch = X_torch[:,:,2]
        X[:, :, 2] = (t_torch+1)/2 - s/2
        X[:, 0, 0] = s[:,0]
        X[:, 1, 1] = s[:,1]
        return X

    def toTorchAffineXXXXX(self, X):
        X_torch = torch.new_zeros(X.shape, dtype=torch.float)
        # scales and translation
        s = torch.stack([X[:, 0, 0], X[:, 1,1]], dim=-1)
        t = X[:,:,2]
        X_torch[:, :, 2] = (t+s/2)*2-1
        X_torch[:, 0, 0] = s[:,0]
        X_torch[:, 1, 1] = s[:,1]
        return X_torch