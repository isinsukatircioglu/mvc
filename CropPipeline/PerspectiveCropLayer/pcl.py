import torch
import IPython

def perspective_grid(P_virt2orig, crop_pixel_size_hw, transform_to_pytorch=False):
    batch_size = P_virt2orig.shape[0]

    # create a grid of linearly increasing indices (one for each pixel, going from 0..1)
    device = P_virt2orig.device
    if not transform_to_pytorch:
        xs = torch.linspace(0, 1, crop_pixel_size_hw[1]).to(device)
        ys = torch.linspace(0, 1, crop_pixel_size_hw[0]).to(device)
    else:
        xs = torch.linspace(0, 1, crop_pixel_size_hw[1]).to(device)
        ys = torch.linspace(1, 0, crop_pixel_size_hw[0]).to(device) # up and down direction is changed

    #rs, cs = torch.meshgrid([xs, ys])  # for pytorch >0.4 instead of following two lines
    cs = ys.view(1, -1).repeat(xs.size(0), 1)
    rs = xs.view(-1, 1).repeat(1, ys.size(0))
    zs = torch.ones(rs.shape).to(device)  # init homogeneous coordinate to 1
    pv = torch.stack([rs, cs, zs])

    # same input grid for all batch elements, expand along batch dimension
    grid = pv.unsqueeze(0).expand([batch_size, 3, crop_pixel_size_hw[1], crop_pixel_size_hw[0]])

    # linearize the 2D grid to a single dimension, to apply transformation
    bpv_lin = grid.view([batch_size, 3, -1])

    # do the projection
    bpv_lin_orig = torch.bmm(P_virt2orig, bpv_lin)
    eps = 0.00000001
    bpv_lin_orig_p = bpv_lin_orig[:, :2, :] / (eps + bpv_lin_orig[:, 2:3, :]) # projection, divide homogeneous coord

    # go back from linear to two–dimensional outline of points
    bpv_orig = bpv_lin_orig_p.view(batch_size, 2, crop_pixel_size_hw[1], crop_pixel_size_hw[0])

    # the sampling function assumes the position information on the last dimension
    bpv_orig = bpv_orig.permute([0, 3, 2, 1])

    if transform_to_pytorch:
        bpv_orig *= 2
        bpv_orig -= 1
        bpv_orig[:,:,:,1] *= -1

    return bpv_orig

def pcl_transforms(bbox_pos_img, bbox_size_img, K, position_and_K_in_pytorch=False, focal_at_image_plane=True, slant_compensation=True):
    if position_and_K_in_pytorch:
        bbox_size_img = 2*bbox_size_img # needs to be multiplied by two, since pytorch scaling is encoded in intrinsics
    K_inv = torch.stack([m.inverse() for m in torch.unbind(K)])

    # get target position from image coordinates (normalized pixels)
    p_position = bmm_homo(K_inv, bbox_pos_img)

    # get rotation from orig to new coordinate frame
    R_virt2orig = virtualCameraRotationFromPosition(p_position)
    #R_orig2virt = R_virt2orig.transpose(1,2)

    # determine target frame
    K_virt = bK_virt(p_position, bbox_size_img, K, focal_at_image_plane, slant_compensation)
    K_virt_inv = torch.stack([m.inverse() for m in torch.unbind(K_virt)])

    # projective transformation orig to virtual camera
    P_virt2orig = torch.bmm(K, torch.bmm(R_virt2orig, K_virt_inv))

    return P_virt2orig, R_virt2orig, K_virt

def bK_virt(p_position, bbox_size_img, K, focal_at_image_plane=True, slant_compensation=True):
    #bbox_size_img = 2*bbox_size_img # Note, *2 necessary to compensate scale of 2*focal length in intrinsics
    bbox_size_img = bbox_size_img # Note, *2 necessary to compensate scale of 2*focal length in intrinsics
    batch_size = bbox_size_img.shape[0]
    p_length = torch.norm(p_position, dim=1, keepdim=True)
    focal_length_factor = 1
    if focal_at_image_plane:
        focal_length_factor *= p_length
    if slant_compensation:
        sx = torch.sqrt(p_position[:,0]**2+p_position[:,2]**2) / p_position[:,2]
        sy = torch.sqrt(p_position[:,1]**2+p_position[:,2]**2) / p_position[:,2]
        focal_length_factor = focal_length_factor * torch.stack([sx,sy], dim=1)

    f_orig = torch.stack([K[:,0,0], K[:,1,1]], dim=1)
    f_compensated = focal_length_factor * f_orig / bbox_size_img
    K_virt        = torch.zeros([batch_size,3,3], dtype=torch.float).to(f_compensated.device)
    K_virt[:,2,2] = 1

    # Note, in unit image coordinates ranging from 0..1
    K_virt[:, 0, 0] = f_compensated[:, 0]
    K_virt[:, 1, 1] = f_compensated[:, 1]
    K_virt[:,:2, 2] = 0.5

    return K_virt

def virtualCameraRotationFromPosition(position):
    x, y, z = position[:, 0], position[:, 1], position[:, 2]
    n1x = torch.sqrt(1 + x ** 2)
    d1x = 1 / n1x
    d1xy = 1 / torch.sqrt(1 + x ** 2 + y ** 2)
    d1xy1x = 1 / torch.sqrt((1 + x ** 2 + y ** 2) * (1 + x ** 2))
    R_virt2orig = torch.stack([d1x, -x * y * d1xy1x, x * d1xy,
                               0*x,      n1x * d1xy, y * d1xy,
                          -x * d1x,     -y * d1xy1x, 1 * d1xy], dim=1).reshape([-1, 3, 3])
    return R_virt2orig

def bmm_homo(K_inv, bbox_center_img):
    batch_size = bbox_center_img.shape[0]
    ones = torch.ones([batch_size, 1], dtype=torch.float).to(bbox_center_img.device)
    bbox_center_px_homo = torch.cat([bbox_center_img, ones],dim=1).reshape([batch_size,3,1])
    cam_pos = torch.bmm(K_inv, bbox_center_px_homo).view(batch_size,-1)
    return cam_pos

