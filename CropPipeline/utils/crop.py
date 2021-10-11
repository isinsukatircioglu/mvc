import numpy as np


def crop(image, bounding_box):
    orig_image_width = image.shape[1]
    orig_image_height = image.shape[0]

    crop_shape = (bounding_box[3], bounding_box[2], image.shape[2])
    crop_frame = np.zeros(tuple(crop_shape), dtype=image.dtype)
    if not (bounding_box[0] > image.shape[1] or bounding_box[0] + bounding_box[2] < 0 or
                    bounding_box[1] > image.shape[0] or bounding_box[1] + bounding_box[3] < 0):
        img_min_x = max(0, bounding_box[0])
        img_max_x = min(bounding_box[0] + bounding_box[2], orig_image_width)
        img_min_y = max(0, bounding_box[1])
        img_max_y = min(bounding_box[1] + bounding_box[3], orig_image_height)

        crop_min_x = img_min_x-bounding_box[0]
        crop_max_x = img_max_x-bounding_box[0]
        crop_min_y = img_min_y-bounding_box[1]
        crop_max_y = img_max_y-bounding_box[1]

    else:
        return None

    crop_frame[crop_min_y:crop_max_y, crop_min_x:crop_max_x, :] = image[img_min_y:img_max_y,
                                                                        img_min_x:img_max_x,
                                                                        :]

    return crop_frame
