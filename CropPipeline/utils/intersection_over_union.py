def bb_intersection_over_union(boxA, boxB):
    """
    Computes the intersection over union of two bounding boxes.

    :param boxA: A 4-tuple containing the coordinates of the bbox in pixels (min_x, min_y, max_x, max_y)
    :param boxB: A 4-tuple containing the coordinates of the bbox in pixels (min_x, min_y, max_x, max_y)
    :return: The intersection-over-union ratio.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)
    if interArea <= 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    return interArea / float(boxAArea + boxBArea - interArea)
