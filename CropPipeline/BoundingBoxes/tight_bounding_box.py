class BoundingBox:
    """
    This class is the implementation of a square bounding box for 2D points.
    """
    def __init__(self, margin=0.0):
        """
        :param margin: The margin that will be added to the bounding box. Since it is squared the same amount
        is added in all directions. That amount is half of the margin
        """
        assert type(margin) == float
        assert 0 <= margin

        self.margin = margin

    def get_bounding_box(self, points_2d):
        """
        The function computes the bounding box given the 2D coordinates of the points.
        It expects a list of tuples as an input.
        For the bounding box computation, it calls the specific implementation the user specified.
        The bounding box is expected to be in the format [x, y, width, height]

        :param points_2d: A list of tuples representing the 2D coordinates of the joints.
        :return: The bounding box in format [x, y, width, height]
        """
        # Check that the input is a list of tuples
        assert type(points_2d) == list
        for elem in points_2d:
            assert len(elem) == 2

        bbox = self._compute_bounding_box(points_2d)
        assert len(bbox) == 4

        return bbox

    def _compute_bounding_box(self, points_2d):
        """
        Computes the bounding box given the 2D coordinates of the joints. Since this method is called
        from the abstract class we know that the coordinates are given as a list of tuples.

        :param points_2d: A list of tuples that contain the 2D coordinates of the joints
        :return: The computed bounding box in format [x, y, width, height]
        """
        max_x = max(map(lambda point: int(point[0]), points_2d))
        min_x = min(map(lambda point: int(point[0]), points_2d))
        max_y = max(map(lambda point: int(point[1]), points_2d))
        min_y = min(map(lambda point: int(point[1]), points_2d))

        width = max_x - min_x + 1
        height = max_y - min_y + 1

        return [min_x, min_y, width, height]
