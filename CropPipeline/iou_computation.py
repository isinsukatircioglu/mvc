import os
import sys
import configobj
import csv
import numpy as np
from utils import bb_intersection_over_union
from Cameras import PinholeCamera


class IOU_Computation:
    def __init__(self, config_path):
        self.config_info = configobj.ConfigObj(config_path)
        self._read_config_info()
        self._convert_config_info()
        self._import_modules()
        self._create_output_folder()
        self._create_bounding_box_obj()

    def _read_config_info(self):
        self._read_input_data()
        self._read_output_data()
        self._read_import_data()
        self._read_frame_info()

    def _read_input_data(self):
        self.camera_calibration_file_path = self.config_info['INPUT']['camera_calibration_file_path']
        self.synchronization_file = self.config_info['INPUT']['synchronization_file']
        self.relevent_frames_csv_file = self.config_info['INPUT']['relevent_frames_csv_file']

    def _read_frame_info(self):
        self.frame_width = self.config_info['FRAME_INFO']['frame_width']
        self.frame_height = self.config_info['FRAME_INFO']['frame_height']

    def _read_output_data(self):
        self.output_folder = self.config_info['OUTPUT']['output_folder']

    def _read_import_data(self):
        self.camera_calibration_code = self.config_info['CODE']['camera_calibration']
        self.synchronization_reader_code = self.config_info['CODE']['synchronization_reader']
        self.image_extractor_code = self.config_info['CODE']['image_extractor']
        self.annotation_reader_code = self.config_info['CODE']['annotation_reader']
        self.bounding_box_code = self.config_info['CODE']['bounding_box']

    def _convert_config_info(self):
        self.frame_width = int(self.frame_width)
        self.frame_height = int(self.frame_height)

        if type(self.relevent_frames_csv_file) is not list:
            self.relevent_frames_csv_file = [self.relevent_frames_csv_file]

    def _import_modules(self):
        sys.path.insert(0, os.path.dirname(self.camera_calibration_code))
        self.camera_reader = __import__(os.path.basename(self.camera_calibration_code))

        sys.path.insert(0, os.path.dirname(self.synchronization_reader_code))
        self.synchronization_reader = __import__(os.path.basename(self.synchronization_reader_code))

        sys.path.insert(0, os.path.dirname(self.image_extractor_code))
        self.image_extractor = __import__(os.path.basename(self.image_extractor_code))

        sys.path.insert(0, os.path.dirname(self.annotation_reader_code))
        self.annotation_reader = __import__(os.path.basename(self.annotation_reader_code))

        sys.path.insert(0, os.path.dirname(self.bounding_box_code))
        self.bounding_box = __import__(os.path.basename(self.bounding_box_code))

    def _create_output_folder(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def _create_bounding_box_obj(self):
        self.bbox_obj = self.bounding_box.BoundingBox()

    def _create_camera_frame_dictionary(self, camera_dictionary, annotation_file):
        frame_dict = {}
        for frame_idx, annotation in self.annotation_reader.AnnotationReader(annotation_file,
                                                                             synchronization_offset=camera_dictionary[
                                                                                 'synchronization_frame'],
                                                                             has_annotation=True).read_annotation_data():
            frame_dict[frame_idx] = annotation

        return frame_dict

    @staticmethod
    def _get_projections(PC, annotations):
        camera_projections = [np.array(PC.world_to_camera(point_3d)).reshape(-1, ).tolist() for point_3d in
                              annotations]

        # Transform the annotations in image coordinates
        image_projections = [np.array(PC.world_to_image(point_3d)).reshape(-1, ).tolist() for point_3d in
                             annotations]

        return camera_projections, image_projections

    def _iou_pipeline(self, camera):
        # Create the pinhole camera and save the intrisic and extrinsic parameters in the final dict
        this_camera_dict = self.camera_dict[camera]
        PC = PinholeCamera(this_camera_dict['R'], this_camera_dict['t'], this_camera_dict['focalLength'],
                           this_camera_dict['pixelAspect'], this_camera_dict['sensorSize'],
                           this_camera_dict['centerOffset'],
                           self.frame_width, self.frame_height)

        # Create a dictionary that will contain the frame number and their annotations
        frame_dict_all = {}
        for annotation_idx, annotation_file in enumerate(self.relevent_frames_csv_file):
            frame_dict = self._create_camera_frame_dictionary(this_camera_dict, annotation_file)
            frame_dict_all[str(annotation_idx)] = frame_dict

        # Create a csv file where to store the ratios found
        csv_file = open(os.path.join(self.output_folder, 'iou_ratio_cam_{}.csv'.format(camera)), 'w')
        csv_writer = csv.writer(csv_file)

        # For every frame in the dictionaries
        frame_idxs = frame_dict_all['0'].keys()
        for frame_idx in frame_idxs:
            image_projections = []
            for key in frame_dict_all:
                _, image_projection = self._get_projections(PC, frame_dict_all[key][frame_idx])
                image_projections.append(image_projection)
            # if any of the projections is outside of the image, ignore this frame
            if any([proj[0] < 0 or int(proj[0]) >= self.frame_width or
                                    proj[1] < 0 or int(proj[1]) >= self.frame_height for proj in image_projection for image_projection in image_projections]):
                continue

            # Compute the bounding box of the two frames
            frame_bbox_1 = self.bbox_obj.get_bounding_box(image_projections[0])
            frame_bbox_2 = self.bbox_obj.get_bounding_box(image_projections[1])
            # Bounding boxes are [x, y, width, height] but need to be [min_x, min_y, max_y, max_y] for iou computation
            frame_bbox_1 = [frame_bbox_1[0], frame_bbox_1[1], frame_bbox_1[0]+frame_bbox_1[2], frame_bbox_1[1]+frame_bbox_1[3]]
            frame_bbox_2 = [frame_bbox_2[0], frame_bbox_2[1], frame_bbox_2[0]+frame_bbox_2[2], frame_bbox_2[1]+frame_bbox_2[3]]
            if frame_bbox_1[2] > self.frame_width or frame_bbox_1[3] > self.frame_height:
                continue
            if frame_bbox_2[2] > self.frame_width or frame_bbox_2[3] > self.frame_height:
                continue

            # Get the iou of the two frames
            iou_ratio = bb_intersection_over_union(frame_bbox_1, frame_bbox_2)

            # Create the absolute frame number
            absolute_frame_idx = frame_idx - this_camera_dict['synchronization_frame']
            absolute_frame_str = str(absolute_frame_idx).zfill(6)
            # Write to the csv
            csv_writer.writerow([absolute_frame_str, iou_ratio])

        csv_file.close()

    def run_pipeline(self):
        # Read the project file to get the offset information. The Offset information is added to camera dictionary
        self.camera_dict = self.synchronization_reader.SynchronizationReader(
            self.synchronization_file).get_camera_dict()
        # Read the calibration file to get the distortion information
        self.camera_dict = self.camera_reader.CameraReader(self.camera_dict).read_camera_file(
                self.camera_calibration_file_path, None)

        for camera in self.camera_dict.keys():
            self._iou_pipeline(camera)


if __name__ == '__main__':
    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--cfg_path',
                                 required=True,
                                 help="Configuration file for the task")
    args = argument_parser.parse_args()
    pipeline = IOU_Computation(args.cfg_path)
    ret_dict = pipeline.run_pipeline()
