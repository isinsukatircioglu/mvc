import pickle
import csv
import shutil

pickle_path = '/Users/victor/workspace/data/BoxingDataV2/Sequence8/data_pickle.pkl'
csv_list = ['/Users/victor/workspace/data/BoxingDataV2/Sequence8/iou_ratio_cam_0.csv',
            '/Users/victor/workspace/data/BoxingDataV2/Sequence8/iou_ratio_cam_1.csv',
            '/Users/victor/workspace/data/BoxingDataV2/Sequence8/iou_ratio_cam_2.csv',
            '/Users/victor/workspace/data/BoxingDataV2/Sequence8/iou_ratio_cam_3.csv',
            '/Users/victor/workspace/data/BoxingDataV2/Sequence8/iou_ratio_cam_4.csv',
            '/Users/victor/workspace/data/BoxingDataV2/Sequence8/iou_ratio_cam_5.csv',
            '/Users/victor/workspace/data/BoxingDataV2/Sequence8/iou_ratio_cam_6.csv']

# Create a dictionary containg the iou info per camera
iou_dict = {}
for cam_idx, csv_path in enumerate(csv_list):
    iou_dict[str(cam_idx)] = {}
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            iou_dict[str(cam_idx)][line[0]] = float(line[1])

# Create a backup of the original file
backup_pickle_path = pickle_path+'.bkp'
shutil.copy2(pickle_path, backup_pickle_path)

# Load the original data and add the iou info
orig_data = pickle.load(open(pickle_path, 'rb'))
for actor in orig_data['actors']:
    for frame_id in orig_data['actors'][actor]:
        for cam_idx in orig_data['actors'][actor][frame_id]:
            orig_data['actors'][actor][frame_id][cam_idx]['iou_ratio'] = iou_dict[cam_idx][frame_id]

# Overwrite the pickle file
pickle.dump(orig_data, open(pickle_path, 'wb'), protocol=2)
