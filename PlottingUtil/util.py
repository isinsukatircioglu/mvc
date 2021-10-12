import matplotlib
from matplotlib import pyplot as plt

from matplotlib import cm
import numpy as np
from scipy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import pdb
import IPython

bones_h36m = [[0, 1], [1, 2], [2, 3],
             [0, 4], [4, 5], [5, 6],
             [0, 7], [7, 8], [8, 9], [9, 10],
             [8, 14], [14, 15], [15, 16],
             [8, 11], [11, 12], [12, 13],
            ]


bones_h36mXXX = [
             [0, 7], [7, 8], [8, 9], [9, 10],
             [8, 14], [14, 15], [15, 16],
             [8, 11], [11, 12], [12, 13],
            ]

bones_mpii = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
         [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
         [6, 8], [8, 9]]
root_index_h36m = 0

bones_hand = [[0, 1], [1, 2], [2, 3], # little finger, starting from tip
              [4, 5], [5, 6], [6, 7], 
              [8, 9], [9, 10], [10, 11], 
              [12, 13], [13, 14], [14, 15], 
              [16, 17], [17, 18], [18, 19]] # thumb
root_index_hand = 19


joint_indices_ski=[    0,             1,          2,           3,            4,          5,           6,       7,     8,      9,        10,        11,            12,         13,         14,             15,         16,         17,         18,      19,        20,   21,   22,   23]
joint_names_ski_meyer_2d = ['head',  'lshoulder',   'lelbow',      'lhand',     'lpole','rshoulder',    'relbow', 'rhand','rpole','lhip',   'lknee',  'lankle',    'lfoottip',    'lheel',   'lskitip',        'rhip',    'rknee',   'rankle', 'rfoottip', 'rheel', 'rskitip', 'b3', 'b4', 'b5']
joint_names_ski_meyer = ['head',      'torso','upper arm right','upper arm left','lower arm right','lower arm left','hand right','hand left','upper leg right','upper leg left','lower leg right','lower leg left','foot right','foot left','pole right','pole left','morpho point',]

joint_names_ski_spoerri = ['HeadGPSAntenna', 'Head', 'Neck', #0,1,3
                           'RightHip', 'LeftHip', # 3,4
                           'RightKnee', 'LeftKnee', # 5,6
                           'RightAnkle', 'LeftAnkle', # 7,8
                           'RightShoulder', 'LeftShoulder', #9,10
                           'RightElbow', 'LeftElbow', # 11,12
                           'RightHand', 'LeftHand', #13,14
                           'RightSkiTip', 'RightSkiTail', #15,16
                           'LeftSkiTip', 'LeftSkiTail', #17,18
                           'RightStickTail', 'LeftStickTail', #19,20
                           'COM_body', 'COM_system', #21, 22
                           ]

joint_indices_h36m=[    0,             1,          2,           3,            4,         5,           6,       7,     8,      9,        10,        11,            12,         13,         14,             15,         16 ]
joint_weights_h36m=[  6.3,          10.0,       10.3,         5.8,         10.0,      10.3,         5.8,       0,   5.0,    7.3,       1.0,      10.0,           2.3,        1.6,       10.0,            2.3,        1.6 ]
joint_names_h36m = ['hip','right_up_leg','right_leg','right_foot','left_up_leg','left_leg', 'left_foot','spine1','neck', 'head','head-top','left-arm','left_forearm','left_hand','right_arm','right_forearm','right_hand']

joint_symmetry_h36m = [[0,0],[1,4],[2,5],[3,6],[7,7],[8,8],[9,9],[10,10],[11,14],[12,15],[13,16]]
joint_torso       = [0,1,4,7,8,11,14]
joint_limbs       = [joint_names_h36m.index('right_leg'),joint_names_h36m.index('right_foot'),joint_names_h36m.index('right_forearm'),joint_names_h36m.index('right_hand'),
                     joint_names_h36m.index('left_leg'),joint_names_h36m.index('left_foot'),joint_names_h36m.index('left_forearm'),joint_names_h36m.index('left_hand')]

bones_h36m_limbSymmetries = [
    [bones_h36m.index([joint_names_h36m.index('left_up_leg'), joint_names_h36m.index('left_leg')]),bones_h36m.index([joint_names_h36m.index('right_up_leg'), joint_names_h36m.index('right_leg')])],
    [bones_h36m.index([joint_names_h36m.index('left_leg'), joint_names_h36m.index('left_foot')]),bones_h36m.index([joint_names_h36m.index('right_leg'), joint_names_h36m.index('right_foot')])],
    [bones_h36m.index([joint_names_h36m.index('left-arm'), joint_names_h36m.index('left_forearm')]),bones_h36m.index([joint_names_h36m.index('right_arm'), joint_names_h36m.index('right_forearm')])],
    [bones_h36m.index([joint_names_h36m.index('left_forearm'), joint_names_h36m.index('left_hand')]),bones_h36m.index([joint_names_h36m.index('right_forearm'), joint_names_h36m.index('right_hand')])],
    ]

joint_names_h36m_origNames = ['Hips',
                              'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase', 'Site',
                              'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase', 'Site',
                              'Spine', 'Spine1', 'Neck', 'Head', 'Site-head',
                              'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandThumb', 'Site',
                              'L_Wrist_End', 'Site',
                              'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandThumb', 'Site',
                              'R_Wrist_End', 'Site']

#joint_names_h36m = ['hip', 'right_up_leg', 'right_leg', 'right_foot', 'left_up_leg', 'left_leg', 'left_foot',
#                    'spine1', 'neck', 'head', 'head-top',
#                    'left-arm', 'left_forearm', 'left_hand',
#                    'right_arm', 'right_forearm', 'right_hand']

joint_names_h36m_origNames_newOrder = [
    'Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot',
    'Spine1', 'Neck', 'Head', 'Site-head',
    'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightArm', 'RightForeArm', 'RightHand']

h36m_origTo17 = [joint_names_h36m_origNames.index(key) for key in joint_names_h36m_origNames_newOrder]

joint_indices_cpm =[    0,             1,          2,           3,            4,         5,           6,       7,     8,      9,        10,        11,            12,         13,         14]
joint_names_cpm =  ['head',        'neck',    'Rsho',      'Relb',       'Rwri',    'Lsho',      'Lelb',  'Lwri','Rhip', 'Rkne',    'Rank',    'Lhip',        'Lkne',     'Lank',     'root']

joint_indices_mpii=[    0,             1,          2,           3,            4,         5,           6,       7,     8,      9,        10,        11,            12,         13,         14,             15,         16 ]
joint_names_mpii=  ['Rank',        'Rkne',    'Rhip',      'Lhip',       'Lkne',    'Lank',      'root',  '????','neck', 'head',    'Rwri',    'Relb',        'Rsho',     'Lsho',     'Lelb',           'Lwri']
joint_symmetry_mpii = [[0,5],[1,4],[2,3], [6,6],[7,7],[8,8],[9,9],[10,15],[11,14],[12,13],[16,16]]

cpm2h36m =         [   14,             8,          9,          10,           11,        12,          13,      14,     1,      0,        0,         5,              6,          7,          2,              3,         4  ]
mpii_to_cpm =      [    9,             8,         12,          11,           10,        13,          14,      15,     2,      1,        0,         3,              4,          5,          6,              7]
mpii_to_h36m = list(np.array(mpii_to_cpm)[cpm2h36m])
h36m_to_mpii = [3, 2, 1, 4, 5, 6, 0, 8, 9, 10, 16, 15, 14, 11, 12, 13]

joint_names_ski_spoerri = ['HeadGPSAntenna', 'Head', 'Neck', 'RightHip', 'LeftHip', 'RightKnee', 'LeftKnee', 'RightAnkle', 'LeftAnkle', 'RightShoulder', 'LeftShoulder', 'RightElbow', 'LeftElbow',
                           'RightHand', 'LeftHand', 'RightSkiTip', 'RightSkiTail', 'LeftSkiTip', 'LeftSkiTail', 'RightStickTail', 'LeftStickTail', 'COM_body', 'COM_system']
color_order_spoerri = [6,7,27, # Antenna, head, neck
                       0,10, # Hips
                       1,11, # Knees
                       2,12, # Ankle
                       3,13, # Shoulder
                       4,14,# Elbow
                       5,15,# hand
                       8,9,# rski
                       25,26,# lski
                      ]
io = joint_names_ski_spoerri.index
joint_symmetry_spoerri = [[io('HeadGPSAntenna'),io('HeadGPSAntenna')],
                          [io('Head'),io('Head')],
                          [io('Neck'),io('Neck')],
                          [io('RightHip'),io('LeftHip')],
                          [io('RightKnee'),io('LeftKnee')],
                          [io('RightAnkle'),io('LeftAnkle')],
                          [io('RightShoulder'),io('LeftShoulder')],
                          [io('RightElbow'),io('LeftElbow')],
                          [io('RightHand'), io('LeftHand')],
                          [io('RightSkiTip'), io('LeftSkiTip')],
                          [io('RightSkiTail'), io('LeftSkiTail')],
                          [io('RightStickTail'), io('LeftStickTail')],
                          ]

ski_spoerri_to_h36m     = np.zeros((17,21))
ski_spoerri_to_h36m[joint_names_h36m.index('hip'), joint_names_ski_spoerri.index('LeftHip')] = 0.5
ski_spoerri_to_h36m[joint_names_h36m.index('hip'), joint_names_ski_spoerri.index('RightHip')] = 0.5
ski_spoerri_to_h36m[joint_names_h36m.index('right_up_leg'), joint_names_ski_spoerri.index('LeftHip')] = 0.1 # shift a little, since the marker is at the leg outside
ski_spoerri_to_h36m[joint_names_h36m.index('right_up_leg'), joint_names_ski_spoerri.index('RightHip')] = 0.9
ski_spoerri_to_h36m[joint_names_h36m.index('right_leg'), joint_names_ski_spoerri.index('RightKnee')] = 1
ski_spoerri_to_h36m[joint_names_h36m.index('right_foot'), joint_names_ski_spoerri.index('RightAnkle')] = 1 #should be rheel, but is wirdly placed
ski_spoerri_to_h36m[joint_names_h36m.index('left_up_leg'), joint_names_ski_spoerri.index('LeftHip')] = 0.9
ski_spoerri_to_h36m[joint_names_h36m.index('left_up_leg'), joint_names_ski_spoerri.index('RightHip')] = 0.1
ski_spoerri_to_h36m[joint_names_h36m.index('left_leg'), joint_names_ski_spoerri.index('LeftKnee')] = 1
ski_spoerri_to_h36m[joint_names_h36m.index('left_foot'), joint_names_ski_spoerri.index('LeftAnkle')] = 1 #should be lheel, but is missing
head_hip_factor = 0.4
ski_spoerri_to_h36m[joint_names_h36m.index('spine1'), joint_names_ski_spoerri.index('LeftHip')] = 0.5*(1-head_hip_factor)
ski_spoerri_to_h36m[joint_names_h36m.index('spine1'), joint_names_ski_spoerri.index('RightHip')] = 0.5*(1-head_hip_factor)
ski_spoerri_to_h36m[joint_names_h36m.index('spine1'), joint_names_ski_spoerri.index('Head')] = head_hip_factor
head_hip_factor = 0.8
ski_spoerri_to_h36m[joint_names_h36m.index('neck'), joint_names_ski_spoerri.index('LeftHip')] = 0.5*(1-head_hip_factor)
ski_spoerri_to_h36m[joint_names_h36m.index('neck'), joint_names_ski_spoerri.index('RightHip')] = 0.5*(1-head_hip_factor)
ski_spoerri_to_h36m[joint_names_h36m.index('neck'), joint_names_ski_spoerri.index('Head')] = head_hip_factor
ski_spoerri_to_h36m[joint_names_h36m.index('head'), joint_names_ski_spoerri.index('Head')] = 1
ski_spoerri_to_h36m[joint_names_h36m.index('head-top'), joint_names_ski_spoerri.index('Head')] = 1 # TODO XXX HACK
ski_spoerri_to_h36m[joint_names_h36m.index('left-arm'), joint_names_ski_spoerri.index('LeftShoulder')] = 1
ski_spoerri_to_h36m[joint_names_h36m.index('left_forearm'), joint_names_ski_spoerri.index('LeftElbow')] = 1
ski_spoerri_to_h36m[joint_names_h36m.index('left_hand'), joint_names_ski_spoerri.index('LeftHand')] = 1
ski_spoerri_to_h36m[joint_names_h36m.index('right_arm'), joint_names_ski_spoerri.index('RightShoulder')] = 1
ski_spoerri_to_h36m[joint_names_h36m.index('right_forearm'), joint_names_ski_spoerri.index('RightElbow')] = 1
ski_spoerri_to_h36m[joint_names_h36m.index('right_hand'), joint_names_ski_spoerri.index('RightHand')] = 1



ski_spoerri_to_human_ski = np.zeros((19,21))
ski_spoerri_to_human_ski_poles = np.eye(21)
for i,vi in enumerate(range(14 + 2 + 2 + 1)):
    ski_spoerri_to_human_ski[i,vi] = 1
#root_index_ski = [joint_names_ski_spoerri.index('Neck')] # Neck (there is no spine)
root_index_ski = [joint_names_ski_spoerri.index('LeftHip'),joint_names_ski_spoerri.index('RightHip')] # mean of bot hips
bones_ski = [[0,1], [1,2], [3,4], [3,5], [4,6], [5,7], [6,8], [9,11], [10,12], [2,9], [2,10], [11,13], [12,14], [15,16], [17,18],
             [joint_names_ski_spoerri.index('Neck'),joint_names_ski_spoerri.index('LeftHip')],
             [joint_names_ski_spoerri.index('Neck'),joint_names_ski_spoerri.index('RightHip')]]  # with skies

joint_names_roman = ['head_top', 'neck',
                             'right_shoulder', 'right_ellbow', 'right_hand', 'right_pole_basket',
                             'left_shoulder', 'left_ellbow', 'left_hand', 'left_pole_basket',
                             'right_hip', 'right_knee', 'right_ankle',
                             'left_hip', 'left_knee', 'left_ankle',
                             'right_ski_tip', 'right_toes', 'right_heel', 'right_ski_rear',
                             'left_ski_tip', 'left_toes', 'left_heel', 'left_ski_rear']
bones_roman = [[0,1], [1,2], [2,3], [3,4], [4,5], [1,6], [6,7], [7,8], [8,9],
                        [2,10], [10,11], [11,12], [6,13], [13,14], [14,15],
                        [16,17], [17,18], [18,19], [12,17], [12,18],
                        [20,21], [21,22], [22,23], [15,21], [15,22]]
ski_spoerri_to_roman     = np.zeros((24,21))
ski_spoerri_to_roman[joint_names_roman.index('head_top'), joint_names_ski_spoerri.index('Head')] = 1
ski_spoerri_to_roman[joint_names_roman.index('neck'), joint_names_ski_spoerri.index('LeftHip')] = 0.5*(1-head_hip_factor)
ski_spoerri_to_roman[joint_names_roman.index('neck'), joint_names_ski_spoerri.index('RightHip')] = 0.5*(1-head_hip_factor)
ski_spoerri_to_roman[joint_names_roman.index('neck'), joint_names_ski_spoerri.index('Head')] = head_hip_factor
ski_spoerri_to_roman[joint_names_roman.index('right_shoulder'), joint_names_ski_spoerri.index('RightShoulder')] = 1
ski_spoerri_to_roman[joint_names_roman.index('right_ellbow'), joint_names_ski_spoerri.index('RightElbow')] = 1
ski_spoerri_to_roman[joint_names_roman.index('right_hand'), joint_names_ski_spoerri.index('RightHand')] = 1
ski_spoerri_to_roman[joint_names_roman.index('right_pole_basket'), joint_names_ski_spoerri.index('RightStickTail')] = 1
ski_spoerri_to_roman[joint_names_roman.index('left_shoulder'), joint_names_ski_spoerri.index('LeftShoulder')] = 1
ski_spoerri_to_roman[joint_names_roman.index('left_ellbow'), joint_names_ski_spoerri.index('LeftElbow')] = 1
ski_spoerri_to_roman[joint_names_roman.index('left_hand'), joint_names_ski_spoerri.index('LeftHand')] = 1
ski_spoerri_to_roman[joint_names_roman.index('left_pole_basket'), joint_names_ski_spoerri.index('LeftStickTail')] = 1
ski_spoerri_to_roman[joint_names_roman.index('right_hip'), joint_names_ski_spoerri.index('LeftHip')] = 0.1 # shift a little, since the marker is at the leg outside
ski_spoerri_to_roman[joint_names_roman.index('right_hip'), joint_names_ski_spoerri.index('RightHip')] = 0.9
ski_spoerri_to_roman[joint_names_roman.index('right_knee'), joint_names_ski_spoerri.index('RightKnee')] = 1
ski_spoerri_to_roman[joint_names_roman.index('right_ankle'), joint_names_ski_spoerri.index('RightAnkle')] = 1 #should be rheel, but is wirdly placed
ski_spoerri_to_roman[joint_names_roman.index('left_hip'), joint_names_ski_spoerri.index('RightHip')] = 0.1 # shift a little, since the marker is at the leg outside
ski_spoerri_to_roman[joint_names_roman.index('left_hip'), joint_names_ski_spoerri.index('LeftHip')] = 0.9
ski_spoerri_to_roman[joint_names_roman.index('left_knee'), joint_names_ski_spoerri.index('LeftKnee')] = 1
ski_spoerri_to_roman[joint_names_roman.index('left_ankle'), joint_names_ski_spoerri.index('LeftAnkle')] = 1 #should be rheel, but is wirdly placed

ski_spoerri_to_roman[joint_names_roman.index('right_ski_tip'), joint_names_ski_spoerri.index('RightSkiTip')] = 1
ski_spoerri_to_roman[joint_names_roman.index('right_toes'), joint_names_ski_spoerri.index('RightSkiTip')] = 0.55
ski_spoerri_to_roman[joint_names_roman.index('right_toes'), joint_names_ski_spoerri.index('RightSkiTail')] = 0.45
ski_spoerri_to_roman[joint_names_roman.index('right_heel'), joint_names_ski_spoerri.index('RightSkiTip')] = 0.35
ski_spoerri_to_roman[joint_names_roman.index('right_heel'), joint_names_ski_spoerri.index('RightSkiTail')] = 0.65
ski_spoerri_to_roman[joint_names_roman.index('right_ski_rear'), joint_names_ski_spoerri.index('RightSkiTail')] = 1 #should be lheel, but is missing
ski_spoerri_to_roman[joint_names_roman.index('left_ski_tip'), joint_names_ski_spoerri.index('LeftSkiTip')] = 1
ski_spoerri_to_roman[joint_names_roman.index('left_toes'), joint_names_ski_spoerri.index('LeftSkiTip')] = 0.55
ski_spoerri_to_roman[joint_names_roman.index('left_toes'), joint_names_ski_spoerri.index('LeftSkiTail')] = 0.45
ski_spoerri_to_roman[joint_names_roman.index('left_heel'), joint_names_ski_spoerri.index('LeftSkiTip')] = 0.35
ski_spoerri_to_roman[joint_names_roman.index('left_heel'), joint_names_ski_spoerri.index('LeftSkiTail')] = 0.65
ski_spoerri_to_roman[joint_names_roman.index('left_ski_rear'), joint_names_ski_spoerri.index('LeftSkiTail')] = 1 #should be lheel, but is missing

# matrix to map from ski to h36m (inverse to map in the other way)
ski_to_h36m     = np.zeros((17,17))
ski_to_h36m[joint_names_h36m.index('hip'), joint_names_ski_meyer.index('upper leg left')] = 0.5
ski_to_h36m[joint_names_h36m.index('hip'), joint_names_ski_meyer.index('upper leg right')] = 0.5
ski_to_h36m[joint_names_h36m.index('right_up_leg'), joint_names_ski_meyer.index('upper leg left')] = 0.1 # shift a little, since the marker is at the leg outside
ski_to_h36m[joint_names_h36m.index('right_up_leg'), joint_names_ski_meyer.index('upper leg right')] = 0.9
ski_to_h36m[joint_names_h36m.index('right_leg'), joint_names_ski_meyer.index('lower leg right')] = 1
ski_to_h36m[joint_names_h36m.index('right_foot'), joint_names_ski_meyer.index('foot right')] = 1 #should be rheel, but is wirdly placed
ski_to_h36m[joint_names_h36m.index('left_up_leg'), joint_names_ski_meyer.index('upper leg left')] = 0.9
ski_to_h36m[joint_names_h36m.index('left_up_leg'), joint_names_ski_meyer.index('upper leg right')] = 0.1
ski_to_h36m[joint_names_h36m.index('left_leg'), joint_names_ski_meyer.index('lower leg left')] = 1
ski_to_h36m[joint_names_h36m.index('left_foot'), joint_names_ski_meyer.index('foot left')] = 1 #should be lheel, but is missing
head_hip_factor = 0.4
ski_to_h36m[joint_names_h36m.index('spine1'), joint_names_ski_meyer.index('upper leg left')] = 0.5*(1-head_hip_factor)
ski_to_h36m[joint_names_h36m.index('spine1'), joint_names_ski_meyer.index('upper leg right')] = 0.5*(1-head_hip_factor)
ski_to_h36m[joint_names_h36m.index('spine1'), joint_names_ski_meyer.index('head')] = head_hip_factor
head_hip_factor = 0.8
ski_to_h36m[joint_names_h36m.index('neck'), joint_names_ski_meyer.index('upper leg left')] = 0.5*(1-head_hip_factor)
ski_to_h36m[joint_names_h36m.index('neck'), joint_names_ski_meyer.index('upper leg right')] = 0.5*(1-head_hip_factor)
ski_to_h36m[joint_names_h36m.index('neck'), joint_names_ski_meyer.index('head')] = head_hip_factor
#ski_to_h36m[joint_names_h36m.index('neck'), joint_names_ski_meyer.index('head')] = 1
ski_to_h36m[joint_names_h36m.index('head'), joint_names_ski_meyer.index('head')] = 1
ski_to_h36m[joint_names_h36m.index('head-top'), joint_names_ski_meyer.index('head')] = 1 # TODO XXX HACK
ski_to_h36m[joint_names_h36m.index('left-arm'), joint_names_ski_meyer.index('upper arm left')] = 1
ski_to_h36m[joint_names_h36m.index('left_forearm'), joint_names_ski_meyer.index('lower arm left')] = 1
ski_to_h36m[joint_names_h36m.index('left_hand'), joint_names_ski_meyer.index('hand left')] = 1
ski_to_h36m[joint_names_h36m.index('right_arm'), joint_names_ski_meyer.index('upper arm right')] = 1
ski_to_h36m[joint_names_h36m.index('right_forearm'), joint_names_ski_meyer.index('lower arm right')] = 1
ski_to_h36m[joint_names_h36m.index('right_hand'), joint_names_ski_meyer.index('hand right')] = 1

ski_to_h36m_meyer_2d     = np.zeros((17,20))
ski_to_h36m_meyer_2d[joint_names_h36m.index('hip'), joint_names_ski_meyer_2d.index('lhip')] = 0.5
ski_to_h36m_meyer_2d[joint_names_h36m.index('hip'), joint_names_ski_meyer_2d.index('rhip')] = 0.5
ski_to_h36m_meyer_2d[joint_names_h36m.index('right_up_leg'), joint_names_ski_meyer_2d.index('lhip')] = 0.1 # shift a little, since the marker is at the leg outside
ski_to_h36m_meyer_2d[joint_names_h36m.index('right_up_leg'), joint_names_ski_meyer_2d.index('rhip')] = 0.9
ski_to_h36m_meyer_2d[joint_names_h36m.index('right_leg'), joint_names_ski_meyer_2d.index('rknee')] = 1
ski_to_h36m_meyer_2d[joint_names_h36m.index('right_foot'), joint_names_ski_meyer_2d.index('rankle')] = 1 #should be rheel, but is wirdly placed
ski_to_h36m_meyer_2d[joint_names_h36m.index('left_up_leg'), joint_names_ski_meyer_2d.index('lhip')] = 0.9
ski_to_h36m_meyer_2d[joint_names_h36m.index('left_up_leg'), joint_names_ski_meyer_2d.index('rhip')] = 0.1
ski_to_h36m_meyer_2d[joint_names_h36m.index('left_leg'), joint_names_ski_meyer_2d.index('lknee')] = 1
ski_to_h36m_meyer_2d[joint_names_h36m.index('left_foot'), joint_names_ski_meyer_2d.index('lankle')] = 1 #should be lheel, but is missing
# ski_to_h36m_meyer_2d[joint_names_h36m.index('right_up_leg'), joint_names_ski_meyer_2d.index('rknee')] = 1
# ski_to_h36m_meyer_2d[joint_names_h36m.index('right_leg'), joint_names_ski_meyer_2d.index('rankle')] = 1
# ski_to_h36m_meyer_2d[joint_names_h36m.index('right_foot'), joint_names_ski_meyer_2d.index('rfoottip')] = 1 #should be rheel, but is wirdly placed
# ski_to_h36m_meyer_2d[joint_names_h36m.index('left_up_leg'), joint_names_ski_meyer_2d.index('lknee')] = 1
# ski_to_h36m_meyer_2d[joint_names_h36m.index('left_leg'), joint_names_ski_meyer_2d.index('lankle')] = 1
# ski_to_h36m_meyer_2d[joint_names_h36m.index('left_foot'), joint_names_ski_meyer_2d.index('lfoottip')] = 1 #should be lheel, but is missing
head_hip_factor = 0.4
ski_to_h36m_meyer_2d[joint_names_h36m.index('spine1'), joint_names_ski_meyer_2d.index('lhip')] = 0.5*(1-head_hip_factor)
ski_to_h36m_meyer_2d[joint_names_h36m.index('spine1'), joint_names_ski_meyer_2d.index('rhip')] = 0.5*(1-head_hip_factor)
ski_to_h36m_meyer_2d[joint_names_h36m.index('spine1'), joint_names_ski_meyer_2d.index('head')] = head_hip_factor
head_hip_factor = 0.8
ski_to_h36m_meyer_2d[joint_names_h36m.index('neck'), joint_names_ski_meyer_2d.index('lhip')] = 0.5*(1-head_hip_factor)
ski_to_h36m_meyer_2d[joint_names_h36m.index('neck'), joint_names_ski_meyer_2d.index('rhip')] = 0.5*(1-head_hip_factor)
ski_to_h36m_meyer_2d[joint_names_h36m.index('neck'), joint_names_ski_meyer_2d.index('head')] = head_hip_factor
#ski_to_h36m_meyer_2d[joint_names_h36m.index('neck'), joint_names_ski_meyer_2d.index('head')] = 1
ski_to_h36m_meyer_2d[joint_names_h36m.index('head'), joint_names_ski_meyer_2d.index('head')] = 1
ski_to_h36m_meyer_2d[joint_names_h36m.index('head-top'), joint_names_ski_meyer_2d.index('head')] = 1 # TODO XXX HACK
ski_to_h36m_meyer_2d[joint_names_h36m.index('left-arm'), joint_names_ski_meyer_2d.index('lshoulder')] = 1
ski_to_h36m_meyer_2d[joint_names_h36m.index('left_forearm'), joint_names_ski_meyer_2d.index('lelbow')] = 1
ski_to_h36m_meyer_2d[joint_names_h36m.index('left_hand'), joint_names_ski_meyer_2d.index('lhand')] = 1
ski_to_h36m_meyer_2d[joint_names_h36m.index('right_arm'), joint_names_ski_meyer_2d.index('rshoulder')] = 1
ski_to_h36m_meyer_2d[joint_names_h36m.index('right_forearm'), joint_names_ski_meyer_2d.index('relbow')] = 1
ski_to_h36m_meyer_2d[joint_names_h36m.index('right_hand'), joint_names_ski_meyer_2d.index('rhand')] = 1

p0 = np.array([1, 3, 2])
p1 = np.array([8, 5, 9])

def filterBones(bones, visibile):
    bones_filtered = []
    for bone in bones:
        if visibile[bone[0]] and visibile[bone[1]]:
            bones_filtered.append(bone)
        else:
            bones_filtered.append([0,0])

    return bones_filtered

def plot3Dsphere(ax, p, radius=5, color=(0.5, 0.5, 0.5)):
    num_samples = 8
    u = np.linspace(0, 2 * np.pi, num_samples)
    v = np.linspace(0, np.pi, num_samples)
    
    x = p[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = p[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = p[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    c = np.ones( (list(x.shape)+[len(color)]) )
    c[:,:] = color
    #ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=color, alpha=1)
    return x, y, z, c
 
def plot3Dcylinder(ax, p0, p1, radius=5, color=(0.5, 0.5, 0.5)):
    num_samples = 8
    origin = np.array([0, 0, 0])
    #vector in direction of axis
    v = p1 - p0
    mag = norm(v)
    if mag==0: # prevent division by 0 for bones of length 0
        return np.zeros((0,0)),np.zeros((0,0)),np.zeros((0,0)),np.zeros((0,0))
    #unit vector in direction of axis
    v = v / mag
    #make some vector not in the same direction as v
    not_v = np.array([1, 0, 0])
    eps = 0.00001
    if norm(v-not_v)<eps:
        not_v = np.array([0, 1, 0])
    #make vector perpendicular to v
    n1 = np.cross(v, not_v)
    n1 /= norm(n1)
    #make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    #surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(0, mag, 2)
    theta = np.linspace(0, 2 * np.pi, num_samples)
    #use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    #generate coordinates for surface
    X, Y, Z = [p0[i] + v[i] * t + radius * np.sin(theta) * n1[i] + radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    #ax.plot_surface(X, Y, Z, color=color, alpha=0.25, shade=True)
    c = np.ones( (list(X.shape)+[4]) )
    c[:,:] = color #(1,1,1,0) #color
    return X, Y, Z, c

def plot_2Dpose(ax, pose_2d, bones, bones_dashed=[], bones_dashdot=[], colormap='hsv', linewidth=1, limits=None, color_order=[0, 5, 9, 15, 2, 10, 12, 4, 14, 13, 11, 3, 7, 8, 6, 1]):

    #print('pose_2d',pose_2d.shape)
    #plt.colormap(ax,'hsv') #'jet', 'nipy_spectral',tab20
    #cmap = plt.get_cmap('gist_ncar')
    #cmap = plt.cm.Set3(np.linspace(0, 1, 12))

#    cmap = plt.get_cmap('gist_ncar')
    cmap = plt.get_cmap(colormap)

    plt.axis('equal')
    maximum = max(color_order) #len(bones)
    for i, bone in enumerate(bones):
        colorIndex = (color_order[i] * cmap.N / float(maximum))
#        color = cmap(int(colorIndex))
#        colorIndex = i / len(bones)
        color = cmap(int(colorIndex))
        ax.plot(pose_2d[0, bone], pose_2d[1, bone], '-', color=color, linewidth=linewidth)
    for bone in bones_dashed:
        ax.plot(pose_2d[0, bone], pose_2d[1, bone], ':', color=color, linewidth=linewidth)
    for bone in bones_dashdot:
        ax.plot(pose_2d[0, bone], pose_2d[1, bone], '--', color=color, linewidth=linewidth)

    if not limits==None:
        ax.set_xlim(limits[0],limits[2])
        ax.set_ylim(limits[1],limits[3])

def plot_3Dpose_simple(ax, pose_3d, bones=bones_h36m, linewidth=1, colormap='gist_rainbow', color_order=[0, 5, 9, 15, 2, 10, 12, 4, 14, 13, 11, 3, 7, 8, 6, 1], plot_handles=None, autoAxisRange=True):
    "Used for live application (real-time)"
    pose_3d = np.reshape(pose_3d, (3, -1))

    X,Y,Z = np.squeeze(np.array(pose_3d[0,:])), np.squeeze(np.array(pose_3d[2,:])), np.squeeze(np.array(pose_3d[1,:]))
    XYZ = np.vstack([X,Y,Z])
    
    # order in z dimension (does not help)
    #bones_z = [0.5*XYZ[2,b[0]]+0.5*XYZ[2,b[1]] for b in bones]
    #bones_sorted = [b for _,b in sorted(zip(bones_z,bones))]
    #bone_i_sorted = [b for _,b in sorted(zip(bones_z,range(len(bones))))]

    if not plot_handles is None:
        for i, bone in enumerate(bones):
            plot_handles['lines'][i].set_data(XYZ[0:2, bone])
            plot_handles['lines'][i].set_3d_properties(XYZ[2,bone])
     #   for pi in range(pose_3d.shape[1]):
     #       plot_handles['points'][i].set_data(XYZ[0:2, pi])
     #       plot_handles['points'][i].set_3d_properties(XYZ[2,pi])
    else:
        ax.view_init(elev=0, azim=-90)
        #plt.colormap(ax,'hsv') #'jet', 'nipy_spectral',tab20
        cmap = plt.get_cmap(colormap)

        plot_handles = {'lines' : [], 'points': []}
        plt.axis('equal')
        maximum = len(bones) #max(color_order) #len(bones)
        for i, bone in enumerate(bones):
            assert i < len(color_order)
            #colorIndex = (color_order[i] * cmap.N / float(maximum))
            colorIndex = (i * cmap.N / float(maximum))
            color = cmap(int(colorIndex))
            line_handle = ax.plot(XYZ[0,bone], XYZ[1,bone], XYZ[2,bone], color=color, linewidth=linewidth, alpha=0.5, solid_capstyle='round')
            plot_handles['lines'].extend(line_handle) #for whatever reason plot already returns a list
       # for pi in range(XYZ.shape[1]):
       #     colorIndex = (color_order[pi % len(color_order)] * cmap.N / float(maximum))
       #     color = cmap(int(colorIndex))
       #     point_handle = ax.scatter(XYZ[0,pi], XYZ[1,pi], XYZ[2,pi], color=color, s=10*linewidth)
       #     # IPython.embed()
       #     plot_handles['points'].append(point_handle) #for whatever reason plot already returns a list
        # revert order
        #plot_handles['lines'] = [plot_handles['lines'][i] for i in bone_i_sorted]
       
        # maintain aspect ratio
        if autoAxisRange:
            max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

            mid_x = (X.max()+X.min()) * 0.5
            mid_y = (Y.max()+Y.min()) * 0.5
            mid_z = (Z.max()+Z.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

    return plot_handles
    #for bone in bones_lines:
    #    ax.plot3D(X[bone], Y[bone], Z[bone], '-')
    #for bone in bones_dashed:
    #    ax.plot3D(X[bone], Y[bone], Z[bone], ':')
    #for bone in bones_dashdot:
    #    ax.plot3D(X[bone], Y[bone], Z[bone], '--')



def plot_3Dpose(ax, pose_3d, bones, radius=10, colormap='gist_rainbow', color_order=[0, 5, 9, 15, 2, 10, 12, 4, 14, 13, 11, 3, 7, 8, 6, 1], set_limits=True, flip_yz=True, fixed_color = False, transparentBG=False):
    pose_3d = np.reshape(pose_3d, (3, -1))

    ax.view_init(elev=0, azim=-90)
    #plt.colormap(ax,'hsv') #'jet', 'nipy_spectral',tab20
    cmap = plt.get_cmap(colormap)

    if flip_yz:
        X,Y,Z = np.squeeze(np.array(pose_3d[0,:])), np.squeeze(np.array(pose_3d[2,:])), np.squeeze(np.array(pose_3d[1,:]))
    else:
        X,Y,Z = np.squeeze(np.array(pose_3d[0,:])), np.squeeze(np.array(pose_3d[1,:])), np.squeeze(np.array(pose_3d[2,:]))
    XYZ = np.vstack([X,Y,Z])
    
    # dummy bridge that connects different components (set to transparent) 
    def bridge_vertices(xs,ys,zs,cs, x,y,z,c):
        num_samples = x.shape[0]
        if num_samples == 0: # don't build a bridge if there is no data
            return
        if len(cs) > 0:
            x_bridge = np.hstack([xs[-1][:,-1].reshape(num_samples,1), x[:,0].reshape(num_samples,1)])
            y_bridge = np.hstack([ys[-1][:,-1].reshape(num_samples,1), y[:,0].reshape(num_samples,1)])
            z_bridge = np.hstack([zs[-1][:,-1].reshape(num_samples,1),z[:,0].reshape(num_samples,1)])
            c_bridge = np.ones( (num_samples,2,4) )
            c_bridge[:,:] = np.array([0,0,0,0])
            xs.append(x_bridge)
            ys.append(y_bridge)
            zs.append(z_bridge)
            cs.append(c_bridge)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        cs.append(c)
        return
        
    maximum = max(color_order) #len(bones)
    xs = []
    ys = []
    zs = []
    cs = []
    for i, bone in enumerate(bones):
        assert i < len(color_order)
        colorIndex = None
        if fixed_color == True:
            colorIndex = ( cmap.N / 4)
        else:
            colorIndex = (color_order[i] * cmap.N / float(maximum))
        # HACK for symmetry in skiing: colorIndex = (color_order[i] % 2)*cmap.N/2
        color = cmap(int(colorIndex))
        x,y,z,c = plot3Dcylinder(ax, XYZ[:,bone[0]], XYZ[:,bone[1]], radius=radius, color=color)
        bridge_vertices(xs,ys,zs,cs, x,y,z,c)
        #x,y,z,c = plot3Dsphere(ax, XYZ[:,bone[0]], radius=radius*1.2, color=color)
        #bridge_vertices(xs,ys,zs,cs, x,y,z,c)
        #x,y,z,c = plot3Dsphere(ax, XYZ[:,bone[1]], radius=radius*1.2, color=color)
        #bridge_vertices(xs,ys,zs,cs, x,y,z,c)
        
    if len(xs) == 0:
        return
        
    # merge all sufaces together to one big one
    x_full = np.hstack(xs)
    y_full = np.hstack(ys)
    z_full = np.hstack(zs)
    c_full = np.hstack(cs)

    ax.plot_surface(x_full, y_full, z_full, rstride=1, cstride=1, facecolors=c_full, linewidth=0, antialiased=True)

    # maintain aspect ratio
    #if set_limits_fixed:
    #    ax.set_xlim(-1.5, 1.5)
    #    ax.set_ylim(-1.5, 1.5)
    #    ax.set_zlim(-1.5, 1.5)
        
    if set_limits:
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2
    
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if transparentBG:
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off',
            labelsize=8) # labels along the bottom edge are off
        ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off',
            labelsize=8) # labels along the bottom edge are off
        ax.tick_params(
            axis='z',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off',
            labelsize=8) # labels along the bottom edge are off
        
        # make the bg white
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

def plot_3Dpose_batch(ax, batch_raw, offset_factor_x=None, offset_factor_y=None, bones=bones_h36m, radius=0.01, colormap='hsv', row_length=8):

#     def orthogonal_proj(zfront, zback):
#         a = (zfront+zback)/(zfront-zback)
#         b = -2*(zfront*zback)/(zfront-zback)
#         return np.array([[1,0,0,0],
#                             [0,1,0,0],
#                             [0,0,a,b],
#                             [0,0,0,zback]])
#     proj3d.persp_transformation = orthogonal_proj

    num_batch_indices = batch_raw.shape[0]
    batch = batch_raw.reshape(num_batch_indices, -1)
    num_joints  = batch.shape[1]//3
    num_bones  = len(bones)
    pose_3d_cat = batch.reshape((-1,3))

    #print('num_batch_indices = {}, num_joints = {}, num_bones = {}, batch.shape = {}, pose_3d_cat.shape = {}'.format(num_batch_indices,num_joints,num_bones,batch.shape, pose_3d_cat.shape))

    bones_cat = []
    color_order_cat = []
    for batchi in range(0,num_batch_indices):
        # offset bones
        bones_new = []
        offset = batchi*num_joints
        for bone in bones:
            bones_new.append([bone[0]+offset, bone[1]+offset])
        bones_cat.extend(bones_new)
        # offset colors
        color_order_cat.extend(range(0,num_bones))
        # offset skeletons horizontally
        if offset_factor_x is None:
            max_val_x = np.max(pose_3d_cat[num_joints*batchi:num_joints*(batchi+1),0])
            min_val_x = np.min(pose_3d_cat[num_joints*batchi:num_joints*(batchi+1),0])
            offset_factor_x = 1.5 * (max_val_x-min_val_x)
            radius = offset_factor_x/50
        if offset_factor_y is None:
            max_val_y = np.max(pose_3d_cat[num_joints*batchi:num_joints*(batchi+1),1])
            min_val_y = np.min(pose_3d_cat[num_joints*batchi:num_joints*(batchi+1),1])
            offset_factor_y = 1.3 * (max_val_y-min_val_y)
        offset_x = offset_factor_x*(batchi % row_length)
        offset_y = offset_factor_y*(batchi // row_length)
        pose_3d_cat[num_joints*batchi:num_joints*(batchi+1),:] += np.array([[offset_x,offset_y,0]])
    #print(bones_cat)
    plot_3Dpose(ax, pose_3d_cat.T, bones_cat, radius=radius, colormap=colormap, color_order=color_order_cat, transparentBG=True)
    
    #ax.tick_params(axis='x', colors='blue')
    #[t.set_color('red') for t in ax.get_xticklines()] 
    #[t.set_color('red') for t in ax.xaxis.get_ticklabels()]
    #[t.set_color('red') for t in ax.get_yticklines()] 
    #[t.set_color('red') for t in ax.yaxis.get_ticklabels()]
    #[t.set_color('red') for t in ax.get_zticklines()] 
    #[t.set_color('red') for t in ax.zaxis.get_ticklabels()]
    
    #[t.set_color('green') for t in ax.iter_ticks()]

    #plt.axis('off')
    
    #plt.setp(ax.get_xticks(), visible=False)
    #plt.setp(ax.get_yticks(), visible=False)
    

def plot_2Dpose_batch(ax, batch, offset_factor=0.8, bones=bones_h36m, colormap='hsv'):

#     def orthogonal_proj(zfront, zback):
#         a = (zfront+zback)/(zfront-zback)
#         b = -2*(zfront*zback)/(zfront-zback)
#         return np.array([[1,0,0,0],
#                             [0,1,0,0],
#                             [0,0,a,b],
#                             [0,0,0,zback]])
#     proj3d.persp_transformation = orthogonal_proj

    num_batches = batch.shape[0]
    pose_2d_batchlinear = batch.reshape((num_batches,-1))
    num_joints  = pose_2d_batchlinear.shape[1]//2
    num_bones  = len(bones)
    pose_2d_cat = batch.reshape((-1,2))
    #print('plot_2Dpose_batch: num_batches={}, num_joints={}'.format(num_batches,num_joints))
#         pose_3d_cat =

    bones_cat = []
    color_order_cat = []
    for batchi in range(0,num_batches):
        # offset bones
        bones_new = []
        offset_i = batchi*num_joints
        for bone in bones:
            bone_new = [bone[0]+offset_i, bone[1]+offset_i]
            if pose_2d_cat[bone_new[0],0] <=0 or pose_2d_cat[bone_new[0],1]<=0 or pose_2d_cat[bone_new[1],0] <=0 or pose_2d_cat[bone_new[1],1] <=0:
                bone_new = [offset_i,offset_i] # disable line drawing, but don't remove to not disturb color ordering
            bones_new.append(bone_new)

        bones_cat.extend(bones_new)
        # offset colors
        color_order_cat.extend(range(0,num_bones))
        # offset skeletons horizontally
        offset_x = offset_factor*(batchi %8)
        offset_y = offset_factor*(batchi//8)
        pose_2d_cat[num_joints*batchi:num_joints*(batchi+1),:] += np.array([[offset_x,offset_y]])
    #print(bones_cat)
    #plot_2Dpose(ax, pose_2d, bones, bones_dashed=[], bones_dashdot=[], color='red', linewidth=1, limits=None):
    plot_2Dpose(ax, pose_2d_cat.T, bones=bones_cat, colormap=colormap, color_order=color_order_cat)


def scatterToBarPlot(X,Y,num_bins):
        bins = np.linspace(0, max(X), num=num_bins)
        bin_assignment = np.digitize(X, bins)
        per_bin_error = [[] for i in range(num_bins)]
        for i_glob, i_bin in enumerate(bin_assignment):
            per_bin_error[(i_bin-1)].append(Y[i_glob])
        for list in per_bin_error:
            if len(list) == 0:
                list.append(0)
        per_bin_mean = [ np.mean(np.array(x)) for x in per_bin_error]
        per_bin_std = [ np.std(x) for x in per_bin_error]
        return bins, per_bin_mean, per_bin_std
    
    
def save_3Dpose_threeViews(pose_3d_centered, out_file_name, bones, invisible_coordinates=True):
        fig = plt.figure(0)
#         ax_img  = fig.add_subplot(1,1,1)
#         plt.xlim([0.0,img_full.shape[1]])
#         plt.ylim([0.0,img_full.shape[0]])
#         ax_img.set_axis_off()
#         ax_img.imshow(img_full)
    
        def prepareGrid(ax_3d):
            ax_3d.axis('equal');
            if not invisible_coordinates:
                plt.xlabel('x');plt.ylabel('y')
            else:
                plt.axis('off')
            for a in (ax_3d.w_xaxis, ax_3d.w_yaxis, ax_3d.w_zaxis):
                for t in a.get_ticklines()+a.get_ticklabels():
                    t.set_visible(False)
                a.line.set_visible(False)
                a.pane.set_visible(False)
        
        # viewed from left 
        ax_3d   = fig.add_subplot(131, projection='3d')
        prepareGrid(ax_3d)
        #ax_3d.xaxis.set_visible(False); ax_3d.yaxis.set_visible(False); ax_3d.set_axis_off();ax_3d.axis('equal');plt.xlabel('x');plt.ylabel('y')
        plot_3Dpose(ax_3d, pose_3d_centered.T, bones, radius=0.01)
        ax_3d.invert_zaxis()
        ax_3d.elev = 0
        ax_3d.azim = -180
        #ax_3d.dist = 8

        # original view
        ax_3d   = fig.add_subplot(132, projection='3d')
        prepareGrid(ax_3d)
        plot_3Dpose(ax_3d, pose_3d_centered.T, bones, radius=0.01)
        ax_3d.invert_zaxis()
        #ax_3d.elev = 0
        #ax_3d.azim = 90
        #ax_3d.dist = 8
        
        # viewed from right
        ax_3d   = fig.add_subplot(133, projection='3d')
        #ax_3d.xaxis.set_visible(False); ax_3d.yaxis.set_visible(False); ax_3d.set_axis_off();ax_3d.axis('equal');plt.xlabel('x');plt.ylabel('y')
        prepareGrid(ax_3d)
        plot_3Dpose(ax_3d, pose_3d_centered.T, bones, radius=0.01)
        ax_3d.invert_zaxis()
        ax_3d.elev = 90
        ax_3d.azim = -90
        #ax_3d.dist = 8

        print("Saving "+out_file_name)
        plt.savefig(out_file_name,  dpi=200)
        plt.close(fig)

def plot_camera(ax, pos, R, scale=1):
    x = pos + np.asarray(np.dot(np.array([1,0,0]),R.T)).squeeze()*scale
    y = pos + np.asarray(np.dot(np.array([0,1,0]),R.T)).squeeze()*scale
    z = pos + np.asarray(np.dot(np.array([0,0,1]),R.T)).squeeze()*scale*2
    ax.plot([pos[0],x[0]], [pos[1],x[1]], [pos[2],x[2]], color='red')
    ax.plot([pos[0],y[0]], [pos[1],y[1]], [pos[2],y[2]], color='green')
    ax.plot([pos[0],z[0]], [pos[1],z[1]], [pos[2],z[2]], color='blue')

def drawDummyPoints(ax):
    ax.plot([-1], [0], [1], ms=0.1)
    ax.plot([+1], [0], [1], ms=0.1)
    ax.plot([-1], [0], [-1], ms=0.1)
    ax.plot([+1], [0], [-1], ms=0.1)
