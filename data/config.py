# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# TextBoxes300 CONFIGS
# min_ratio = 20
# max_ratio = 95
# step_size = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2))) #18
cfg300 = {
    'num_classes': 2,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10], [2,3,5,7,10]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': '300'
}

# Original Code
# 'min_sizes': [30, 60, 114, 168, 222, 276],
# 'max_sizes': [[], 114, 168, 222, 276, 330],
# each layer has 12, 14, 14, 14, 14, 14 default boxes per cell, 
# the first layer has only 2 as=1 default boxes, while the other has 4.
# My implements has 14 default boxes for each cell
