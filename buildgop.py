import os
from datetime import datetime
from math import floor
import copy
import torch
import wandb as wb
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import auc, f1_score, roc_curve
from packages.video_utils import H264Extractor, Video
from packages.constants import GOP_SIZE, FRAME_HEIGHT, FRAME_WIDTH, DATASET_ROOT, N_GOPS_FROM_DIFFERENT_DEVICE, N_GOPS_FROM_SAME_DEVICE, SAME_DEVICE_LABEL
from packages.dataset import VisionGOPDataset, GopPairDataset
from packages.common import create_custom_logger
from packages.network import H4vdmNet

if not os.path.exists(DATASET_ROOT):
    raise Exception(f'Dataset root does not exist: {DATASET_ROOT}')

log = create_custom_logger('h4vdm.ipynb')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f'Using device: {device}')

bin_path = os.path.abspath(os.path.join(os.getcwd(), 'h264-extractor', 'bin'))
h264_ext_bin = os.path.join(bin_path, 'h264dec_ext_info')
h264_extractor = H264Extractor(bin_filename=h264_ext_bin, cache_dir=DATASET_ROOT)
Video.set_h264_extractor(h264_extractor)

vision_gop_dataset = VisionGOPDataset(
    root_path=DATASET_ROOT,
    devices=[],
    media_types = ['videos'],
    properties=[],
    extensions=['mp4', 'mov', '3gp'],
    gop_size=GOP_SIZE,
    frame_width=FRAME_WIDTH,
    frame_height=FRAME_HEIGHT,
    gops_per_video=4,
    build_on_init=False,
    force_rebuild=False,
    download_on_init=False,
    ignore_local_dataset=False,
    shuffle=False)

is_loaded = vision_gop_dataset.load()
if not is_loaded:
    log.info('Dataset was not loaded. Building...')
else:
    log.info('Dataset was loaded.')

print(f'Dataset length: {len(vision_gop_dataset)}')

for device in vision_gop_dataset.get_devices():
    for video_metadata in vision_gop_dataset.dataset[device]:
        print(f'Processing video: {video_metadata["filename"]}')
        video = vision_gop_dataset._get_video_from_metadata(video_metadata)
        gops = video.get_gops()

        Video.h264_extractor.clean_cache()
        video = None
        gops = None
        del video
        del gops

print('Done')
