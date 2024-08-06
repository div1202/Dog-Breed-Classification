from detectron2.engine import DefaultPredictor

import os
import pickle

from utils import *

cfg_save_path = ".\detectron2\OD_BC_cfg.pickle"

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.49

predictor = DefaultPredictor(cfg)

image_path="Path to your image file"
on_image(image_path, predictor)