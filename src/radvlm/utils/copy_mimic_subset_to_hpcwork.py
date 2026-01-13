from PIL import Image
import os
import sys
import pydicom
import numpy as np
import subprocess
from src.radvlm.utils.config import DATA_ORIGINAL_DIR, DATA_RAW_DIR_FULL_DATASET, DATA_PROCESSED_DIR
from src.radvlm.utils.preprocess_images import copy_data_to_new_dir

if __name__ == "__main__":
    # src = os.path.join(DATA_ORIGINAL_DIR, "p10")
    # dst = os.path.join(DATA_RAW_DIR_FULL_DATASET, "p10")
    src = DATA_ORIGINAL_DIR
    dst = DATA_RAW_DIR_FULL_DATASET

    copy_data_to_new_dir(src, dst)