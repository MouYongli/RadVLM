from PIL import Image
import os
import pydicom
import numpy as np
from src.radvlm.utils.config import DATA_ORIGINAL_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR
from src.radvlm.utils.preprocess_images import copy_data_to_new_dir

if __name__ == "__main__":
    mimic_subset = os.path.join(DATA_ORIGINAL_DIR, "p10")
    copy_data_to_new_dir(mimic_subset, os.path.join(DATA_RAW_DIR, "p10"))