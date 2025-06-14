import os
import cv2
from utils.etc_functions import to_int8
from utils.EPGP_functions import pg_image_grab

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

ret, image, metadata = pg_image_grab(
    step=(64, 64),
    num_steps=(512, 512),
    sample_average_exponent=2,
    frame_average_exponent=2,
)

cv2.imwrite(os.path.join(OUT_DIR, "image_grab_test.png"), to_int8(image))
