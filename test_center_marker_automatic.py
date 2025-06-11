import os
import cv2
from utils.etc_functions import detect_marker_center, compute_move_to_center, to_int8
from utils.EPGP_functions import pg_image_grab, pg_move_position

OUT_DIR = os.path.join(os.path.dirname(__file__), "output", "center_marker_automatic")
os.makedirs(OUT_DIR, exist_ok=True)

moves = []
for i in range(10):
    ret, image, metadata = pg_image_grab(
        step=(32, 32),
        num_steps=(512, 512),
        sample_average_exponent=2,
        frame_average_exponent=2,
    )

    marker_center = detect_marker_center(image)
    relative_move_um = compute_move_to_center(marker_center, metadata)

    print(f"Marker center: {marker_center}")
    print(f"Relative move to center: {relative_move_um}")

    pg_move_position(relative_move_um, relative=True)

    moves.append(relative_move_um)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.circle(image, tuple(marker_center), 5, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(OUT_DIR, f"{i}_image_with_marker.png"), to_int8(image))

# Save the moves to a file
with open(os.path.join(OUT_DIR, "moves.txt"), "w") as f:
    for move in moves:
        f.write(f"{move}\n")
