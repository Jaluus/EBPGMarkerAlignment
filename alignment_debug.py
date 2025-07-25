import cv2
import os
import argparse
from utils.etc_functions import (
    compute_move_to_center,
    detect_marker_candidates,
    preprocess_candidate,
    compute_descriptors,
    compute_score,
    compute_candidate_center,
)
from utils.EPGP_functions import pg_image_grab, pg_move_position
import numpy as np


def parse_arguments():
    """Parse command line arguments for alignment parameters."""
    parser = argparse.ArgumentParser(description="EBPG Marker Alignment Script")

    parser.add_argument(
        "--step-size-x",
        type=int,
        default=64,
        help="Step size in X direction (default: 64)",
    )
    parser.add_argument(
        "--step-size-y",
        type=int,
        default=64,
        help="Step size in Y direction (default: 64)",
    )
    parser.add_argument(
        "--resolution-x",
        type=int,
        default=512,
        help="Resolution in X direction (default: 512)",
    )
    parser.add_argument(
        "--resolution-y",
        type=int,
        default=512,
        help="Resolution in Y direction (default: 512)",
    )
    parser.add_argument(
        "--sample-average-exponent",
        type=int,
        default=2,
        help="Sample average exponent (default: 2)",
    )
    parser.add_argument(
        "--frame-average-exponent",
        type=int,
        default=2,
        help="Frame average exponent (default: 2)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of alignment iterations (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/pg/users/bsuo/alignment_images",
        help="Directory to save output images and moves (default: /home/pg/users/bsuo/alignment_images)",
    )

    return parser.parse_args()


def main():
    """Main alignment function."""
    args = parse_arguments()

    OUT_DIR = args.output_dir
    STEP_SIZE = (args.step_size_x, args.step_size_y)
    RESOLUTION = (args.resolution_x, args.resolution_y)
    SAMPLE_AVERAGE_EXPONENT = args.sample_average_exponent
    FRAME_AVERAGE_EXPONENT = args.frame_average_exponent

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    print("Starting EBPG Marker Alignment with the following parameters:")
    print(f"Step Size: {STEP_SIZE}")
    print(f"Resolution: {RESOLUTION}")
    print(f"Sample Average Exponent: {SAMPLE_AVERAGE_EXPONENT}")
    print(f"Frame Average Exponent: {FRAME_AVERAGE_EXPONENT}")
    print(f"Number of Iterations: {args.iterations}")

    for i in range(args.iterations):
        ret, image, metadata = pg_image_grab(
            step=STEP_SIZE,
            num_steps=RESOLUTION,
            sample_average_exponent=SAMPLE_AVERAGE_EXPONENT,
            frame_average_exponent=FRAME_AVERAGE_EXPONENT,
        )

        marker_candidates = detect_marker_candidates(image)
        image_with_candidates = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image_with_candidates, marker_candidates, -1, (0, 255, 0), 2)
        cv2.imwrite(
            os.path.join(OUT_DIR, f"{i}_marker_candidates.png"),
            image_with_candidates,
        )

        processed_marker_candidates = [
            preprocess_candidate(c) for c in marker_candidates
        ]
        image_with_candidates = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(
            image_with_candidates, processed_marker_candidates, -1, (0, 255, 0), 2
        )
        cv2.imwrite(
            os.path.join(OUT_DIR, f"{i}_processed_candidates.png"),
            image_with_candidates,
        )

        candidate_descriptors = [
            compute_descriptors(c) for c in processed_marker_candidates
        ]
        candidate_scores = [compute_score(d) for d in candidate_descriptors]

        best_candidate_index = np.argmax(candidate_scores)
        best_candidate = marker_candidates[best_candidate_index]
        best_score = candidate_scores[best_candidate_index]

        marker_center = compute_candidate_center(best_candidate)

        image_with_candidates = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image_with_candidates, [best_candidate], -1, (0, 255, 0), 2)
        cv2.circle(
            image_with_candidates, tuple(map(int, marker_center)), 5, (0, 0, 255), -1
        )
        cv2.putText(
            image_with_candidates,
            f"Score: {best_score:.2f}",
            tuple(map(int, marker_center)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.imwrite(
            os.path.join(OUT_DIR, f"{i}_best_candidate.png"),
            image_with_candidates,
        )

        # marker_center = detect_marker_center(image)
        relative_move_um = compute_move_to_center(marker_center, metadata)

        print(f"Current Iteration: {i + 1} of {args.iterations}")
        print(f"Found Marker Center at: {marker_center} Pixel Coordinates")
        print(f"Moving by: {relative_move_um} microns")

        pg_move_position(relative_move_um, relative=True)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.circle(image, tuple(map(int, marker_center)), 5, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(OUT_DIR, f"{i}_iteration_alignment.png"), image)


if __name__ == "__main__":
    main()
