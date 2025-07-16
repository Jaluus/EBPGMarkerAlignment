import argparse
from utils.etc_functions import detect_marker_center, compute_move_to_center
from utils.EPGP_functions import pg_image_grab, pg_move_position


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

    return parser.parse_args()


def main():
    """Main alignment function."""
    args = parse_arguments()

    STEP_SIZE = (args.step_size_x, args.step_size_y)
    RESOLUTION = (args.resolution_x, args.resolution_y)
    SAMPLE_AVERAGE_EXPONENT = args.sample_average_exponent
    FRAME_AVERAGE_EXPONENT = args.frame_average_exponent

    for i in range(args.iterations):
        ret, image, metadata = pg_image_grab(
            step=STEP_SIZE,
            num_steps=RESOLUTION,
            sample_average_exponent=SAMPLE_AVERAGE_EXPONENT,
            frame_average_exponent=FRAME_AVERAGE_EXPONENT,
        )

        marker_center = detect_marker_center(image)
        relative_move_um = compute_move_to_center(marker_center, metadata)

        pg_move_position(relative_move_um, relative=True)


if __name__ == "__main__":
    main()
