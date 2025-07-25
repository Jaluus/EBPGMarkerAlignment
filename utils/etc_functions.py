import cv2
import numpy as np
import re
import os


def detect_marker_candidates(
    image: np.ndarray,
    median_blur_ksize: int = 3,
    gaussian_blur_ksize: int = 5,
    size_threshold: int = 1000,
) -> list[np.ndarray]:

    image_int8 = to_int8(image)

    image_processed = cv2.medianBlur(image_int8, median_blur_ksize)
    image_processed = cv2.GaussianBlur(
        image_processed,
        (gaussian_blur_ksize, gaussian_blur_ksize),
        0,
    )
    _, image_thresh = cv2.threshold(
        image_processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        image_thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = [c for c in contours if cv2.contourArea(c) > size_threshold]

    return contours


def to_int8(image: np.ndarray) -> np.ndarray:
    """Convert image to uint8 type."""
    if image.dtype != np.uint8:
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype(np.uint8)
    return image


def preprocess_candidate(
    candidate: np.ndarray,
    simplify: bool = True,
) -> np.ndarray:
    if simplify:
        # simplify the contour to reduce noise
        epsilon = 0.001 * cv2.arcLength(candidate, True)
        candidate = cv2.approxPolyDP(candidate, epsilon, True)

    min_area_rect = cv2.minAreaRect(candidate)
    box_points = cv2.boxPoints(min_area_rect).astype(np.int32)
    return box_points.reshape((-1, 1, 2))


def compute_descriptors(candidate: np.ndarray) -> dict[str, float]:

    # Compute the minimum area rectangle for the contour to compare it to a box
    min_area_rect = cv2.minAreaRect(candidate)
    box_points = cv2.boxPoints(min_area_rect).astype(np.int32)

    # compute the area ratio of the contour to the box
    # rectangle box has area ratio of 1.0
    contour_area = cv2.contourArea(candidate)
    box_area = cv2.contourArea(box_points)
    area_ratio = contour_area / (box_area + 1e-6)

    # compute the length ratio of the box edges
    # Square box has length ratio of 1.0
    box_long_edge = max(min_area_rect[1])
    box_short_edge = min(min_area_rect[1])
    edge_ratio = box_short_edge / (box_long_edge + 1e-6)

    return {
        "area_ratio": area_ratio,
        "edge_ratio": edge_ratio,
    }


def compute_score(descriptor: dict[str, float]) -> float:
    """Compute a score based on the descriptor values."""
    area_ratio = descriptor["area_ratio"]
    edge_ratio = descriptor["edge_ratio"]

    # Adjust weights as needed
    area_weight = 0.5
    edge_weight = 0.5

    score = (area_weight * area_ratio) + (edge_weight * edge_ratio)

    return score


def compute_candidate_center(candidate: np.ndarray) -> tuple[float, float]:
    """Get the center of the candidate contour."""
    M = cv2.moments(candidate)
    if M["m00"] == 0:
        return (0.0, 0.0)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy)


def load_image(
    image_path: str,
    image_width: int = 256,
    image_height: int = 256,
    image_dtype: np.dtype = np.uint16,
) -> np.ndarray:
    """Load an image from the specified path."""
    try:
        # Read the binary data from the file
        raw_data = np.fromfile(image_path, dtype=image_dtype)

        # Check if the amount of data read matches the expected size
        expected_pixels = image_width * image_height
        if raw_data.size != expected_pixels:
            raise ValueError(
                f"File size mismatch: expected {expected_pixels} pixels, got {raw_data.size}"
            )
        else:
            # Reshape the 1D array into a 2D image array
            image_data = raw_data.reshape((image_height, image_width))

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {image_path}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the image: {e}")

    return image_data


def load_image_and_metadata(identifier_path: str) -> tuple[np.ndarray, dict[str, any]]:
    """Load an image and its metadata from the specified path."""
    image_path = identifier_path + ".img"
    meta_path = identifier_path + ".txt"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} does not exist.")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file {meta_path} does not exist.")

    meta = parse_metadata(meta_path)
    image_width = meta["Image Size [exels]"][0]
    image_height = meta["Image Size [exels]"][1]

    image = load_image(
        image_path,
        image_width=image_width,
        image_height=image_height,
    )

    return image, meta


def parse_metadata(filepath):
    metadata = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                # Split only on the first colon
                key_part, value_part = line.split(":", 1)
                key_part = key_part.strip()
                key = re.sub(r"\s+", " ", key_part)

                value_str = value_part.strip()

                if "," in value_str:
                    try:
                        # Try converting to integers first
                        values = tuple(int(v.strip()) for v in value_str.split(","))
                        metadata[key] = values
                    except ValueError:
                        try:
                            # If integer conversion fails, try converting to floats
                            values = tuple(
                                float(v.strip()) for v in value_str.split(",")
                            )
                            metadata[key] = values
                        except ValueError:
                            # If both fail, store as string (or handle error as needed)
                            metadata[key] = value_str  # Fallback to string
                else:
                    try:
                        # Try converting to integer
                        metadata[key] = int(value_str)
                    except ValueError:
                        # If not an integer, store as string
                        metadata[key] = value_str
    return metadata


def detect_marker_center(image: np.ndarray) -> tuple[float, float]:
    """Detect the center of the marker in the image."""

    if image.dtype != np.uint8:
        image = to_int8(image)

    marker_candidates = detect_marker_candidates(image)
    marker_candidates = [preprocess_candidate(c) for c in marker_candidates]
    candidate_scores = [
        compute_score(compute_descriptors(c)) for c in marker_candidates
    ]

    best_candidate_index = np.argmax(candidate_scores)
    best_candidate = marker_candidates[best_candidate_index]

    return compute_candidate_center(best_candidate)


def compute_move_to_center(
    marker_center: tuple[float, float],
    metadata: dict,
) -> tuple[float, float]:
    """
    Compute the relative move to center the marker in the image.

    Args:
        marker_center (tuple): The (x, y) coordinates of the marker center in pixels.
        metadata (dict): Metadata containing image size and pixel size.

    Returns:
        tuple: Relative move in x and y directions in microns.
    """
    pixel_size_nm = metadata["Exel Size [nm]"]
    image_size_pixels = metadata["Image Size [exels]"]

    pixel_size_x_nm = pixel_size_nm[0]
    pixel_size_y_nm = pixel_size_nm[1]

    image_size_x_pixels = image_size_pixels[0]
    image_size_y_pixels = image_size_pixels[1]

    center_x_pixels = image_size_x_pixels / 2
    center_y_pixels = image_size_y_pixels / 2

    marker_center_x_pixels = marker_center[0]
    marker_center_y_pixels = marker_center[1]

    relative_move_x_pixels = center_x_pixels - marker_center_x_pixels
    relative_move_y_pixels = center_y_pixels - marker_center_y_pixels

    relative_move_x_um = relative_move_x_pixels * pixel_size_x_nm / 1000
    relative_move_y_um = relative_move_y_pixels * pixel_size_y_nm / 1000

    return -relative_move_x_um, relative_move_y_um
