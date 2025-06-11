import subprocess
import os
from typing import Optional, Tuple
import numpy as np
from .etc_functions import load_image_and_metadata


def pg_image_grab(
    step: Tuple[int, int],
    num_steps: Tuple[int, int],
    identifier: str,
    offset: Tuple[int, int] = (0, 0),
    reference: Optional[str] = None,
    sample_average_exponent: int = 0,
    frame_average_exponent: int = 0,
    sub: Optional[Tuple[int, int]] = None,
    meander: bool = False,
    vertical: bool = False,
    nomove: bool = False,
    fast: bool = False,
) -> tuple[subprocess.CompletedProcess, np.ndarray, dict]:
    """
    Execute the pg image grab command to capture an image using pattern generator and video circuitry.

    This function captures images by controlling the electron beam deflection system. The image is stored
    as a raw image file in the subdirectory specified by the $PG_IMAGES environment variable. Two files
    are created:
    - <identifier>.img: Contains raw 16-bit unsigned data (little-endian)
    - <identifier>.txt: Contains image metadata and parameters

    The captured raw images can be viewed using ImageJ with "16 bit Unsigned" and "Little-endian Byte order"
    settings, or directly using the ijraw script.

    Args:

        step: Tuple of (step_x, step_y) in main or sub resolution units.
            Can be zero, negative, or positive. Defines the step size between measurement points.
            Units depend on deflection mode: main DAC units for main deflection, sub DAC units for sub deflection.

        num_steps: Tuple of (num_x, num_y) - number of steps in each direction.
            Must be positive integers. Determines the image resolution (pixels).
            Size constraints:
            - Without frame averaging: num_x * num_y < (2048 * 2048) / 2 = 2,097,152
            - With frame averaging: num_x * num_y < (2048 * 2048) / 4 = 1,048,576

        identifier: String identifier used as the base filename for output files.
            Should not include file extension as .img and .txt are added automatically.

        offset: Optional Tuple of (offset_x, offset_y) in main resolution units (default is (0, 0)).
            Positive and negative values allowed. Specifies the offset of the image center
            with respect to the center of the main deflection field (or sub deflection field if --sub used).

        reference: Optional reference identifier for cross-correlation reference image.
            Used for image alignment and drift correction purposes.

        sample_average_exponent: Number of sample averaging (default=0).
            Number of samples averaged = 2^sample_average_exponent.
            Higher values improve signal-to-noise ratio but increase acquisition time.

        frame_average_exponent: Number of frame averaging (default=0).
            Number of frames averaged = 2^frame_average_exponent.
            Reduces image size limits when > 0. Improves image quality but slows acquisition.

        sub: Optional tuple of (main_defl_x, main_defl_y) for sub deflection scanning mode.
            When specified, uses sub (trap) deflection for scanning while keeping main deflection
            fixed at the provided coordinates. All step/offset units become sub DAC units.

        meander: If True, scans in meandering pattern instead of raster order.
            Meandering can be faster but may affect image quality. Not compatible with sub deflection.

        vertical: If True, scans vertically instead of horizontally.
            Only applies to main deflection mode. Not compatible with sub deflection.

        nomove: If True, keeps deflection at offset position but adds settling waits.
            Useful for measuring noise without actual scanning motion.

        fast: If True, enables fast scanning mode (~5-10x faster).
            Trades speed for accuracy (several pixels of x-axis shift possible).
            Can only be combined with frame_average_exponent > 0.

    Returns:
        subprocess.CompletedProcess: Result of the command execution containing:
            - returncode: 0 if successful
            - stdout: Command output text
            - stderr: Error messages if any
        np.ndarray: Loaded image data as a NumPy array (16-bit unsigned integers).
        dict: Parsed metadata from the associated .txt file.

    Raises:
        ValueError: If parameters violate constraints:
            - num_x or num_y <= 0
            - num_x * num_y exceeds size limits based on frame averaging
        FileNotFoundError: If 'pg' command is not found in system PATH
        subprocess.CalledProcessError: If the pg command execution fails

    Notes:
        - Image grabbing requires settling time controlled by parameters:
          imagembsbase, imagembsfactor (main deflection)
          imagesbsbase, imagesbsfactor (sub deflection when --sub used)
        - Lower settling values lead to less accurate positioning
        - All coordinate values are in DAC units (Digital-to-Analog Converter units)
        - The function validates input constraints before execution
    """

    # Validate inputs
    if num_steps[0] <= 0 or num_steps[1] <= 0:
        raise ValueError("num_x and num_y must be positive")

    # Check size limits based on frame averaging
    max_pixels = (2048 * 2048) // (4 if frame_average_exponent > 0 else 2)
    if num_steps[0] * num_steps[1] >= max_pixels:
        raise ValueError(f"num_x * num_y must be < {max_pixels}")

    # Build command
    cmd = ["pg", "image", "grab"]

    # Add required parameters
    cmd.append(f"{offset[0]},{offset[1]}")
    cmd.append(f"{step[0]},{step[1]}")
    cmd.append(f"{num_steps[0]},{num_steps[1]}")
    cmd.append(identifier)

    # Add optional reference
    if reference:
        cmd.append(reference)

    # Add options
    if sample_average_exponent > 0:
        cmd.append(f"--sample_average_exponent={sample_average_exponent}")

    if frame_average_exponent > 0:
        cmd.append(f"--frame_average_exponent={frame_average_exponent}")

    if sub:
        cmd.extend(["--sub", f"{sub[0]},{sub[1]}"])

    if meander:
        cmd.append("--meander")

    if vertical:
        cmd.append("--vertical")

    if nomove:
        cmd.append("--nomove")

    if fast:
        cmd.append("--fast")

    # Execute command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        raise
    except FileNotFoundError:
        raise FileNotFoundError("pg command not found. Make sure it's in your PATH.")

    # Load the captured image
    image_path = os.path.join(os.environ.get("PG_IMAGES", "."), f"{identifier}.img")
    metadata_path = os.path.join(os.environ.get("PG_IMAGES", "."), f"{identifier}.txt")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} does not exist.")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file {metadata_path} does not exist.")

    try:
        image, metadata = load_image_and_metadata(identifier_path=identifier)
    except Exception as e:
        print(f"Error loading image or metadata: {e}")
        raise

    return result, image, metadata


def pg_move_position(
    position: Tuple[float, float],
    relative: bool = False,
) -> subprocess.CompletedProcess:
    """
    Execute the pg move position command to move the table to a specified position.
    This function moves the table to the specified position in microns. If the system has a Z-stage,
    the Z position will be adjusted based on the last found calibration marker position, cup position,
    or substrate height.
    Args:
        position: Tuple of (pos_x, pos_y) in microns.
            Specifies the target position to move the table to.
        relative: If True, moves relative to the current position instead of absolute.
    Returns:
        subprocess.CompletedProcess: Result of the command execution containing:
            - returncode: 0 if successful
            - stdout: Command output text
            - stderr: Error messages if any
    Raises:
        ValueError: If position values are not valid numbers.
        FileNotFoundError: If 'pg' command is not found in system PATH.
        subprocess.CalledProcessError: If the pg command execution fails.
    """
    # Validate inputs
    if not all(isinstance(coord, (int, float)) for coord in position):
        raise ValueError("Position coordinates must be numbers (int or float).")

    # Build command
    cmd = ["pg", "move", "position"]

    if relative:
        cmd.append("--relative")

    cmd.append(f"{position[0]},{position[1]}")

    # Execute command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        raise
    except FileNotFoundError:
        raise FileNotFoundError("pg command not found. Make sure it's in your PATH.")

    return result


def pg_get_table_position(measure: bool = True) -> Tuple[float, float]:
    """
    Execute the pg get table position command to retrieve the current table position.

    Returns:
        Tuple[float, float]: Current table position as (pos_x, pos_y) in microns.

    Raises:
        FileNotFoundError: If 'pg' command is not found in system PATH.
        subprocess.CalledProcessError: If the pg command execution fails.
    """
    cmd = ["pg", "get", "table_position"]
    if measure:
        cmd.append("--measure")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        pos_x, pos_y = map(float, result.stdout.strip().split(","))
        return pos_x, pos_y
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        raise
    except FileNotFoundError:
        raise FileNotFoundError("pg command not found. Make sure it's in your PATH.")
