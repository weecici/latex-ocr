import albumentations as alb
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Preprocess an image to match the im2latex-100k dataset format.

    Steps:
        1. Convert to grayscale
        2. Binarize with Otsu's threshold (clean black-on-white)
        3. Crop whitespace (find bounding box of non-white content + padding)
        4. Convert back to 3-channel (RGB) for model input compatibility

    Args:
        pil_image: Input PIL Image (any mode).

    Returns:
        Preprocessed image as a numpy array (H, W, 3) in uint8.
    """
    # 1. Convert to grayscale numpy
    gray = np.array(pil_image.convert("L"))

    # 2. Binarize with Otsu's threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Crop whitespace — find bounding box of non-white pixels
    coords = cv2.findNonZero(255 - binary)  # invert: non-white becomes non-zero
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 8  # small padding around the formula
        y0 = max(0, y - pad)
        x0 = max(0, x - pad)
        y1 = min(binary.shape[0], y + h + pad)
        x1 = min(binary.shape[1], x + w + pad)
        cropped = binary[y0:y1, x0:x1]
    else:
        # Entirely white image — keep as-is
        cropped = binary

    # 4. Convert single-channel to 3-channel RGB
    rgb = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
    return rgb

inference_transform = alb.Compose(
    [
        alb.PadIfNeeded(
            min_height=None,
            min_width=None,
            position="center",
            pad_height_divisor=16,
            pad_width_divisor=16,
            border_mode=0,
            fill=[255, 255, 255],
        ),
        alb.ToGray(p=1.0),
        alb.Normalize((0.9491, 0.9491, 0.9491), (0.1846, 0.1846, 0.1846)),
        ToTensorV2(),
    ]
)

visual_transform = alb.Compose(
    [
        alb.PadIfNeeded(
            min_height=None,
            min_width=None,
            position="center",
            pad_height_divisor=16,
            pad_width_divisor=16,
            border_mode=0,
            fill=[255, 255, 255],
        ),
    ]
)
