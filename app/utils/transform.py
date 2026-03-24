import albumentations as alb
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image


def smart_resize(
    gray_img: np.ndarray, target_height: int = 128, max_width: int = 1024
) -> np.ndarray:
    """
    Intelligently resizes an image while preserving aspect ratio,
    handling extreme edge cases for sequence modeling.
    """
    h, w = gray_img.shape[:2]

    # 1. Calculate new dimensions based on target height
    aspect_ratio = w / h
    new_h = target_height
    new_w = int(new_h * aspect_ratio)

    # 2. Edge Case: Width >> Height (Extremely long equations)
    if new_w > max_width:
        new_w = max_width
        new_h = int(new_w / aspect_ratio)  # Scale height down proportionally

    # 3. Edge Case: Width << Height (Tall matrices/column vectors)
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # 4. Adaptive Interpolation
    # INTER_AREA prevents thin lines from disappearing when shrinking.
    # INTER_CUBIC keeps edges crisp when enlarging.
    if new_h < h or new_w < w:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC

    resized = cv2.resize(gray_img, (new_w, new_h), interpolation=interp)

    # 5. Height Padding for Long Equations
    # If the image was scaled down to fit max_width, new_h is now less than target_height.
    # We pad the top and bottom with white so the neural network always sees a consistent height.
    if new_h < target_height:
        pad_top = (target_height - new_h) // 2
        pad_bottom = target_height - new_h - pad_top
        # Assuming grayscale input where 255 is white
        resized = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=255
        )

    return resized


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    gray = np.array(pil_image.convert("L"))

    # Crop non-white region using a simple intensity threshold
    # (good for clean formula images)
    mask = gray < 200
    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 8
        y0 = max(0, y - pad)
        x0 = max(0, x - pad)
        y1 = min(gray.shape[0], y + h + pad)
        x1 = min(gray.shape[1], x + w + pad)
        cropped = gray[y0:y1, x0:x1]
    else:
        cropped = gray

    h, w = cropped.shape
    resized = smart_resize(cropped)

    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return rgb


inference_transform = alb.Compose(
    [
        alb.PadIfNeeded(
            min_height=None,
            min_width=None,
            position="center",
            pad_height_divisor=16,
            pad_width_divisor=16,
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,
        ),
        # alb.ToGray(p=1.0),
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
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,
        ),
    ]
)
