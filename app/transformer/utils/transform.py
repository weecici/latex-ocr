import albumentations as alb
from albumentations.pytorch import ToTensorV2

transform = alb.Compose(
    [
        alb.PadIfNeeded(
            min_height=None, 
            min_width=None, 
            position="center",
            pad_height_divisor=16, 
            pad_width_divisor=16, 
            border_mode=0,
            fill=[255, 255, 255]
        ),
        alb.ToGray(p=1.0),
        alb.Normalize((0.9491, 0.9491, 0.9491), (0.1846, 0.1846, 0.1846)),
        ToTensorV2(),
    ]
)