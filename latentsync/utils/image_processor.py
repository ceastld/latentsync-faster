from torchvision import transforms
from einops import rearrange
import torch
import numpy as np
from typing import Optional, Union
from latentsync.utils.timer import Timer
from PIL import Image
import torchvision.transforms.functional as TF

"""
If you are enlarging the image, you should prefer to use INTER_LINEAR or INTER_CUBIC interpolation. If you are shrinking the image, you should prefer to use INTER_AREA interpolation.
https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
"""


def load_fixed_mask(resolution: int) -> torch.Tensor:
    mask_image = Image.open("latentsync/utils/mask.png").convert("RGB")
    mask_image = mask_image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    mask_image = TF.to_tensor(mask_image)
    return mask_image


class ImageProcessor:
    def __init__(
        self,
        resolution: int = 512,
        device: str = "cpu",
        mask_image=None,
    ):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)

        if mask_image is None:
            self.mask_image = load_fixed_mask(resolution)
        else:
            self.mask_image = mask_image

        self.device = device

    def preprocess_image(self, image: torch.Tensor):
        image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * self.mask_image
        return pixel_values, masked_pixel_values, self.mask_image[0:1]

    @Timer()
    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "b h w c -> b c h w")

        results = [self.preprocess_image(image) for image in images]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return (
            torch.stack(pixel_values_list),
            torch.stack(masked_pixel_values_list),
            torch.stack(masks_list),
        )

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "b h w c -> b c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values
