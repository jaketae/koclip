import json
from typing import Callable, Optional

import torch
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize
from torchvision.transforms.functional import InterpolationMode


class Transform(torch.nn.Module):
    """
    returns transformed version of the input image
    >>> preprocess = Transform(config.vision_config.image_size)
    >>> preprocess = torch.jit.script(preprocess)
    """

    def __init__(self, image_size):
        super().__init__()
        self.transforms = torch.nn.Sequential(
            Resize([image_size], interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            ConvertImageDtype(torch.float),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transforms(x)


class ImageTextDataset(VisionDataset):
    """
    Dtaset for loading image-text data for tasks like CLIP training, Image Captioning.
    Args:
        root: (string): The root path where the dataset is stored
        file_path: (string): Path to the file containing the image_paths and associated captions.
            The expected format is jsonlines where each line is a json object containing to keys.
            `image_path`: The path to the image.
            `captions`: An `array` of captions.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        file_path: str,
        captions_per_image=5,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        with open(file_path, "r") as f:
            examples = json.load(f)

        self.captions = []
        self.image_paths = []

        for example in examples:
            captions = example["captions"][:captions_per_image]
            self.captions.extend(captions)
            self.image_paths.extend([example["file_path"]] * len(captions))

    def _load_image(self, idx: int):
        path = self.image_paths[idx]
        try:
            return read_image(path, mode=ImageReadMode.RGB)
        except:
            print(path)
            raise RuntimeError
       
        
    def _load_target(self, idx):
        return self.captions[idx]

    def __getitem__(self, index: int):
        image = self._load_image(index)
        target = self._load_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target

    def __len__(self) -> int:
        return len(self.captions)
