import jax.numpy as jnp
from transformers import CLIPProcessor


class FlaxHybridCLIPProcessor(CLIPProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        encoding = super().__call__(text, images, return_tensors, **kwargs)
        # flax expects channels last
        if "pixel_values" in encoding.keys():
            encoding["pixel_values"] = jnp.transpose(
                encoding["pixel_values"], axes=[0, 2, 3, 1]
            )
        return encoding
