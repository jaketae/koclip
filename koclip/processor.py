import jax.numpy as jnp
from transformers import CLIPProcessor


class FlaxHybridCLIPProcessor(CLIPProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        encoding = super()(text, images, return_tensors, **kwargs)
        # flax expects channels last
        if hasattr(encoding, "pixel_values"):
            encoding.pixel_values = jnp.transpose(
                encoding.pixel_values, axes=[0, 2, 3, 1]
            )
        return encoding