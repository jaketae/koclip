from transformers import AutoTokenizer, ViTFeatureExtractor

from .model import FlaxHybridCLIP
from .processor import FlaxHybridCLIPProcessor


def load_koclip(model_name):
    assert model_name in {"koclip-base", "koclip-large"}
    model = FlaxHybridCLIP.from_pretrained(f"koclip/{model_name}")
    processor = FlaxHybridCLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    processor.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
    if model_name == "koclip/koclip-large":
        processor.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-large-patch16-224"
        )
    return model, processor
