from model import FlaxHybridCLIP
from processor import FlaxHybridCLIPProcessor
from transformers import AutoTokenizer, ViTFeatureExtractor


def load_koclip(model_name="koclip/koclip-base"):
    assert model_name in {"koclip/koclip-base", "koclip/koclip-large"}
    model = FlaxHybridCLIP.from_pretrained(model_name)
    processor = FlaxHybridCLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    processor.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
    if model_name == "koclip/koclip-large":
        processor.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-large-patch16-224"
        )
    return model, processor

