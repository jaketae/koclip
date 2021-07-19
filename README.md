# KoCLIP

This repository contains code for KoCLIP, a Korean port of OpenAI's CLIP. This project was conducted as part of Hugging Face's [Flax/JAX community week](https://github.com/huggingface/transformers/blob/master/examples/research_projects/jax-projects/README.md#quickstart-flax-and-jax) co-organized with Google's Flax, JAX, and Cloud teams ([announcement](https://discuss.huggingface.co/t/open-to-the-community-community-week-using-jax-flax-for-nlp-cv/7104)).

## Demo

Check out our Streamlit app [here](https://tinyurl.com/koclip-app). Please understand that the app might take some time to load.

## Quickstart

1. Import dependencies and initialize a KoCLIP model and a `CLIPProcessor`.

```python
import requests
import jax
from PIL import Image

from koclip.model import FlaxHybridCLIP
from transformers import CLIPProcessor

koclip = "koclip/koclip-base"
model = FlaxHybridCLIP.from_pretrained(koclip)
processor = CLIPProcessor.from_pretrained(koclip)
```

2. Prepare image and text captions.

```python
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = ["고양이들", "강아지 두 마리", "우산을 쓴 사람", "도시 풍경"]
image
```

3. Run inference.

```python
inputs = processor(
    text=text,
    images=image, 
    return_tensors="pt", 
    padding=True
)

outputs = model(**inputs)
probs = jax.nn.softmax(outputs.logits_per_image, axis=1)

for idx, prob in sorted(enumerate(*probs), key=lambda x: x[1], reverse=True):
    print(text[idx], prob)
```

## Models

On a high level, the KoCLIP pipeline is equivalent to that of the original CLIP model: it is composed of a transformer text encoder and an image encoder. Specific model configurations are listed below.

* `koclip-base` uses `klue/roberta` as its language model backbone and `openai/clip-vit-base-patch32` as its image encoder backbone. 
* `koclip-large` uses the same language model backbone, but uses a larger image encoder, `google/vit-large-patch16-224`.

## Training

KoCLIP was trained on 82783 images from the [MS COCO dataset](https://cocodataset.org/). Korean translations of caption annotations were obtained from [AI Hub](https://aihub.or.kr/keti_data_board/visual_intelligence). Each image comes with 5 possible ground-truth captions. KoCLIP was trained on a TPU3-v8 VM. Both text and image encoder backbones were loaded from their pretrained checkpoints. During fine-tuning, KoCLIP was trained to maximize the similarity score between matching pairs of images and captions.

## Team

* @ampehta
* @jaketae
* @tree-park
* @trent-dev

## Acknowledgement

The `FlaxHybridCLIP` model was adpated from the Hugging Face transformer repository, under [jax-projects](https://github.com/huggingface/transformers/tree/master/examples/research_projects/jax-projects/hybrid_clip).  We also express gratitude to the teams at Google for generously offering TPU VMs for this project. Last but not least, we thank the [KLUE team](https://github.com/KLUE-benchmark) for making pretrained Korean RoBERTa-large weights publicly available.

