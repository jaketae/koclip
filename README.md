# KoCLIP

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tinyurl.com/koclip-app) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nPY78-vjBarhkYxHshM5gSSEGAm-BXsn?usp=sharing)

This repository contains code for KoCLIP, a Korean port of OpenAI's CLIP. This project was conducted as part of Hugging Face's [Flax/JAX community week](https://github.com/huggingface/transformers/blob/master/examples/research_projects/jax-projects/README.md#quickstart-flax-and-jax) co-organized with Google's Flax, JAX, and Cloud teams ([announcement](https://discuss.huggingface.co/t/open-to-the-community-community-week-using-jax-flax-for-nlp-cv/7104)).

## Demo

Check out our Streamlit app [here](https://tinyurl.com/koclip-app). The demo illustrates three potential uses cases of KoCLIP on different downstream tasks:

* *Image to Text*: This is essentially a zero-shot image classification task. Given an input image, the models finds the most likely caption among the text labels provided.
* *Text to Image*: This is essentially an image retrieval task. Given a text, the model looks up a database of pre-computed image embeddings to retrieve the image that best matches given text. 
* *Text to Patch*: This is also a variant of zero-shot image classification. Given a text and an image, the image is partitioned into subsections, and the model ranks them based on their relevance with the text query.

## Quickstart

To follow along the code snippets below, we recommend that you refer to the [Colab notebook](./inference.ipynb). 

1. Import dependencies and initialize a KoCLIP model along with its processor.

```python
import requests
import jax
from PIL import Image

from koclip import load_koclip

model, processor = load_koclip("koclip-base")
```

2. Prepare image and text captions.

```python
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = ["소파 위에 고양이", "강아지와 강아지 주인", "쳇바퀴를 달리는 햄스터", "자동차"]
image
```

3. Run inference.

```python
inputs = processor(
    text=text,
    images=image, 
    return_tensors="jax", # could also be "pt" 
    padding=True
)

outputs = model(**inputs)
probs = jax.nn.softmax(outputs.logits_per_image, axis=1)

for idx, prob in sorted(enumerate(*probs), key=lambda x: x[1], reverse=True):
    print(text[idx], prob)
```

## Models

We trained a total of two models, `koclip-base` and `koclip-large`. Both models use RoBERTa-large. The decision to use a somewhat large language model was motivated by the intuition that annotated Korean datasets are rare; a well-trained, performant LM would be key to good multimodal pipeline given limited data.

| KoCLIP         | LM                   | ViT                            |
| -------------- | -------------------- | ------------------------------ |
| `koclip-base`  | `klue/roberta-large` | `openai/clip-vit-base-patch32` |
| `koclip-large` | `klue/roberta-large` | `google/vit-large-patch16-224` |

## Training

KoCLIP was fine-tuned using 82,783 images from the [MSCOCO](https://cocodataset.org/#home) 2014 image captioning dataset. Korean translations of image captions were obtained from [AI Hub](https://aihub.or.kr/keti_data_board/visual_intelligence), an open database maintained by subsidiaries of the Korean Ministry of Science and ICT. Validation metrics were monitored using approximately 40,000 images from the validation set of the aforementioned dataset. 

KoCLIP was trained on a TPU3-v8 VM. Both text and image encoder backbones were loaded from their pretrained checkpoints. KoCLIP was trained to maximize the similarity score between matching pairs of images and captions.

## Findings

In this section, we detail some interesting findings we made throughout the project.

### Prompting

We found that KoCLIP performs better when prompting is used to induce zero-shot behavior. Namely, instead of feeding it a single word or short phrase, casting a template such as

```
이것은 {{}} 이다.
```

noticably helped the model produce more reliable results. We hypothesize that this is due to the nature of captions in the MSCOCO datset, which are most often full sentences, albeit sometimes short in length.  

### Multilinguality

Although KoCLIP was trained exclusively on a Korean dataset, we found that English queries also work surprisingly well for simple words (e.g. "dog", "car"). This could be one of two reasons, or a combination thereof:

* *ViT Pretraining*: The ViT backbone for `koclip-base`, `openai/clip-vit-base-patch32`, was already pretrained on an English dataset. Hence, it is possible that its embeddings still lie in a latent space where vector arithematic can be performed with English text embeddings. One reason against this hypothesis is that `koclip-large` also demonstrates similar multilingual behavior.

* *LM Knowledge Bleed*: `klue/roberta-large` was trained on a large corpus of Korean text in a self-supervised fashion. One might reasonably suspect that English words were included in parts of the corpus, especially given the high frequency of English word transliterations in contemporary conversational Korean. This might also explain why English queries work for both `koclip-base` and `koclip-large`. One reason against this hypothesis is that the authors of KLUE explicitly state in their paper that one criterion for text selection was that "the corpus must be written in contemporary Korean."

At the end of the day, we still found it intriguing that a model that was fine-tuned exclusively on Korean managed to produce semantic embeddings from English queries that work well with ViT.

## Team

* [Guijin Son](https://github.com/ampehta)
* [Hansol Park](https://github.com/tree-park)
* [Jake Tae](https://github.com/jaketae)
* [Trent Oh](https://github.com/trent-dev)

## Acknowledgement

The `FlaxHybridCLIP` model was adpated from the Hugging Face transformer repository, under [jax-projects](https://github.com/huggingface/transformers/tree/master/examples/research_projects/jax-projects/hybrid_clip).  We also express gratitude to the teams at Google for generously offering TPU VMs for this project. Last but not least, we thank the [KLUE team](https://github.com/KLUE-benchmark) for making pretrained Korean RoBERTa-large weights publicly available.

## References

```bibtex
@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation}, 
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jung-Woo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```bibtex
@misc{radford2021learning,
      title={Learning Transferable Visual Models From Natural Language Supervision}, 
      author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
      year={2021},
      eprint={2103.00020},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{lin2015microsoft,
      title={Microsoft COCO: Common Objects in Context}, 
      author={Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Dollár},
      year={2015},
      eprint={1405.0312},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{srinivasan2021wit,
      title={WIT: Wikipedia-based Image Text Dataset for Multimodal Multilingual Machine Learning}, 
      author={Krishna Srinivasan and Karthik Raman and Jiecao Chen and Michael Bendersky and Marc Najork},
      year={2021},
      eprint={2103.01913},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
