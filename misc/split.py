import json
import random


def main(file_path="annotations.json"):
    with open(file_path, "r") as f:
        examples = json.load(f)
    train = []
    valid = []
    for example in examples:
        example["caption_en"] = example["captions"]
        example["captions"] = example["caption_ko"]
        example["file_path"] = "/home/koclip_experiment/dataset/coco/" + example["file_path"]
        if "train2014" in example["file_path"]:
            train.append(example)
        else:
            valid.append(example)
    with open("train_annotations.json", "w") as f:
        json.dump(train, f)
    with open("valid_annotations.json", "w") as f:
        json.dump(valid, f)
    

if __name__ == "__main__":
    main()