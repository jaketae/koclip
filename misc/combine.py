import json


def main():
    for mode in ["train", "valid"]:
        annotations = []
        for dataset in ["coco", "wit"]:
            with open(f"{dataset}/{mode}_annotations.json", "r") as f:
                annotation = json.load(f)
            annotations.extend(annotation)
        with open(f"{mode}_annotations.json", "w") as f:
            json.dump(annotations, f)

if __name__ == "__main__":
    main()