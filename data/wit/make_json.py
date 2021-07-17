import os
import csv
import json
import re
import random


over_100_count = 0
captions_len = []
def make_caption(captions):
    global over_100_count
    global captions_len
    result = ". ".join([caption for caption in captions if caption])
    if len(result.split()) > 100:
        over_100_count += 1
    captions_len.append(len(result.split()))
    return result


def main(file_path="wit_ko.csv"):
    examples = []
    file2ext = {}
    total = 0
    allowed_extensions = ['.jpeg', '.png']
    for file_ in os.listdir("img/fixed"):
        title, extension = os.path.splitext(file_)
        file2ext[title] = extension
    with open(file_path) as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            title = row[0]
            if title in file2ext:
                extension = file2ext[title]
            
                if extension not in allowed_extensions:
                    # print(extension)
                    continue
                file_ = f"{title}{extension}"
                captions = set()
                # 6th: caption_reference_description
                # 7th: caption_attribution_description
                for idx in [6, 7]:
                    if row[idx] and len(re.compile('[가-힣]').findall(row[idx])) > 5:
                        captions.add(row[idx])
                # 3th(page_title) + 15th(context_page_description)
                if row[12] == "true":
                    captions.add(make_caption([row[3], row[15]]))
                # 4th(section_title) + 5th('hierarchical_section_title') + 16th(context_page_description)
                captions.add(make_caption([row[4], row[5], row[16]]))
                if len(captions) == 0:
                    print("Skipping image w/o descriptions")
                    continue
                examples.append({
                    "file_path": f"/home/shared/dataset/wit/img/fixed/{file_}",
                    "captions": list(captions),
                })
                total += 1
                #print(captions)

    print(total)
    random.shuffle(examples)
    total_size = len(examples)
    train_size = int(total_size * 0.95)
    with open("train_annotations.json", "w") as f:
        json.dump(examples[:train_size], f)
    with open("valid_annotations.json", "w") as f:
        json.dump(examples[train_size:], f)
    

if __name__ == "__main__":
    main()
    print(over_100_count)
    print(sum(captions_len)/len(captions_len))
