
import csv
import glob
from typing import Text, List
import urllib.request
import requests
from multiprocessing import Pool
import socket
timeout = 10
socket.setdefaulttimeout(timeout)


DATA_PATH='/home/shared/dataset/wit'


def load_file(path):
    """
    load csv
    """
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t', quotechar='"')
        data = list(reader)
    return data


def extract_ko(data):
    """
    Extract lang=ko data samples
    """
    trainset = []
    for samp in data[1:]:
        if samp[0] != 'ko':
            continue
        trainset.append(samp)
    return trainset


def rewrite_wit(data_paths):
    """
    we need only korean set. extract only korean set.
    https://drive.google.com/file/d/1y_DxYrmUF4vw3m7UOlVsHSkcO_v0XuLv/view?usp=sharing

    """
    samples = []
    for path in data_paths:
        data = load_file(path)
        samples += extract_ko(data)
    return [[i, *samp] for i, samp in enumerate(samples)]

err_list = []
def req_imgs(url_info):
    """ download imgs """
    # request.get 요청
    global err_list
    try:
        response = requests.get(url_info[1], headers={'User-agent': 'your bot 0.1'})
        file_ext = url_info[1].split('.')[-1].lower()
        # jpeg and jpg are identical extensions, manually cast to jpeg
        if file_ext == "jpg":
            file_ext = "jpeg"
        # check file extensions
        # NOTE: torchvision can only read .png & .jpeg
        assert file_ext in {"png", "jpeg"}
        with open(f'{DATA_PATH}/img/retry/{url_info[0]}.{file_ext}', 'wb') as f:
            f.write(response.content)
    except Exception as e:
        err_list.append(url_info)
        print(f"{url_info}:  \n {e}")


def down_imgs(urls):
    with Pool(90) as p:
        p.map(req_imgs, urls)

if __name__ == '__main__':
    # path_list = glob.glob('/home/shared/dataset/wit')
    # path_list = glob.glob(f'{DATA_PATH}/info/*')

    # samples = rewrite_wit(path_list)

    # with open(f'{DATA_PATH}/wit_ko.csv', 'w') as f:
    #      writer = csv.writer(f, delimiter='\t', quotechar='"')
    #      writer.writerows(samples)
    samples = load_file(f'{DATA_PATH}/wit_ko.csv')

    url_list = [[samp[0], samp[3]] for samp in samples]
    down_imgs(url_list)

    with open(f'{DATA_PATH}/err_img_list.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t', quotechar='"')
        writer.writerows(err_list)
