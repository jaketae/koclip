import os
import shutil
from multiprocessing import Pool
from torchvision.io import read_image


def main():
    root = "/home/shared/dataset/wit/img/retry"
    corrupt = "/home/shared/dataset/wit/img/corrupt"
    for file_ in os.listdir(root):
        file_path = os.path.join(root, file_)
        if os.path.getsize(file_path) <= 1822:
        # try:
        #     read_image(file_path)
        # except RuntimeError:
        #     print("moving")
            shutil.move(file_path, os.path.join(corrupt, file_))


if __name__ == "__main__":
    # with Pool(10) as p:
    #     p.map(main())
    main()