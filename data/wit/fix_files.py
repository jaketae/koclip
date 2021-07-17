import glob
import itertools
from argparse import ArgumentParser
from joblib import Parallel, delayed
import os
import subprocess
from collections import Counter
import shutil


parser = ArgumentParser()
parser.add_argument("--in_dir", type=str, default="img/retry")
parser.add_argument("--out_dir", type=str, default="img/fixed")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

files = itertools.chain(
    # glob.iglob(f"{args.in_dir}/*.jpg"),
    glob.iglob(f"{args.in_dir}/*.jpeg", recursive=False),
    glob.iglob(f"{args.in_dir}/*.png", recursive=False),
    # glob.iglob(f"{args.in_dir}/*.svg"),
)

def process_file(path):
    basename = os.path.basename(path)
    ext = os.path.splitext(basename)[1]
    name = os.path.splitext(basename)[0]

    dirname = os.path.dirname(path)
    try:
        r = subprocess.run(
            f'convert {path} -resize "224^>" -colorspace RGB -density 1200 {args.out_dir}/{name}.jpeg',
            shell=True,
            timeout=10
        )
        rcode = r.returncode
    except subprocess.TimeoutExpired:
        print("conversion timeout expired")
        rcode = -1

    if rcode == 0:
        os.remove(path)

    return rcode

codes = Parallel(n_jobs=32, prefer="threads", verbose=1)(delayed(process_file)(f) for f in files)
print(Counter(codes))
