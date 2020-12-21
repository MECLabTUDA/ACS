import zipfile
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default='.', help="input file")
parser.add_argument("--output", type=str, default='.', help="output path")
args = parser.parse_args()

with zipfile.ZipFile(args.input, "r") as z:
  z.extractall(args.output)
