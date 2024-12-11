import os
import shutil
import argparse

parser = argparse.ArgumentParser(description="clean folders")
parser.add_argument(
    "--location",
    "-lo",
    type=str,
    help="gaolitong data or daojin data",
)

args = parser.parse_args()
location = args.location

if os.path.exists("metrics"):
    shutil.rmtree("metrics")
if os.path.exists("figs//" + location):
    shutil.rmtree("figs//" + location)
