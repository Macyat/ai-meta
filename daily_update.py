import subprocess
import argparse
import sys

parser = argparse.ArgumentParser(
    description="To daily update all models and get new outcomes"
)
parser.add_argument(
    "--compared_label",
    type=str,
    help="another element to be compared with",
)

args = parser.parse_args()
compared_label = args.compared_label

commands = [
    "black *.py",
    "python clean_metrics.py",
    "python run_futian.py --location gaolitong --select 0 --compared_label "
    + compared_label,
    "python run_futian.py --location gaolitong --select 1 --compared_label "
    + compared_label,
    "python run_futian.py --location daojin --select 0 --compared_label "
    + compared_label,
    "python run_futian.py --location daojin --select 1 --compared_label "
    + compared_label,
    "python run_lab.py --location gaolitong --compared_label " + compared_label,
    "python run_lab.py --location daojin --compared_label " + compared_label,
]

for command in commands:
    print(command)
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # print(process.stdout)
    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            sys.stdout.write(output)
            sys.stdout.flush()
