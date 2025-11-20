#!/usr/bin/env python3.9

import sys
import shutil
import os
import pandas
import numpy
import torch
import matplotlib
import seaborn
import fcsparser
import FlowCal


# Check versions
print(f"Torch version {torch.__version__}")
print(f"Pandas version {pandas.__version__}")
print(f"Numpy version {numpy.__version__}")
print(f"Matplotlib version {matplotlib.__version__}")
print(f"Seaborn version {seaborn.__version__}")
print(f"fcsparser version {fcsparser.__version__}")
print(f"FlowCal version {FlowCal.__version__}")

def main():
    print("start of processing")
    src = os.environ['INPUT_DIR']
    dest = os.environ['OUTPUT_DIR']
    resources = os.environ['RESOURCES_DIR'] # static resources

    print("Command line arguments ...")
    print(sys.argv)
    print("ENV variables ...")
    print(os.environ)

    list_files(resources)

    # create a file static-file.txt in the resources directory and read it here
    if os.path.exists(f'{resources}/static-file.txt'):
        with open(f'{resources}/static-file.txt', "r") as file:
            content = file.read()

        print(content)

    shutil.copytree(src, dest, dirs_exist_ok=True)
    print("end of processing")

def list_files(startpath):
    for root, _, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

if __name__ == '__main__':
    main()