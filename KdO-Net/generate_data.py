# -*- coding: utf-8 -*-
# Code:
import glob
import subprocess

from config.config import DATA_PATH
from core import run_cnn
import glob
import os
import time
import numpy as np

PCL_EXE_PATH = "../PCL_EXE/build/KdO-Net"

dirPath = glob.iglob("./data/ETH/")


if __name__ == '__main__':
    for docum_path in dirPath:

        files = os.listdir(docum_path)
        for file in files:
            filePath = "./data/test/" + file
            keypointsPath = filePath + "/01_Keypoints/"
            file = os.listdir(filePath)
            keyfile = os.listdir(keypointsPath)

            for f in file:
                for k in keyfile:
                    if f.endswith(".ply") and (f[:-4]==k[:-13]):  

                        args = PCL_EXE_PATH + " -f " + filePath + "/" + f + " -k " \
                               + keypointsPath + k + " -o " + filePath + "/sdv/"
                        print(args)
                        subprocess.call(args, shell=True)

    print('Got .csv files')
