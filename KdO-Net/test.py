# Code:
# -*- coding: utf-8 -*-
from core import run_cnn
import glob
import tensorflow as tf
from config.config import DATA_PATH


dirPath = glob.iglob("./data/3DMatch/")


dataPath = "./data/3DMatch/"

filePath = [dataPath + "sun3d-hotel_uc-scan3/", dataPath + "sun3d-hotel_umd-maryland_hotel1/", \
            dataPath + "kitchen/", dataPath + "sun3d-home_md-home_md_scan9_2012_sep_30/", \
            dataPath + "sun3d-mit_76_studyroom-76-1studyroom2/", dataPath + "sun3d-hotel_umd-maryland_hotel3/", \
            dataPath + "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika/", dataPath + "sun3d-home_at-home_at_scan1_2013_jan_1/"]


if __name__ == '__main__':


    for i in range(0, len(filePath)):
        run_cnn.to_network([
            "--run_mode=test",
            "--evaluate_input_folder=" + filePath[i] + "sdv/",
            "--evaluate_output_folder=" + filePath[i] + "evaluate_output"
        ])
        print('filePath {} Inference completed'.format(filePath[i]))

        tf.reset_default_graph()

