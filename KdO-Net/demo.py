
import subprocess

from config.config import DATA_PATH
from core import run_cnn
from core.fig_draw import *
from open3d import *

DATA_PATH = "../data/"
PCL_EXE_PATH = "../PCL_EXE/build/KdO-Net"
DEMO_DATA_PATH = DATA_PATH + "demo/"
NPZ_SUFFIX = "_0.150000_16_1.desc.KdO-Net.bin.npy"

# Run the input parametrization
point_cloud_files = [DEMO_DATA_PATH + "cloud_bin_0.ply", DEMO_DATA_PATH + "cloud_bin_1.ply"]
keypoints_files = [DEMO_DATA_PATH + "cloud_bin_0_keypoints.txt", DEMO_DATA_PATH + "cloud_bin_1_keypoints.txt"]

if __name__ == '__main__':
    for i in range(0, len(point_cloud_files)):
        args = PCL_EXE_PATH + " -f " + point_cloud_files[i] + " -k " \
               + keypoints_files[i] + " -o " + DEMO_DATA_PATH + "sdv/"
        subprocess.call(args, shell=True)
    print('Input parametrization complete. Start inference')

    # Run the inference as shell
    run_cnn.to_network([
        "--run_mode=test",
        "--evaluate_input_folder=" + DEMO_DATA_PATH + "sdv/",
        "--evaluate_output_folder=" + DEMO_DATA_PATH
    ])

    print('Inference completed perform nearest neighbor search and registration')

    # Load the descriptors and estimate the transformation parameters using RANSAC
    reference_desc = np.load(DEMO_DATA_PATH + '32_dim/cloud_bin_0.ply' + NPZ_SUFFIX)
    test_desc = np.load(DEMO_DATA_PATH + '32_dim/cloud_bin_1.ply' + NPZ_SUFFIX)
    ref_data = reference_desc
    test_data = test_desc

    print("reference_desc", reference_desc)
    print("len(ref_data):",len(ref_data))
    print("column of ref_data:", len(ref_data[0]))

    toDraw = O3D_Draw(ref_data, test_data)
    toDraw.fig_draw(point_cloud_files, keypoints_files)
