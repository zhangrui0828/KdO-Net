import sys

sys.path.append('../')
import open3d
import numpy as np
import time
import os
from tools import get_pcd, get_keypts, get_desc, loadlog
from sklearn.neighbors import KDTree

import time
import networkx as nx
inlier_ratio_threshold = 0.05
inlier_distance_threshold = 0.1
def calculate_M(source_desc, target_desc):
    """
    Find the mutually closest point pairs in feature space.
    source and target are descriptor for 2 point cloud key points. [5000, 32]
    """
    kdtree_s = KDTree(target_desc) 
    sourceNNdis, sourceNNidx = kdtree_s.query(source_desc, 1)
    kdtree_t = KDTree(source_desc)
    targetNNdis, targetNNidx = kdtree_t.query(target_desc, 1)
    result = []
    for i in range(len(sourceNNidx)):
        if targetNNidx[sourceNNidx[i]] == i:    
            result.append([i, sourceNNidx[i][0]])
    return np.array(result)  

def register2Fragments(id1, id2, keyptspath, descpath, resultpath, tMat, desc_name='KdO-Net'):
    cloud_bin_s = f'cloud_bin_{id1}'
    cloud_bin_t = f'cloud_bin_{id2}'
    write_file = f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'
    num_inliers = 0
    inlier_ratio = 0
    error_ratio = 0.0
    gt_flag = 0
    if os.path.exists(os.path.join(resultpath, write_file)):
        print(f"{write_file} already exists.")
        with open(os.path.join(resultpath, f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'), 'r') as f:
            content = f.readlines()
        nums = content[0].replace("\n", "").split("\t")[2:6]
        num_inliers = int(nums[0])
        inlier_ratio = float(nums[1])
        error_ratio = float(nums[2])
        gt_flag = int(nums[3])
        return num_inliers, inlier_ratio, error_ratio, gt_flag

    if is_D3Feat_keypts:  #main中定义，初值为false
        keypts_path = './D3Feat_contralo-54-pred/keypoints/' + keyptspath.split('/')[-2] + '/' + cloud_bin_s + '.npy'
        source_keypts = np.load(keypts_path)
        source_keypts = source_keypts[-num_keypoints:, :]
        keypts_path = './D3Feat_contralo-54-pred/keypoints/' + keyptspath.split('/')[-2] + '/' + cloud_bin_t + '.npy'
        target_keypts = np.load(keypts_path)
        target_keypts = target_keypts[-num_keypoints:, :]
        source_desc = get_desc(descpath, cloud_bin_s, desc_name=desc_name)
        target_desc = get_desc(descpath, cloud_bin_t, desc_name=desc_name)
        source_desc = np.nan_to_num(source_desc)
        target_desc = np.nan_to_num(target_desc)
        source_desc = source_desc[-num_keypoints:, :]
        target_desc = target_desc[-num_keypoints:, :]
    else:
        source_keypts = get_keypts(keyptspath, cloud_bin_s)
        target_keypts = get_keypts(keyptspath, cloud_bin_t)
        source_desc = get_desc(descpath, cloud_bin_s, desc_name=desc_name)
        target_desc = get_desc(descpath, cloud_bin_t, desc_name=desc_name)
        target_desc = np.nan_to_num(target_desc)
        if source_desc.shape[0] > num_keypoints:  
            rand_ind = np.random.choice(source_desc.shape[0], num_keypoints, replace=False)
            source_keypts = source_keypts[rand_ind]
            target_keypts = target_keypts[rand_ind]
            source_desc = source_desc[rand_ind]
            target_desc = target_desc[rand_ind]
    key = str(id1)+'_'+str(id2)

    if key not in gtLog.keys():
        corr = calculate_M(source_desc, target_desc)   
        if id2 in list(tMat[id1].keys()): 
            gtTrans_0 = np.identity(4)
            transPath = tMat[id1][id2]
            tLen = len(transPath) - 1
            for i in range(tLen):
                trg_i = transPath[tLen - i]
                src_i = transPath[tLen - i-1]
                theKey = str(src_i)+'_'+str(trg_i)
                tmpTrans = gtLog[theKey]
                gtTrans_0 = np.dot(gtTrans_0, tmpTrans)

            frag1_0 = source_keypts[corr[:, 0]]  
            frag2_pc_0 = open3d.geometry.PointCloud()
            frag2_pc_0.points = open3d.utility.Vector3dVector(target_keypts[corr[:, 1]])
            frag2_pc_0.transform(gtTrans_0)  
            frag2_0 = np.asarray(frag2_pc_0.points)

            tDistance = np.sqrt(np.sum(np.power(frag1_0 - frag2_0, 2), axis=1))
            tNum_inliers = np.sum(tDistance < inlier_distance_threshold)
            tNnlier_ratio = tNum_inliers / len(tDistance)
            error_ratio = tNnlier_ratio
    else:
        # find mutually closest point.
        corr = calculate_M(source_desc, target_desc) 


        gtTrans = gtLog[key]  
        frag1 = source_keypts[corr[:, 0]]   
        frag2_pc = open3d.geometry.PointCloud()
        frag2_pc.points = open3d.utility.Vector3dVector(target_keypts[corr[:, 1]])

        frag2_pc.transform(gtTrans) 
        frag2 = np.asarray(frag2_pc.points)
        distance = np.sqrt(np.sum(np.power(frag1 - frag2, 2), axis=1))
        num_inliers = np.sum(distance < inlier_distance_threshold)
        inlier_ratio = num_inliers / len(distance)
        gt_flag = 1

        # calculate the transformation matrix using RANSAC, this is for Registration Recall.
        source_pcd = open3d.geometry.PointCloud()
        source_pcd.points = open3d.utility.Vector3dVector(source_keypts)
        target_pcd = open3d.geometry.PointCloud()
        target_pcd.points = open3d.utility.Vector3dVector(target_keypts)
        s_desc = open3d.registration.Feature()
        s_desc.data = source_desc.T
        t_desc = open3d.registration.Feature()
        t_desc.data = target_desc.T


        # Another registration method
        corr_v = open3d.utility.Vector2iVector(corr)
        result = open3d.registration.registration_ransac_based_on_correspondence(
            source_pcd, target_pcd, corr_v,
            0.05,
            open3d.registration.TransformationEstimationPointToPoint(False), 3,
            open3d.registration.RANSACConvergenceCriteria(50000, 1000))

        # write the transformation matrix into .log file for evaluation.
        with open(os.path.join(logpath, f'{desc_name}_{timestr}.log'), 'a+') as f:
            trans = result.transformation
            trans = np.linalg.inv(trans)
            s1 = f'{id1}\t {id2}\t  37\n'
            f.write(s1)
            f.write(f"{trans[0, 0]}\t {trans[0, 1]}\t {trans[0, 2]}\t {trans[0, 3]}\t \n")
            f.write(f"{trans[1, 0]}\t {trans[1, 1]}\t {trans[1, 2]}\t {trans[1, 3]}\t \n")
            f.write(f"{trans[2, 0]}\t {trans[2, 1]}\t {trans[2, 2]}\t {trans[2, 3]}\t \n")
            f.write(f"{trans[3, 0]}\t {trans[3, 1]}\t {trans[3, 2]}\t {trans[3, 3]}\t \n")

    s = f"{cloud_bin_s}\t{cloud_bin_t}\t{num_inliers}\t{inlier_ratio:.8f}\t{error_ratio:.8f}\t{gt_flag}"
    with open(os.path.join(resultpath, f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'), 'w+') as f:
        f.write(s)

    return num_inliers, inlier_ratio, error_ratio, gt_flag


def read_register_result(id1, id2):
    cloud_bin_s = f'cloud_bin_{id1}'
    cloud_bin_t = f'cloud_bin_{id2}'
    with open(os.path.join(resultpath, f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'), 'r') as f:
        content = f.readlines()
    nums = content[0].replace("\n", "").split("\t")[2:6]
    return nums

def acceptMatch(tmpLog, figNum):
    tNodeList = []
    for i in range(figNum):
        for j in range(figNum):
            keyVal = str(i)+'_'+str(j)
            if keyVal in tmpLog.keys():
                tNodeList.append((i,j))
    G2 = nx.DiGraph()
    G2.add_edges_from(tNodeList)  
    tMat = nx.shortest_path(G2)   
    
    count = 0
    for i in range(figNum):
        if i not in list(tMat.keys()):
            continue
        for j in range(i+1, figNum):
            if j not in list(tMat[i].keys()):
                count = count + 1
    print('Num of no transformation matrix:', count)
 
    return tMat

if __name__ == '__main__':

    scene_list = [
        'kitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]

    desc_name = 'KdO-Net'
    timestr = time.strftime('%m%d%H%M')
    inliers_list = []
    recall_list = []
    precision_list = []
    inliers_ratio_list = []
    num_keypoints = 5000
    is_D3Feat_keypts = False
    for scene in scene_list:
        pcdpath = f"../data/3DMatch/{scene}/"
        interpath = f"../data/3DMatch/intermediate-files-real/{scene}/"  
        gtpath = f'../data/3DMatch/{scene}/'
        keyptspath = interpath 
        descpath = f"../data/3DMatch/{scene}/evaluate_output/32_dim/"   
        gtLog = loadlog(gtpath)  
        logpath = f"log_result/{scene}-evaluation"
        resultpath = os.path.join(".", f"pred_result/{scene}/{desc_name}_result_{timestr}") 
        if not os.path.exists(resultpath):
            os.makedirs(resultpath)
        if not os.path.exists(logpath):
            os.makedirs(logpath)

        # register each pair
        num_frag = len(os.listdir(pcdpath))
        print(f"Start Evaluate Descriptor {desc_name} for {scene}")
        start_time = time.time()

        tMat = acceptMatch(gtLog, num_frag)
        print("-----------------------------------------------")
        tRes = []
        for id1 in range(num_frag):
            if id1 not in tMat.keys():
                continue
            for id2 in range(id1 + 1, num_frag):
                if id2 not in list(tMat[id1].keys()):
                    continue
                num_inliers, inlier_ratio, error_ratio, gt_flag = register2Fragments(id1, id2, keyptspath, descpath, resultpath,
                                                                        tMat, desc_name)
                tRes.append([num_inliers, inlier_ratio, error_ratio, gt_flag])
        print(f"Finish Evaluation, time: {time.time() - start_time:.2f}s")
        # evaluate
        result = []
        for id1 in range(num_frag):
            if id1 not in tMat.keys():
                continue
            for id2 in range(id1 + 1, num_frag):
                if id2 not in list(tMat[id1].keys()):
                    continue
                line = read_register_result(id1, id2)   
                result.append([int(line[0]), float(line[1]), float(line[2]), int(line[3])])
        result = np.array(result)
        indices_results = np.sum(result[:, 3] == 1)
        correct_match = np.sum(result[:, 1] > inlier_ratio_threshold)
        fail_match = np.sum(result[:, 2] > inlier_ratio_threshold)
        recall = float(correct_match / indices_results) * 100
        precision =  float(correct_match / (correct_match+fail_match)) * 100
        recall = round(recall, 2)
        precision = round(precision, 2)
        print(recall, " ------------- ", precision)
        print(f"Correct Match {correct_match}, ground truth Match {indices_results}")
        print(f"Recall {recall}%")
        print(f"Correct Match {correct_match}, predicted Match {correct_match+fail_match}")
        print(f"Precision {precision}%")
        ave_num_inliers = np.sum(np.where(result[:, 1] > inlier_ratio_threshold, result[:, 0], np.zeros(result.shape[0]))) / correct_match
        ave_num_inliers = round(ave_num_inliers, 2)
        print(f"Average Num Inliners: {ave_num_inliers}")
        ave_inlier_ratio = np.sum(np.where(result[:, 1] > inlier_ratio_threshold, result[:, 1], np.zeros(result.shape[0]))) / correct_match
        ave_inlier_ratio = round(ave_inlier_ratio, 2)
        print(f"Average Num Inliner Ratio: {ave_inlier_ratio}")
        recall_list.append(recall)
        precision_list.append(precision)
        inliers_list.append(ave_num_inliers)
        inliers_ratio_list.append(ave_inlier_ratio)
    print('recall_list:', recall_list)
    print('precision_list:', precision_list)
    average_recall = sum(recall_list) / len(recall_list)
    average_precision = sum(precision_list)/len(precision_list)
    average_recall = round(average_recall, 2)
    average_precision = round(average_precision, 2)
    std_recall = np.std(recall_list)
    std_precision = np.std(precision_list)
    print(f"All 8 scene, average recall: {average_recall}%, std: {std_recall}")
    print(f"All 8 scene, average precision: {average_precision}%, std: {std_precision}")
    average_inliers = sum(inliers_list) / len(inliers_list)
    average_inliers = round(average_inliers, 2)
    print(f"All 8 scene, average num inliers: {average_inliers}")
    average_inliers_ratio = sum(inliers_ratio_list) / len(inliers_list)
    average_inliers_ratio = round(ave_inlier_ratio, 2)
    print(f"All 8 scene, average num inliers ratio: {average_inliers_ratio}")
