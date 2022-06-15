import copy

import numpy as np
import open3d
from open3d import *


class O3D_Draw(object):
    """docstring for O3D_Draw"""

    def __init__(self, ref_data, test_data):
        super(O3D_Draw, self).__init__()
        self.ref = open3d.registration.Feature()
        self.ref.data = ref_data.T

        self.test = open3d.registration.Feature()
        self.test.data = test_data.T

    def point_transfer(self, point_cloud_files, keypoints_files):
        # Load point cloud and extract the keypoints
        # print('zr-point_cloud_files:', point_cloud_files)
        # print('zr-keypoints_files:', keypoints_files)
        reference_pc = read_point_cloud(point_cloud_files[0])
        test_pc = read_point_cloud(point_cloud_files[1])
        # print('zr-reference_pc:', reference_pc) #geometry::PointCloud with 258342 points.

        indices_ref = np.genfromtxt(keypoints_files[0])
        indices_test = np.genfromtxt(keypoints_files[1])
        # print('zr-indices_ref:', indices_ref) #keypoints文件中的索引

        reference_pc_keypoints = np.asarray(reference_pc.points)[indices_ref.astype(int), :]
        test_pc_keypoints = np.asarray(test_pc.points)[indices_test.astype(int), :]
        print('zr-reference_pc.points:', reference_pc.points) #std::vector<Eigen::Vector3d> with 258342 elements.
        print('zr-reference_pc_keypoints:', reference_pc_keypoints)

        # Save ad open3d point clouds
        ref_key = open3d.geometry.PointCloud()
        ref_key.points = open3d.utility.Vector3dVector(reference_pc_keypoints)
        print('zr-ref_key:', ref_key) #geometry::PointCloud with 1000 points.
        print('zr-ref_key.points:', ref_key.points)

        test_key = open3d.geometry.PointCloud()
        test_key.points = open3d.utility.Vector3dVector(test_pc_keypoints)

        result_ransac = self.execute_global_registration(ref_key, test_key,
                                                         self.ref, self.test, 0.05)
        print('zr-result_ransca:', result_ransac)

        return reference_pc, test_pc, result_ransac

    def fig_draw(self, point_cloud_files, keypoints_files):
        reference_pc, test_pc, result_ransac = self.point_transfer(point_cloud_files, keypoints_files)
        # First plot the original state of the point clouds
        self.draw_registration_result(reference_pc, test_pc, np.identity(4))
        self.draw_registration_fragment_1(reference_pc, test_pc, np.identity(4))
        self.draw_registration_fragment_2(reference_pc, test_pc, np.identity(4))

        # Plot point clouds after registration
        # print('zr-reference_pc:', reference_pc)
        # print('zr-test_pc:', test_pc)
        # print('zr-result_ransac.transformation:', result_ransac.transformation)
        self.draw_registration_result(reference_pc, test_pc,
                                      result_ransac.transformation)

    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        draw_geometries([source_temp, target_temp])

    def draw_registration_fragment_1(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([255, 255, 255])
        source_temp.transform(transformation)
        draw_geometries([source_temp, target_temp])
    def draw_registration_fragment_2(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([255, 255, 255])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        draw_geometries([source_temp, target_temp])

    def execute_global_registration(self, source_down, target_down, reference_desc,
                                    target_desc, distance_threshold):
        result = registration_ransac_based_on_feature_matching(
            source_down, target_down, reference_desc, target_desc,
            distance_threshold,
            TransformationEstimationPointToPoint(False), 4,
            [CorrespondenceCheckerBasedOnEdgeLength(0.9),
             CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            RANSACConvergenceCriteria(4000000, 500))
        return result

    def refine_registration(self, source, target, source_fpfh, target_fpfh, voxel_size, result_ransac=None):
        distance_threshold = voxel_size * 0.4
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
        result = registration_icp(source, target, distance_threshold,
                                  result_ransac.transformation,
                                  TransformationEstimationPointToPlane())
        return result
