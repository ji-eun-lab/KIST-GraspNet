import os
import sys
import argparse
import numpy as np
import time
import glob
import cv2

import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

import rospy
import message_filters
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
import ros_numpy as rnp
from PIL import Image as PILImage

import copy


class DataROS:
    def __init__(self, FLAGS, global_config):
        rospy.init_node('data_ros_node') 
        rgb_sub     = message_filters.Subscriber('/rgb', Image)
        depth_sub   = message_filters.Subscriber('/depth', Image)
        pcl_sub     = message_filters.Subscriber('/depth_pcl', PointCloud2)

        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, pcl_sub], 10, 0.1, allow_headerless=True)
        
        self.det_res_sub = rospy.Subscriber('/det_res',Detection2DArray, self.callback_sub)
        
        self.modified_mask = np.zeros((720, 1280))

        self.global_config = global_config
        self.flags = FLAGS
        self.det_dish = None
        self.sess = None
        
        self.grasp_estimator = self.init_network()
        self.pred_grasps_cam2 = {}
        
    def run(self):
        self.ts.registerCallback(self.callback)
        rospy.spin()
        
    def init_network(self):
        checkpoint_dir = self.flags.ckpt_dir
    
        # Build the model
        grasp_estimator = GraspEstimator(self.global_config)
        grasp_estimator.build_network()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        self.sess = sess

        # Load weights
        grasp_estimator.load_weights(self.sess, saver, checkpoint_dir, mode='test')

        os.makedirs('results', exist_ok=True)
        
        return grasp_estimator

    def callback_sub(self, data):
        self.det_dish = data
        #test = self.det_dish.detections[0].source_img
        num_obj = len(self.det_dish.detections)
        # print(num_obj)
        
        self.modified_mask = np.zeros((720, 1280))
        
        # mask list    
        for i in range(num_obj):
            mask_img = rnp.numpify(self.det_dish.detections[i].source_img)
            self.modified_mask += mask_img*(i+1)
            
        self.det_res_sub.unregister()
    
    

    

    def callback(self, rgb_msg, depth_msg, pcl_msg):
        rgb_img = rnp.numpify(rgb_msg)
        depth_img = rnp.numpify(depth_msg)
        pcd = rnp.numpify(pcl_msg)
        pcd = rnp.point_cloud2.get_xyz_points(pcd, remove_nans=False).reshape(-1,3)
        # mask_info = self.det_dish.detections[0].source_img
        # pcd = rnp.point_cloud2.get_xyz_points(pcd).reshape(-1,3)

        z_range = eval(str(self.flags.z_range))
        K = self.flags.K
        local_regions=self.flags.local_regions
        filter_grasps=self.flags.filter_grasps
        segmap_id=self.flags.segmap_id
        forward_passes=self.flags.forward_passes
        skip_border_objects=self.flags.skip_border_objects

        """
        Predict 6-DoF grasp distribution for given model and input data
        
        :param global_config: config.yaml from checkpoint directory
        :param checkpoint_dir: checkpoint directory
        :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
        :param K: Camera Matrix with intrinsics to convert depth to point cloud
        :param local_regions: Crop 3D local regions around given segments. 
        :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
        :param filter_grasps: Filter and assign grasp contacts according to segmap.
        :param segmap_id: only return grasps from specified segmap_id.
        :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
        :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
        """

        pc_segments = {}
        all_pc_segments = np.empty((0,3))
        if rgb_img is not None and depth_img is not None and pcd is not None:
            print('Converting depth to point cloud(s)...')
            
            cam_K = np.array(K).reshape(3, 3)
            
            pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds_wtih_full_pcd(pcd, segmap= self.modified_mask, rgb=rgb_img, skip_border_objects=skip_border_objects, z_range=z_range)
            
            # pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds(depth = depth_img, K = cam_K, segmap= self.modified_mask, rgb=rgb_img, skip_border_objects=skip_border_objects, z_range=z_range)
            
            print('Generating Grasps...')
            pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(self.sess, 
                                                                                                pc_full=pc_full, 
                                                                                                pc_segments=pc_segments, 
                                                                                                local_regions=local_regions, 
                                                                                                filter_grasps=filter_grasps, 
                                                                                                forward_passes=forward_passes
                                                                                                )
            
            # Set up parameters for the mask map
            grid_resolution = 0.0007  # Adjust the resolution as needed
            
            # Discretize the x and y axes into bins
            x_min, x_max = np.min(pc_full[:, 0]), np.max(pc_full[:, 0])
            y_min, y_max = np.min(pc_full[:, 1]), np.max(pc_full[:, 1])
            
            x_bins = np.arange(x_min, x_max, grid_resolution)
            y_bins = np.arange(y_min, y_max, grid_resolution)
            
            # width
            width = 0.08
            
            # gripper_depth
            gripper_depth = 0.1034
            
            ##
            res = np.zeros((len(y_bins), len(x_bins), 3), dtype=np.uint8)
            
            for (dish_idx, pc_seg), (grasp_key, grasp_value), (contact_key, contact_value) in zip(pc_segments.items(), pred_grasps_cam.items(), contact_pts.items()):
                # Assign b_vec from the current value in pred_grasps_cam
                if not grasp_value.any():
                    continue
                    
                # all_pc_segments = np.append(all_pc_segments, pc_seg, axis=0)
            
                # Assign each point to a bin
                x_indices = np.digitize(pc_seg[:, 0], x_bins)
                y_indices = np.digitize(pc_seg[:, 1], y_bins)
                
                mask = np.zeros((len(y_bins), len(x_bins), 3), dtype=np.uint8)
                mask[y_indices-1, x_indices-1] = [255,255,255]
                cv2.imwrite(f'1_grid_{dish_idx}.png', mask)
            
                # closing, opening operation with cv2.dilate and cv2.erode functions
                kernel_size = 5
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
                cv2.imwrite(f'2_dil_{dish_idx}.png', mask)
                mask = cv2.erode(mask, kernel, iterations=6)
                cv2.imwrite(f'3_ero_{dish_idx}.png', mask)
                
                gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                ret, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
                contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                
                obj_ct = []
                for ctr_i in contours[1:]:
                    hull = cv2.convexHull(ctr_i, clockwise=True)
                    mask = cv2.drawContours(mask, [hull], 0, (255, 255, 255), thickness=-1)
                    cv2.imwrite(f'4_convex_hull_{dish_idx}.png', mask)
                    
                    moments = cv2.moments(hull)
                    
                    if moments["m00"] != 0:
                        cX = int(moments["m10"] / moments["m00"])
                        cY = int(moments["m01"] / moments["m00"])
                            
                        obj_ct.append([cX, cY])
                
                ## draw contact points in mask map
                # Draw the original contact point
                x = contact_value[:, 0]
                y = contact_value[:, 1]
                z = contact_value[:, 2]

                x_indices = np.digitize(x, x_bins)
                y_indices = np.digitize(y, y_bins)
                
                res = np.where(res==[0,0,0], mask, res)
                for xi, yi in zip(x_indices, y_indices):
                    res = cv2.circle(res, (xi-1, yi-1), radius=2, color=(0, 0, 255), thickness=3)
                
                
                b_vec = grasp_value[:, :3, 0]

                # Calculate the new contact point
                new_contact_point = contact_value + b_vec * width
                
                # Draw the new contact point
                x_new = new_contact_point[:, 0]
                y_new = new_contact_point[:, 1]
                z_new = new_contact_point[:, 2]

                x_indices_new = np.digitize(x_new, x_bins)
                y_indices_new = np.digitize(y_new, y_bins)

                for xi, yi in zip(x_indices_new, y_indices_new):
                    res = cv2.circle(res, (xi-1, yi-1), radius=2, color=(0, 255, 0), thickness=3)

                # Connect the original and new contact points with lines
                for xi, yi, xi_new, yi_new in zip(x_indices, y_indices, x_indices_new, y_indices_new):
                    res = cv2.line(res, (xi-1, yi-1), (xi_new-1, yi_new-1), color=(0, 255, 255), thickness=1)
                    
                for x, y in obj_ct:
                    cv2.circle(res, (x, y), 3, (0, 127, 127), -1)
                    
                cv2.imwrite(f'5_pts&lines.png', res)
                
                ## remove contact points in the obj mask             
                indices = (mask[y_indices,x_indices] == [0,0,0]).all(axis=-1)

                ## get b_vectors in 2D    
                contact_pt      = np.stack((x_indices,      y_indices),     axis=1)
                new_contact_pt  = np.stack((x_indices_new,  y_indices_new), axis=1)
                
                b_vec = new_contact_pt - contact_pt
                b_vec = np.divide(b_vec,np.linalg.norm(b_vec, axis=1)[:, None])
                
                # obj_ct = obj_ct[(np.newaxis,) * (contact_pt - 1)]
                vec_xc = np.array(obj_ct) - contact_pt
                vec_xc = np.divide(vec_xc,np.linalg.norm(vec_xc, axis=1)[:, None])
                
                dot_b_xc = np.einsum('ij, ij -> i', b_vec, vec_xc)
                
                ## remove low similarity between b vector and orig
                indices = indices & (dot_b_xc > 0.8)
                
                ## extract filtered transformation matrices
                filtered_T = grasp_value[indices]
                
                ## project b vector to x-y plane and normalize
                b_vecs = filtered_T[:, :3, 0]
                b_vecs[..., 2] = 0
                b_vecs = np.divide(b_vecs, np.linalg.norm(b_vecs, axis=1)[:, None])
                
                ## edit a vector to [0,0,-1]
                n, _, _ = filtered_T.shape
                a_vecs = filtered_T[:, :3, 2]
                a_vecs[:] = [0, 0, 1]
                
                ## calcul a x b
                # cross_ab = np.einsum('ij, ij -> i', a_vecs, b_vecs)
                cross_ab = np.cross(a_vecs, b_vecs)
                
                ## obtain new Transformation
                new_T = np.tile(np.eye(4), (n, 1, 1))
                new_T[:, :3, 0] = b_vecs
                new_T[:, :3, 1] = cross_ab
                new_T[:, :3, 2] = a_vecs
                # ??
                trans = contact_value[indices] + (width/2-0.01) * b_vecs - gripper_depth * a_vecs
                new_T[:, :3, 3] = trans
                
                self.pred_grasps_cam2[grasp_key] = new_T[:]
                
         

            # Save the image
        cv2.imwrite('6_res.png', mask)

        cv2.destroyAllWindows()
            
            # # Visualize results          
        show_image(rgb_img, self.modified_mask)

        # visualize_grasps(pc_full, self.pred_grasps_cam2, scores, plot_opencv_cam=True, pc_colors=pc_colors, gripper_width=self.global_config['DATA']['gripper_width'])
        visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors, gripper_width=self.global_config['DATA']['gripper_width'])
            
        a = 0
            # # 결과를 시각화합니다.
            # self.show_image(rgb_img, segmap=self.modified_mask)
            # visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
            
            
            # self.show_image_cv(rgb_img, self.modified_mask)
            
            
            # self.show_image_cv(rgb = rgb_img, segmap=self.modified_mask)
            # self.visualize_grasps_with_opencv(rgb_img, pred_grasps_cam, scores, pc_colors=pc_colors)

        # else:
        #     print('Some image data is missing.')
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default= '', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    parser.add_argument('--np_path', default='test_data/7.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
    parser.add_argument('--K', default=[[923.26428, 0, 648.79083, 0, 922.75781, 313.74042, 0, 0, 1]], help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    
    print(str(global_config))
    print('pid: %s'%(str(os.getpid())))
    
    try:
    
        data_ros = DataROS(FLAGS, global_config)
        
        data_ros.run()
        
    except rospy.ROSInternalException:
        pass