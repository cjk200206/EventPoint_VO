import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import os

from scripts.visualization.eventreader import EventReader

from norm_visual_odometry import PinholeCamera, VisualOdometry
from sp_visual_odometry import VisualOdometry_without_gt as sp_VisualOdometry
from scripts.dataset.representations import get_timesurface

def render_for_model(x: np.ndarray, y: np.ndarray, pol: np.ndarray, H: int, W: int) -> np.ndarray:
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H,W), fill_value=0,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1],x[mask1]]= 1
    img[mask==1]= 1
    return img

def render(x: np.ndarray, y: np.ndarray, pol: np.ndarray, H: int, W: int) -> np.ndarray:
    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H,W,3), fill_value=255,dtype='uint8')
    mask = np.zeros((H,W),dtype='int32')
    pol = pol.astype('int')
    pol[pol==0]=-1
    mask1 = (x>=0)&(y>=0)&(W>x)&(H>y)
    mask[y[mask1],x[mask1]]=pol[mask1]
    img[mask==0]=[255,255,255]
    img[mask==-1]=[255,0,0]
    img[mask==1]=[0,0,255]
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Events Odometry')
    parser.add_argument('--img_file', type=str,default="/home/cjk2002/datasets/DSEC/train/images/interlaken_00_c_images_rectified_left")
    # parser.add_argument('--img_file', type=str,default="/home/cjk2002/datasets/DSEC/zurich_city_00_a_images_rectified_left")
    parser.add_argument('--pose_file', type=str,default="/home/cjk2002/datasets/VECTOR/gt/corridors_dolly1.synced.gt.txt", help='Path to pose file')
  
    # parser.add_argument('--img_file', type=str,default="/home/cjk2002/datasets/VECTOR/imgs/corridors_dolly1.synced.left_camera")
    # parser.add_argument('--pose_file', type=str,default="/home/cjk2002/datasets/VECTOR/gt/corridors_dolly1.synced.gt.txt", help='Path to pose file')
    parser.add_argument('--delta_time_ms', '-dt_ms', type=float, default=100.0, help='Time window (in milliseconds) to summarize events for visualization')
    parser.add_argument('--representation', '-rep', type=str, default='voxel', help='Event representations, voxel or sae')
    args = parser.parse_args()

    event_filepath = Path(args.img_file)
    dt = args.delta_time_ms

    # define FPS from events
    FPS = 1000/args.delta_time_ms 

    # for each camera model
    # cam_vector = PinholeCamera(1224,1024,886.19107,886.59163,610.57891,514.59271) #VECtor
    cam_vector = PinholeCamera(1440,1080,1164.6238115833075,1164.6238115833075,713.5791168212891,570.9349365234375) #DESC
    # height = 480
    # width = 640
    height = 1080
    width = 1440

    # pose_path
    pose_path = args.pose_file
    vo = VisualOdometry(cam_vector, pose_path)
    sp_vo = sp_VisualOdometry(cam_vector, pose_path)

    traj = np.zeros((1000, 1000, 3), dtype=np.uint8)

    # # log
    # log_fopen = open("results/VECtor_"+"corridors_dolly1"+".txt", mode='w+')
    log_fopen = open("results/DESC_"+"interlaken_00_c"+".txt", mode='w+')
    # log_fopen = open("results/DESC_"+"zurich_city_00_a"+".txt", mode='w+')

    # list
    sp_errors = []
    norm_errors = []
    sp_feature_nums = []
    norm_feature_nums = []
    
    # img list
    img_list = sorted(os.listdir(args.img_file))

    img_id = 0
    # read events and transform to img
    # for index in range(int(len(img_list)/3)): #VECtor
    for index in range(int(len(img_list))):

        ##普通的img
        try:
            # img_train_path = img_list[img_id*3] #VECtor
            img_train_path = img_list[img_id]
        except:
            img_train_path = img_list[-1]

        img_train = cv2.imread(os.path.join(args.img_file,img_train_path),cv2.COLOR_GRAY2RGB)
        img = cv2.cvtColor(img_train,cv2.COLOR_RGB2GRAY)
        # === superpoint ==============================
        sp_vo.update(img, img_id)

        sp_roll,sp_pitch,sp_yaw = sp_vo.roll,sp_vo.pitch,sp_vo.yaw
        sp_cur_t = sp_vo.cur_t
        if(img_id > 2) and (sp_vo.frame_stage == 2):
            sp_x, sp_y, sp_z = sp_cur_t[0], sp_cur_t[1], sp_cur_t[2]
        else:
            sp_x, sp_y, sp_z = 0., 0., 0.

        sp_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for (u, v) in sp_vo.px_ref:
            cv2.circle(sp_img, (u, v), 3, (0, 255, 0))

        # === normal ==================================
        vo.update(img, img_id)

        cur_t = vo.cur_t
        if(img_id > 2):
            x, y, z = cur_t[0], cur_t[1], cur_t[2]
        else:
            x, y, z = 0., 0., 0.

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for (u, v) in vo.px_ref:
            cv2.circle(img, (int(u), int(v)), 3, (0, 255, 0))

        # === calculation =============================
        # calculate error
        sp_est_point = np.array([sp_x, sp_z]).reshape(2)
        norm_est_point = np.array([x, z]).reshape(2)
        gt_point = np.array([sp_vo.trueX, sp_vo.trueZ]).reshape(2)
        gt_point = np.array([vo.trueX, vo.trueZ]).reshape(2)
        sp_error = np.linalg.norm(sp_est_point - gt_point)
        norm_error = np.linalg.norm(norm_est_point - gt_point)

        # append
        sp_errors.append(sp_error)
        norm_errors.append(norm_error)
        sp_feature_nums.append(len(sp_vo.px_ref))
        norm_feature_nums.append(len(vo.px_ref))

        # average
        avg_sp_error = np.mean(np.array(sp_errors))
        avg_norm_error = np.mean(np.array(norm_errors))
        avg_sp_feature_num = np.mean(np.array(sp_feature_nums))
        avg_norm_feature_num = np.mean(np.array(norm_feature_nums))

        # === log writer ==============================
        print(img_id, len(sp_vo.px_ref), len(vo.px_ref),
            float(sp_x), float(sp_y), float(sp_z), float(x), float(y), float(z),
            sp_vo.trueX, sp_vo.trueY, sp_vo.trueZ,
            float(sp_roll), float(sp_pitch), float(sp_yaw),file=log_fopen)
            
        # print(img_id, len(vo.px_ref),
        #       float(x), float(y), float(z),
        #        file=log_fopen)
        
        # print(img_id, len(sp_vo.px_ref),\
        #     float(sp_x), float(sp_y), float(sp_z), \
        #     file=log_fopen)

        # === drawer ==================================
        # each point
        sp_draw_x, sp_draw_y = int(sp_x) + 500, int(sp_y) + 500
        norm_draw_x, norm_draw_y = int(x) + 500, int(z) + 500
        true_x, true_y = int(vo.trueX) + 500, int(vo.trueY) + 500
        true_x, true_y = int(sp_vo.trueX) + 500, int(sp_vo.trueY) + 500
        tqdm.write('\rx = {} y = {} true_x = {} true_y = {} stage = {} absolute_scale = {}'\
                   .format(sp_x,sp_y,sp_vo.trueX,sp_vo.trueY,sp_vo.frame_stage,sp_vo.absolute_scale))
        tqdm.write('\rx = {} y = {} true_x = {} true_y = {} stage = {} absolute_scale = {}'\
                   .format(x,y,vo.trueX,vo.trueY,vo.frame_stage,vo.absolute_scale))

        # draw trajectory
        cv2.circle(traj, (sp_draw_x, sp_draw_y), 1, (255, 0, 0), 1)
        # cv2.circle(traj, (norm_draw_x, norm_draw_y), 1, (0, 255, 0), 1)
        cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 2)
        cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
        # draw text
        text = "Superpoint: [AvgFeature] %4.2f [AvgError] %2.4fm" % (
            avg_sp_feature_num, avg_sp_error)
        cv2.putText(traj, text, (20, 40),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, 8)
        text = "Normal    : [AvgFeature] %4.2f [AvgError] %2.4fm" % (
            avg_norm_feature_num, avg_norm_error)
        cv2.putText(traj, text, (20, 60),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, 8)

        # cv2.imshow('Road facing camera', np.concatenate((sp_img, img), axis=1))
        cv2.imshow('Road facing camera', sp_img)
        # cv2.imshow('Road facing camera', img)
        cv2.imshow('Trajectory', traj)
        cv2.waitKey(1)
        # read next img
        img_id += 1

    cv2.imwrite('map.png', traj)
