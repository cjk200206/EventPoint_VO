import numpy as np
import cv2
from math import atan2, acos, pi,sqrt,pow

from sp_extractor import SuperPointFrontend, PointTracker, EventPointFrontend


STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2


class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
    def __init__(self, cam, annotations):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self.detector = SuperPointFrontend(weights_path="weights/superpoint_v1.pth",
                                           nms_dist=4,
                                           conf_thresh=0.015,
                                           nn_thresh=0.7,
                                           cuda=True)
        self.tracker = PointTracker(
            max_length=2, nn_thresh=self.detector.nn_thresh)
        with open(annotations) as f:
            self.annotations = f.readlines()

    def featureTracking(self):
        pts, desc, heatmap = self.detector.run(self.new_frame)
        # Add points and descriptors to the tracker.
        self.tracker.update(pts, desc)
        # Get tracks for points which were match successfully across all frames.
        tracks = self.tracker.get_tracks(min_length=2)
        # Normalize track scores to [0,1].
        tracks[:, 1] /= float(self.detector.nn_thresh)
        kp1, kp2 = self.tracker.draw_tracks(tracks)
        return kp1, kp2
    
    def rotm2euler(self):
        """
        Converts a rotation matrix to Euler angles (yaw, pitch, roll).
        
        Args:
            R (numpy.ndarray): 3x3 rotation matrix
        
        Returns:
            tuple: Euler angles (yaw, pitch, roll) in radians
        """
        # Yaw (around Z axis)
        self.yaw = atan2(self.cur_R[1,0], self.cur_R[0,0])
        
        # Pitch (around Y axis)
        # self.pitch = acos(self.cur_R[2,2])
        # if self.pitch > pi:
        #     self.pitch = 2*pi - self.pitch
        self.pitch = atan2(-self.cur_R[2,0],sqrt(pow(self.cur_R[2,1],2)+pow(self.cur_R[2,2],2)))
        
        # Roll (around X axis)    
        self.roll = atan2(self.cur_R[2,1], self.cur_R[2,2])

        self.yaw = self.yaw*180/pi
        self.pitch = self.pitch*180/pi
        self.roll = self.roll*180/pi
        
        return self.yaw, self.pitch, self.roll

    def getAbsoluteScale(self, frame_id):  # specialized for KITTI odometry dataset
        ss = self.annotations[frame_id - 1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.annotations[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev))

    def processFirstFrame(self):
        self.px_ref, self.px_cur = self.featureTracking()
        self.frame_stage = STAGE_SECOND_FRAME

    def processSecondFrame(self):
        self.px_ref, self.px_cur = self.featureTracking()

        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref,
                                       focal=self.focal, pp=self.pp,
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref,
                                                          focal=self.focal, pp=self.pp)
        self.frame_stage = STAGE_DEFAULT_FRAME
        self.px_ref = self.px_cur

    def processFrame(self, frame_id):
        self.px_ref, self.px_cur = self.featureTracking()

        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref,
                                       focal=self.focal, pp=self.pp,
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref,
                                        focal=self.focal, pp=self.pp)
        absolute_scale = self.getAbsoluteScale(frame_id)
            
        if(absolute_scale > 0.1):
            self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)
        self.px_ref = self.px_cur

    def update(self, img, frame_id):
        assert(img.ndim == 2 and img.shape[0] == self.cam.height and img.shape[1] ==
               self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        self.new_frame = img
        if(self.frame_stage == STAGE_DEFAULT_FRAME):
            self.processFrame(frame_id)
            self.rotm2euler()
        elif(self.frame_stage == STAGE_SECOND_FRAME):
            self.processSecondFrame()
        elif(self.frame_stage == STAGE_FIRST_FRAME):
            self.processFirstFrame()
        # self.processFrame(frame_id) # 新增测试
        self.last_frame = self.new_frame


class EventVisualOdometry(VisualOdometry):
    def __init__(self, cam, annotations):
        super(EventVisualOdometry,self).__init__(cam, annotations)

        self.detector = EventPointFrontend(weights_path= "weights/superpoint_05061917_best.pth",
                                    nms_dist=8,
                                    conf_thresh=0.020,
                                    nn_thresh=0.7,
                                    cuda=True)
        # self.detector = SuperPointFrontend(weights_path= "weights/superpoint_v1.pth",
        #                             nms_dist=8,
        #                             conf_thresh=0.015,
        #                             nn_thresh=0.7,
        #                             cuda=True)
        self.annotations = self.annotations[1:]
        # 用作测试
        self.absolute_scale = None
    

    def getAbsoluteScale(self, frame_id):  # specialized for VECtor odometry dataset
        ss = self.annotations[frame_id - 1].strip().split()
        x_prev = float(ss[1])
        y_prev = float(ss[2])
        z_prev = float(ss[3])
        ss = self.annotations[frame_id].strip().split()
        x = float(ss[1])
        y = float(ss[2])
        z = float(ss[3])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev))
    

    def processSecondFrame(self):
        self.px_ref, self.px_cur = self.featureTracking()
        try :
            E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref,
                                        focal=self.focal, pp=self.pp,
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref,
                                                            focal=self.focal, pp=self.pp)
            self.frame_stage = STAGE_DEFAULT_FRAME
            self.px_ref = self.px_cur
        except:
            self.frame_stage == STAGE_FIRST_FRAME


    def processFrame(self, frame_id):
        self.px_ref, self.px_cur = self.featureTracking()
        try :
            E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref,
                                        focal=self.focal, pp=self.pp,
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)

            _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref,
                                            focal=self.focal, pp=self.pp)
            self.absolute_scale = self.getAbsoluteScale(frame_id)
            
            if(self.absolute_scale > 0.01):
                # self.cur_t = self.cur_t + self.absolute_scale * self.cur_R.dot(t)
                self.cur_t = self.cur_t + self.cur_R.dot(t)
                self.cur_R = R.dot(self.cur_R)
            self.px_ref = self.px_cur
        except:
            self.frame_stage == STAGE_FIRST_FRAME

class VisualOdometry_without_gt(VisualOdometry):
    def __init__(self, cam, annotations = None):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        
        self.detector = SuperPointFrontend(weights_path="weights/superpoint_v1.pth",
                                           nms_dist=4,
                                           conf_thresh=0.025,
                                           nn_thresh=0.7,
                                           cuda=True)
        # self.detector = EventPointFrontend(weights_path= "weights/superpoint_05022138_best.pth",
        #                             nms_dist=4,
        #                             conf_thresh=0.025,
        #                             nn_thresh=0.3,
        #                             cuda=True)

        self.tracker = PointTracker(
            max_length=2, nn_thresh=self.detector.nn_thresh)


        # 用作测试
        self.absolute_scale = 1
    

    def getAbsoluteScale(self, frame_id):  # specialized for VECtor odometry dataset
        ss = self.annotations[frame_id - 1].strip().split()
        x_prev = float(ss[1])
        y_prev = float(ss[2])
        z_prev = float(ss[3])
        ss = self.annotations[frame_id].strip().split()
        x = float(ss[1])
        y = float(ss[2])
        z = float(ss[3])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev))
    

    def processSecondFrame(self):
        self.px_ref, self.px_cur = self.featureTracking()
        try :
            E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref,
                                        focal=self.focal, pp=self.pp,
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref,
                                                            focal=self.focal, pp=self.pp)
            self.frame_stage = STAGE_DEFAULT_FRAME
            self.px_ref = self.px_cur
        except:
            self.frame_stage == STAGE_FIRST_FRAME


    def processFrame(self, frame_id):
        self.px_ref, self.px_cur = self.featureTracking()
        try :
            E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref,
                                        focal=self.focal, pp=self.pp,
                                        method=cv2.RANSAC, prob=0.999, threshold=1.0)

            _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref,
                                            focal=self.focal, pp=self.pp)
            # self.absolute_scale = self.getAbsoluteScale(frame_id)
            
            
            self.cur_t = self.cur_t + self.absolute_scale*self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)
            self.px_ref = self.px_cur
        except:
            self.frame_stage == STAGE_FIRST_FRAME
        

    