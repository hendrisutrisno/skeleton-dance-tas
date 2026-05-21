import cv2
import os
import numpy as np
import pandas as pd
from tracker import Tracker
from loguru import logger
import traceback

try:
    import pyopenpose as op
    logger.success("OpenPose library loaded successfully.")
except ImportError:
    logger.exception("Error: OpenPose library could not be found.")
    raise ImportError("OpenPose library is required but not found.")

keypoint_map = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}
keypoint_map_25 = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "MidHip": 8,
    "RHip": 9,
    "RKnee": 10,
    "RAnkle": 11,
    "LHip": 12,
    "LKnee": 13,
    "LAnkle": 14,
    "REye": 15,
    "LEye": 16,
    "REar": 17,
    "LEar": 18,
    "LBigToe": 19,
    "LSmallToe": 20,
    "LHeel": 21,
    "RBigToe": 22,
    "RSmallToe": 23,
    "RHeel": 24
}

head_drop = [0, 14, 15]

class OpenposeModel:
    def __init__(self):
        self.model = self.init_model()
        self.conf_threshold = 0.7
        self.frames = []
        self.skeletons = []

    def init_model(self):
        try:
            params = {
                'model_folder': os.path.abspath(os.path.join(os.path.dirname(__file__), "../openpose/models")),
                "model_pose": "BODY_25" #"COCO" 
            }
            wrapper = op.WrapperPython()
            wrapper.configure(params)
            logger.info("OpenPose model initialized.")
            return wrapper
        except Exception as e:
            logger.exception(f"Failed to initialize OpenPose model: {e}")
            raise

    def openpose_extraction(self, video_path, result_path, check_path):
        self.tracker = Tracker()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result_video = cv2.VideoWriter(
            result_path, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=30.0, frameSize=(width, height)
        )
        frame_number = 1
        self.frames = []
        self.skeletons = []

        try:
            datum = op.Datum()
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                datum.cvInputData = frame
                self.model.emplaceAndPop(op.VectorDatum([datum]))
                keypoints = datum.poseKeypoints
                annotated_frame = datum.cvOutputData
                
                self.tracker.predict()
                pre_id = 1
                ids, bbox = self.tracker.update(keypoints)

                if not ids:
                    self.skeletons.append([frame_number, 0] + [0]*75) # 75
                
                else:
                    for id, i in ids:
                        key = keypoints[i].flatten().tolist()
                        _, key = self.check(key)
                        self.skeletons.append([frame_number, int(id)] + key)
                        cv2.putText(annotated_frame, str(id), (int(key[0]), int(key[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    """
                    id, i  = ids[0]
                    key = keypoints[i].flatten().tolist()
                    _, key = self.check(key)
                    self.skeletons.append([frame_number, int(id)] + key)
                    #cv2.putText(annotated_frame, str(id), (int(key[0]), int(key[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    """
                for i, box in enumerate(bbox):
                    cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, str(i), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                #self.frames.append(annotated_frame)
                result_video.write(annotated_frame)
                frame_number += 1

        except Exception as e:
            logger.error(f"Error during OpenPose extraction {result_path}: {e}")
            logger.error(traceback.format_exc()) 
        finally:
            result_video.release()
            cap.release()
            logger.success("OpenPose extraction finished.")

    def check(self, key):
        key_points = np.array(key, dtype=float)
        conf = key_points[::3]
        if conf.shape[0] != 25:
            logger.warning(f"Expected 25 confidence scores, but got {conf.shape[0]}")
            return True, key_points.tolist()

        miss_point = np.where(conf < self.conf_threshold)
        for col in miss_point[0]:
            key_points[col*3: (col+1)*3] = 0.
        
        logger.info("Confidence check completed.")
        if conf[2] ==0 or conf[5] ==0:
            return True, key_points.tolist()
        else:
            return False, key_points.tolist()
        



