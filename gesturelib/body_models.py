import numpy as np
import cv2
from hand_model import Palm, HandLandMarks
from pose_engine import Pose, PersonSegmentation

palm_detection_cpu = "models/palm_detection_without_custom_op.tflite"
palm_anchors = "models/anchors.csv"
pose_detection_tpu = "models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite"

GREEN = (0,255,0)
RED = (0,0,255)

def square_crop(image):
    h, w = image.shape[0], image.shape[1]
    size = min(h, w) // 2
    image = image[h // 2 - size:h // 2 + size, w // 2 - size:w // 2 + size, :]
    return image

def render_keypoints(image, keypoints, color):
    if keypoints is not None:
        for point in keypoints:
            x, y = point
            x, y = int(x), int(y)
            cv2.circle(image, (x, y), 3, color, -1)

class BodyModel:
    def __init__(self):
        self.palm = Palm(palm_detection_cpu, palm_anchors)
        self.pose = Pose(pose_detection_tpu)

    def get_all_body_points(self, original_image_int):
        pose_image_int = cv2.resize(original_image_int, (self.pose.image_width, self.pose.image_height))
        poses_kp, poses_scores = self.pose.detect(pose_image_int)
        if poses_kp is not None and len(poses_kp) > 0:
            pose1_kp, pose1_scores = poses_kp[0], poses_scores[0]
            pose1_kp = [(y,x) for x,y in pose1_kp]
            render_keypoints(pose_image_int, pose1_kp, GREEN)

        square_img_int = square_crop(original_image_int)
        square_img_int = cv2.resize(square_img_int, (self.palm.image_width, self.palm.image_height))
        square_img_float = np.asarray(square_img_int, dtype=np.float32)
        square_img_float_norm = 2 * (square_img_float / 255 - 0.5)
        palm_kp, palm_flag = self.palm.detect(square_img_float_norm)
        if palm_flag:
            render_keypoints(square_img_int, palm_kp, RED)


        return square_img_int, pose_image_int


import sys
body = BodyModel()
from camera import Camera
from tqdm import tqdm
cam = Camera(sys.argv[1])
cam.start()

for i in tqdm(range(10000)):
    frame, count = cam.get()
    image = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    palm, pose = body.get_all_body_points(frame)
    cv2.imshow("win", palm)
    cv2.imshow("win2", pose)
    cv2.waitKey(1)