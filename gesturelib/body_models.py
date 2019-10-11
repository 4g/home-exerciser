import numpy as np
import cv2
from hand_model import Palm, HandLandMarks
from pose_engine import Pose, PersonSegmentation

pose_detection_tpu = "models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite"
segmentation_cpu = "models/deeplabv3_257_mv_gpu.tflite"
palm_cpu = "models/palm_detection_without_custom_op.tflite"
palm_anchors = "models/anchors.csv"

GREEN = (0,255,0)
RED = (0,0,255)

KEYPOINTS = (
  'nose',
  'left eye',
  'right eye',
  'left ear',
  'right ear',
  'left shoulder',
  'right shoulder',
  'left elbow',
  'right elbow',
  'left wrist',
  'right wrist',
  'left hip',
  'right hip',
  'left knee',
  'right knee',
  'left ankle',
  'right ankle'
)

EDGES = (
    ('nose', 'left eye'),
    ('nose', 'right eye'),
    ('nose', 'left ear'),
    ('nose', 'right ear'),
    ('left ear', 'left eye'),
    ('right ear', 'right eye'),
    ('left eye', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right elbow'),
    ('right shoulder', 'right hip'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip'),
    ('left hip', 'left knee'),
    ('right hip', 'right knee'),
    ('left knee', 'left ankle'),
    ('right knee', 'right ankle'),
)

def crop(image, width, height):
    desired_ratio = width / height
    image_width = image.shape[1]
    image_height = image.shape[0]
    image_ratio = image_width / image_height
    if image_ratio > desired_ratio:
        new_width = int(image_height * desired_ratio)
        image = image[:, image_width // 2 - new_width // 2 : image_width // 2 + new_width // 2]
    return image


class PoseLib:
    def __init__(self):
        self.pose = Pose(pose_detection_tpu)
        self.height = self.pose.image_height
        self.width = self.pose.image_width
        self.nodes = {name:i for i,name in enumerate(KEYPOINTS)}
        self.edges = [(self.nodes[x], self.nodes[y]) for x,y in EDGES]
        self.center = ['left hip', 'right hip',]
        self.center_indices = [self.nodes[x] for x in self.center]


    def get_crop_around_kp(self, image, keypoints, width, height):
        minx = min([i[0] for i in keypoints])
        maxx = max([i[0] for i in keypoints])

        miny = min([i[1] for i in keypoints])
        maxy = max([i[1] for i in keypoints])

        starty = (miny + maxy) // 2 - height // 2
        startx = (minx + maxx) // 2 - width // 2

        print (maxy - miny, maxx - minx, startx, starty)
        return image[starty:starty + height, startx:startx + width]


    def detect(self, original_image_int):
        pose_flag, pose1_kp, pose1_scores = False, None, None
        pose_image_int = original_image_int
        pose_image_int = cv2.resize(pose_image_int, (self.width, self.height))
        poses_kp, poses_scores = self.pose.detect(pose_image_int)

        if poses_kp is not None and len(poses_kp) > 0:
            pose_flag = True
            pose1_kp, pose1_scores = poses_kp[0], poses_scores[0]
            pose1_kp = [(int(y),int(x)) for x,y in pose1_kp]

        return pose_flag, pose1_kp, pose1_scores

    def create_keypoints_image(self, keypoints):
        zeros = np.zeros((self.pose.image_height, self.pose.image_width, 3), dtype=np.uint8)
        center_x = sum([keypoints[i][0] for i in self.center_indices]) // len(self.center_indices)
        center_y = sum([keypoints[i][1] for i in self.center_indices]) // len(self.center_indices)

        new_keypoints = []
        for x,y in keypoints:
            new_x = (x - center_x) + self.pose.image_width // 2
            new_y = (y - center_y) + self.pose.image_height // 2

            new_keypoints.append((new_x, new_y))

        self.draw(zeros, new_keypoints)
        return zeros

    def rescale_keypoints(self, keypoints, shape):
        scale_x = shape[1] / self.width
        scale_y = shape[0] / self.height
        keypoints = [(int(i[0] * scale_x), int(i[1] * scale_y)) for i in keypoints]
        return keypoints

    def draw(self, image, keypoints):
        if keypoints is not None:
            for point in keypoints:
                x, y = point
                x, y = x, y
                cv2.circle(image, (x, y), 3, GREEN, -1)

            for node_i,node_j in self.edges:
                pt1 = keypoints[node_i]
                pt2 = keypoints[node_j]
                cv2.line(image, pt1, pt2, RED, lineType=cv2.LINE_AA)

        return image

    def demo(self, cam):
#        import pyautogui
#        controller = 'right wrist'
#        controller_index = self.nodes[controller]
        cam.start()
        for i in tqdm(range(10000)):
            frame, count = cam.get()
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = crop(frame, self.pose.image_width, self.pose.image_height)
            frame = cv2.resize(frame, (self.pose.image_width, self.pose.image_height))
            pose_flag, keypoints, scores = self.detect(frame)


            if keypoints is not None:
#                print(keypoints[controller_index])
#                x,y = keypoints[controller_index]
#                w,h = pyautogui.size()
#                x = (x / self.width) * w
#                y = (y / self.height) * h
#                pyautogui.moveTo(x,y)
                scaled_keypoints = self.rescale_keypoints(keypoints, frame.shape)
                pose = self.draw(frame, scaled_keypoints)
                #skel = self.create_keypoints_image(keypoints)
                pose = cv2.cvtColor(pose, cv2.COLOR_RGB2BGR)
                cv2.imshow("win1", pose)
                #cv2.imshow("skel", skel)
                cv2.waitKey(1)

class PersonSegLib:
    def __init__(self):
        self.segmenter = PersonSegmentation(segmentation_cpu)
        self.height = self.segmenter.height
        self.width = self.segmenter.width

    def detect(self, image):
        image = self.segmenter.preprocess(image)
        heatmap = self.segmenter.detect(image)
        return heatmap

    def demo(self, cam):
        cam.start()
        for i in tqdm(range(10000)):
            frame, count = cam.get()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = crop(frame, self.width, self.height)
            heatmap = self.detect(frame)
            cv2.imshow("test", heatmap)
            cv2.waitKey(10)


class PalmLib:
    def __init__(self):
        self.palm_detector = Palm(palm_cpu, palm_anchors)
        self.width = self.palm_detector.image_width
        self.height = self.palm_detector.image_height

    def detect(self, image):
        image = np.asarray(image, dtype=np.float32)
        image = cv2.resize(image, (self.height, self.width))
        cv2.imshow("processing", image)
        image = 2 * ((image / 255.0) - 0.5)
        keypoints = self.palm_detector.detect(image)
        return keypoints

    def draw(self, image, keypoints):
        if keypoints is not None:
            for point in keypoints:
                x, y = point
                x, y = int(x), int(y)
                cv2.circle(image, (x, y), 3, GREEN, -1)
        return image

    def rescale_keypoints(self, keypoints, shape):
        scale_x = shape[1] / self.width
        scale_y = shape[0] / self.height
        keypoints = [(int(i[0] * scale_x), int(i[1] * scale_y)) for i in keypoints]
        return keypoints

    def demo(self, cam):
        cam.start()
        for i in tqdm(range(10000)):
            frame, count = cam.get()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = crop(frame, self.width, self.height)
            keypoints, handflag = self.detect(frame)
            if handflag:
                keypoints = self.rescale_keypoints(keypoints, frame.shape)
                self.draw(frame, keypoints)

            cv2.imshow("test", frame)
            cv2.waitKey(10)



if __name__ == "__main__":
    from camera import Camera
    from tqdm import tqdm
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None, required=True)
    parser.add_argument("--name", default=None, required=True)

    args = parser.parse_args()
    path = args.path
    
    if str.isdigit(args.path):
        path = int(args.path)

    cam = Camera(path, 30)


    if "seg" in args.name:
        seg = PersonSegLib()
        seg.demo(cam)

    if "pose" in args.name:
        body = PoseLib()
        body.demo(cam)

    if "palm" in args.name or "hand" in args.name:
        palm = PalmLib()
        palm.demo(cam)