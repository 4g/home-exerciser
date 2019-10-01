from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow_core.lite.python.interpreter import Interpreter
from camera import Camera
import cv2
from tqdm import tqdm

class PoseNet:
    def __init__(self, size=257):
        if size not in {257, 513}:
            size = 257

        model = f'models/posenet_mobilenet_v1_100_{size}x{size}_multi_kpt_stripped.tflite'
        self.interpreter = Interpreter(model_path=model)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

    def interpret_image(self, image):
        image = (np.asarray(image, dtype=np.float32) - 127.5) / 127.5
        image = np.expand_dims(image, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        outputs = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]
        outputs = [np.squeeze(x) for x in outputs]
        return outputs

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def interpretation_to_pose(self, interpretation):
        heatmap, offsets, fwd_displacement, bkwd_displacement = interpretation
        heatmap = self.sigmoid(heatmap)
        height = heatmap.shape[0]
        width = heatmap.shape[1]

        numkeypoints = heatmap.shape[-1]

        keypoint_positions = {}

        for i in range(numkeypoints):
            heatmap_slice = heatmap[:,:,i]
            maxindex = np.argmax(heatmap_slice)
            maxindex = np.unravel_index(maxindex, heatmap_slice.shape)
            y, x = maxindex[0], maxindex[1]
            confidence = heatmap[y, x, i]
            y_offset, x_offset = offsets[y,x,i], offsets[y,x,i+numkeypoints]
            y = y / (height - 1) * self.height + y_offset
            x = x / (width - 1) * self.width + x_offset
            keypoint_positions[i] = (int(y), int(x), confidence)

        return keypoint_positions

    def square_crop(self, image):
        size = min(image.shape[0], image.shape[1])
        image = image[:size, :size, :]
        return image

    def run(self):
        q = Camera()
        q.start()
        while True:
            original_image, frame_count = q.get()
            original_image = self.square_crop(original_image)
            original_image = cv2.resize(original_image, (self.height, self.width))

            interpretation = self.interpret_image(original_image)
            pose = self.interpretation_to_pose(interpretation)
            for i in pose:
                y, x, confidence = pose[i]
                cv2.circle(original_image, (x,y), 5, (0,0,255), -1)
            cv2.imshow("image", original_image)
            cv2.waitKey(1)

posenet = PoseNet(513)
posenet.run()