import numpy as np
from pkg_resources import parse_version
from edgetpu import __version__ as edgetpu_version
assert parse_version(edgetpu_version) >= parse_version('2.11.1'), \
        'This demo requires Edge TPU version >= 2.11.1'

from edgetpu.basic.basic_engine import BasicEngine
from tflitemodel import TFLiteModel

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

class Pose(BasicEngine):
    def __init__(self, model_path, mirror=False):
        BasicEngine.__init__(self, model_path)
        self._mirror = mirror

        self._input_tensor_shape = self.get_input_tensor_shape()
        _, self.image_height, self.image_width, self.image_depth = self.get_input_tensor_shape()

        offset = 0
        self._output_offsets = [0]
        for size in self.get_all_output_tensors_sizes():
            offset += size
            self._output_offsets.append(offset)

    def detect(self, img):
        assert (img.shape == tuple(self._input_tensor_shape[1:]))

        # Run the inference (API expects the data to be flattened)
        inference_time, output = self.run_inference(img.flatten())
        outputs = [output[i:j] for i, j in zip(self._output_offsets, self._output_offsets[1:])]

        keypoints = outputs[0].reshape(-1, len(KEYPOINTS), 2)
        keypoint_scores = outputs[1].reshape(-1, len(KEYPOINTS))
        nposes = int(outputs[3][0])
        assert nposes < outputs[0].shape[0]
        return keypoints[:nposes], keypoint_scores[:nposes]

class PersonSegmentation(TFLiteModel):
    def __init__(self, model_path):
        self.load_model(model_path)

    def detect(self, image):
        interpretation = self.get_model_output(image)
        heatmap = interpretation[0]
        personmap = heatmap[:,:,0] / (heatmap[:,:,15] + 0.1)
        min = np.min(personmap)
        max = np.max(personmap)
        personmap= (personmap - min) / (max - min)
        return personmap


if __name__ == "__main__":
    import argparse, cv2
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, required=True)
    parser.add_argument("--image", default=None, required=True)

    args = parser.parse_args()
    tracker = Pose(model_path=args.model)

    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (tracker.image_width, tracker.image_height))
    keypoints = tracker.detect(image)
    print(keypoints)

