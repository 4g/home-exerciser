from queue import LifoQueue
import cv2
import threading
import random
from threading import Lock

class Camera(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.capture = cv2.VideoCapture(0)
        self.frame_count = 0
        self.latest = (0, 0)

    def run(self):
        while self.capture.isOpened():
            retval, frame = self.capture.read()
            self.frame_count += 1
            self.latest = (frame, self.frame_count)

        return 0

    def get(self):
        return self.latest


def main():
    q = Camera()
    q.start()

    for i in range(1000):
        image, frame_count = q.get()
        delay = random.randint(10, 100)
        processed_image = dummy(image, delay)
        cv2.imshow("window", processed_image)

def dummy(frame, t):
    cv2.waitKey(t)
    frame = cv2.flip(frame, 1)
    return frame

if __name__ == "__main__":
    main()
