import cv2
import threading
import random
from queue import LifoQueue

class Camera(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.capture = cv2.VideoCapture(0)
        self.frame_count = 0
        self.q = LifoQueue(1000)

    def run(self):
        while self.capture.isOpened():
            retval, frame = self.capture.read()
            self.frame_count += 1
            self.q.put((frame, self.frame_count))

    def get(self):
        o = self.q.get()
        self.q.queue.clear()
        return o

def main():
    q = Camera()
    q.start()

    for i in range(1000):
        image, frame_count = q.get()
        delay = random.randint(10, 10)
        print (f"delay {delay} frame {frame_count}")
        processed_image = dummy(image, delay)
        cv2.imshow("window", processed_image)

def dummy(frame, t):
    cv2.waitKey(t)
    frame = cv2.flip(frame, 1)
    return frame

if __name__ == "__main__":
    main()
