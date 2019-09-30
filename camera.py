from queue import LifoQueue
import cv2
import threading
import random

class CameraImageQueue(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.min_size = 10
        self.max_size = 100
        self.frame_count = 0
        self.q = LifoQueue(self.max_size)

    def run(self):
        reader = cv2.VideoCapture(0)
        while reader.isOpened():
            retval, frame = reader.read()
            self.frame_count += 1
            self.q.put((frame, self.frame_count))

    def get(self):
        o = self.q.get()
        self.q.queue.clear()
        return o

def main():
    q = CameraImageQueue()
    q.start()
    while True:
        image, frame_count = q.get()
        delay = random.randint(10, 300)
        processed_image = dummy(image, delay)
        print (f"Delay {delay} frame {frame_count} size {q.q.qsize()}" )
        cv2.imshow("window", processed_image)

def dummy(frame, t):
    cv2.waitKey(t)
    return frame

if __name__ == "__main__":
    main()
