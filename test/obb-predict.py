import os
import queue
import threading
import time
from datetime import datetime
import cv2
from utils import common
from utils import model_complete as model
import numpy as np

pwd = "/home/cat/yolo"
imgs_path = f"{pwd}/images/7"

imgsz1 = 640
model_path1 = f"{pwd}/models/obb.rknn"
CLASSES1 = ("target",)
conf1 = 0.5
iou1 = 0.2

coefficient1 = 1.2  # 靶子截取范围系数
worker_num = 3  # 同时开启work_num个detect线程

timeout = 5


def camera(queues1):
    for idx, img in enumerate(os.listdir(imgs_path)):
        frame = cv2.imread(f"{imgs_path}/{img}")
        queues1[idx % worker_num].put(frame)


class Detect:
    def __init__(self, queue1, queue2, idx):
        self.queue1 = queue1
        self.queue2 = queue2
        self.npuid = idx % 3
        self.model1 = model.OBBModel(
            model_path1, imgsz1, conf1, iou1, CLASSES1, npuid=self.npuid
        )

    def detect(self):
        global flag, target_num
        while flag:
            try:
                frame = self.queue1.get(timeout=timeout)

                xyxys, xyxyxyxys = self.predict1(frame)

                for left, top, right, bottom in xyxys:
                    cv2.rectangle(
                        frame,
                        (int(left), int(top)),
                        (int(right), int(bottom)),
                        (0, 255, 0),
                        2,
                    )

                bounded_image = self.plot_polygon(xyxyxyxys, frame)

                self.queue2.put(bounded_image)

            except queue.Empty:
                lock.acquire()
                print("detect从camera获取帧超时")
                lock.release()
                time.sleep(1)
                continue

        lock.acquire()
        print("detect子线程关闭")
        lock.release()

    def predict1(self, img):
        xywhrs, _, _ = self.model1.run(img)
        if len(xywhrs) > 0:
            # xywhrs->xyxyxyxy
            x, y, w, h, r = np.split(xywhrs, [1, 2, 3, 4], axis=1)
            sin_r, cos_r = np.sin(r), np.cos(r)
            xyxyxyxys = np.concatenate(
                [
                    x - w / 2 * cos_r - h / 2 * sin_r,
                    y - w / 2 * sin_r + h / 2 * cos_r,
                    x + w / 2 * cos_r - h / 2 * sin_r,
                    y + w / 2 * sin_r + h / 2 * cos_r,
                    x + w / 2 * cos_r + h / 2 * sin_r,
                    y + w / 2 * sin_r - h / 2 * cos_r,
                    x - w / 2 * cos_r + h / 2 * sin_r,
                    y - w / 2 * sin_r - h / 2 * cos_r,
                ],
                axis=1,
            )
            xs = xyxyxyxys[:, ::2]
            ys = xyxyxyxys[:, 1::2]
            xyxys = np.concatenate(
                [
                    np.min(xs, axis=1, keepdims=True),
                    np.min(ys, axis=1, keepdims=True),
                    np.max(xs, axis=1, keepdims=True),
                    np.max(ys, axis=1, keepdims=True),
                ],
                axis=1,
            )
        else:
            xyxyxyxys = []
            xyxys = []
        return xyxys, xyxyxyxys

    def plot_polygon(self, xyxyxyxys, image):
        texts = ["target" for _ in range(len(xyxyxyxys))]
        img = common.plot_polygon(image, xyxyxyxys, texts, 0.001)

        return img


def save(queue1):
    while True:
        try:
            frame = queue1.get(timeout=timeout)
            cv2.imwrite(f"{pwd}/output/{now_time}/{time.time()}.jpg", frame)
        except queue.Empty:
            break


now_time = cv2.getTickCount()
os.makedirs(f"{pwd}/output/{now_time}")

lock = threading.Lock()
detect_queues = [queue.Queue() for _ in range(worker_num)]
save_queues = queue.Queue()

t1 = threading.Thread(target=camera, args=(detect_queues,))
tasks1 = [
    threading.Thread(target=Detect(detect_queues[idx], save_queues, idx).detect)
    for idx in range(worker_num)
]

flag = True
t1.start()
for t in tasks1:
    t.start()
save(save_queues)
flag = False
for t in tasks1:
    t.join()
