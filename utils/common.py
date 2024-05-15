import cv2
import numpy as np


def letterbox(img, imgsz):
    shape = img.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(imgsz / shape[0], imgsz / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = imgsz - new_unpad[0], imgsz - new_unpad[1]  # wh padding

    # divide padding into 2 sides
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_NEAREST)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # add border

    return img


def deletterbox(xywh, org_w, org_h, imgsz):
    r = max(org_w / imgsz, org_h / imgsz)
    xywh *= r
    new_pad = imgsz * r
    dw, dh = new_pad - org_w, new_pad - org_h
    xywh[0] -= dw / 2
    xywh[1] -= dh / 2
    return xywh


def plot_polygon(img, pointss, texts, coefficient=0.006):
    h, w = img.shape[:2]
    for points, text in zip(pointss, texts):
        points = np.array(points, dtype=np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(img, [points], isClosed=True, color=(255, 0, 0), thickness=int(min(h, w) * coefficient))
        cv2.putText(img, text,
                    (np.min(points[:, :, 0]).item(), np.min(points[:, :, 1]).item() - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    min(h, w) * coefficient,
                    (0, 0, 255), 2)

    return img



