import cv2
import numpy as np


# 函数letterbox用于图像的缩放和填充
def letterbox(img, imgsz):
    shape = img.shape[:2]  # 获取图像的宽度和高度

    # 计算缩放比例
    r = min(imgsz / shape[0], imgsz / shape[1])

    # 计算填充
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = imgsz - new_unpad[0], imgsz - new_unpad[1]  # 计算上下左右填充

    # 将填充分为两半
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_NEAREST)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )  # 添加边界

    return img


# 函数deletterbox用于将缩放和填充还原
def deletterbox(xywh, org_w, org_h, imgsz):
    r = max(org_w / imgsz, org_h / imgsz)
    xywh *= r
    new_pad = imgsz * r
    dw, dh = new_pad - org_w, new_pad - org_h
    xywh[0] -= dw / 2
    xywh[1] -= dh / 2
    return xywh


# 函数plot_polygon用于绘制多边形
def plot_polygon(img, pointss, texts, coefficient=0.006):
    h, w = img.shape[:2]
    for points, text in zip(pointss, texts):
        points = np.array(points, dtype=np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(
            img,
            [points],
            isClosed=True,
            color=(255, 0, 0),
            thickness=int(min(h, w) * coefficient),
        )
        cv2.putText(
            img,
            text,
            (np.min(points[:, :, 0]).item(), np.min(points[:, :, 1]).item() - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            min(h, w) * coefficient,
            (0, 0, 255),
            2,
        )

    return img


# 以上是添加中文注释的代码
