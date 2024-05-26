import math
import os
import cv2
import numpy as np
import onnxruntime
import torch

model_path = "./models/obb.onnx"
img_path = r""
imgsz = 640
conf = 0.5
iou = 0.15
CLASSES = ("target",)


class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, imgsz):
        self.imgsz = imgsz

    def __call__(self, img):
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(self.imgsz / shape[0], self.imgsz / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.imgsz - new_unpad[0], self.imgsz - new_unpad[1]  # wh padding

        # divide padding into 2 sides
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

        return img


class ONNX_model_container:
    def __init__(self, model_path):
        sess_option = onnxruntime.SessionOptions()
        sess_option.log_severity_level = 3
        self.sess = onnxruntime.InferenceSession(
            model_path, sess_option=sess_option, providers=["CPUExecutionProvider"]
        )
        self.model_path = model_path

    def run(self, img):
        input_data = img.transpose((2, 0, 1))
        input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
        input_data = input_data / 255.0
        input_dict = {}
        input_dict[self.sess.get_inputs()[0].name] = input_data

        output_list = []
        for output in self.sess.get_outputs():
            output_list.append(output.name)

        res = self.sess.run(output_list, input_dict)
        return res


def preprocess(img):
    img = letterbox(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class PostProcess:
    def __init__(self):
        self.max_wh = 7680

    def main(self, input_data):
        boxes, classes_conf, rads = [], [], []
        defualt_branch = 3
        pair_per_branch = len(input_data) // defualt_branch
        for i in range(defualt_branch):
            boxes.append(self.box_process(input_data[pair_per_branch * i]))
            classes_conf.append(input_data[pair_per_branch * i + 1])
            rads.append(input_data[pair_per_branch * i + 2])

        # [n,c,h,w] -> [n*h*w,c]
        boxes = [self.sp_flatten(_v) for _v in boxes]
        classes_conf = [self.sp_flatten(_v) for _v in classes_conf]
        rads = [self.sp_flatten(_v) for _v in rads]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        rads = np.concatenate(rads)

        # filter according to threshold
        boxes, classes, rads, scores = self.filter_boxes(boxes, classes_conf, rads)
        classes = classes.reshape((-1, 1))
        # nms
        xywhrs = np.concatenate(
            [boxes[:, :2] + classes * self.max_wh, boxes[:, 2:], rads], axis=-1
        )
        i = self.nms_rotated(torch.from_numpy(xywhrs), torch.from_numpy(scores), iou)
        boxes = boxes[i]
        classes = classes[i]
        rads = rads[i]
        scores = scores.reshape((-1, 1))[i]

        if len(rads) == 1:
            return [boxes], [classes], [rads], [scores]
        elif len(rads) == 0:
            return [], [], [], []
        else:
            return boxes, classes, rads, scores

    def box_process(self, position):

        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([imgsz // grid_h, imgsz // grid_w]).reshape((1, 2, 1, 1))

        position = self.dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xywh = np.concatenate(
            ((box_xy + box_xy2) * stride / 2, (box_xy2 - box_xy) * stride), axis=1
        )

        return xywh

    def dfl(self, position):
        # Distribution Focal Loss (DFL)
        x = torch.tensor(position)
        n, c, h, w = x.shape
        p_num = 4
        mc = c // p_num
        y = x.reshape(n, p_num, mc, h, w)
        y = y.softmax(2)
        acc_metrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
        y = (y * acc_metrix).sum(2)
        return y.numpy()

    def sp_flatten(self, _in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    def filter_boxes(self, boxes, box_class_probs, rads):
        """Filter boxes with object threshold."""

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)
        _class_pos = np.where(class_max_score >= conf)

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]
        rads = rads[_class_pos]
        scores = class_max_score[_class_pos]

        return boxes, classes, rads, scores

    def nms_rotated(self, boxes, scores, threshold=0.45):
        """
        NMS for obbs, powered by probiou and fast-nms.

        Args:
            boxes (torch.Tensor): (N, 5), xywhr.
            scores (torch.Tensor): (N, ).
            threshold (float): IoU threshold.

        Returns:
        """
        if len(boxes) == 0:
            return np.empty((0,), dtype=np.int8)
        sorted_idx = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_idx]
        ious = self.batch_probiou(boxes, boxes).triu_(diagonal=1)
        pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
        return sorted_idx[pick]

    def batch_probiou(self, obb1, obb2, eps=1e-7):
        """
        Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

        Args:
            obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
            obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

        Returns:
            (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
        """
        obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
        obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

        x1, y1 = obb1[..., :2].split(1, dim=-1)
        x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
        a1, b1, c1 = self._get_covariance_matrix(obb1)
        a2, b2, c2 = (x.squeeze(-1)[None] for x in self._get_covariance_matrix(obb2))

        t1 = (
            ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
            / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
        ) * 0.25
        t2 = (
            ((c1 + c2) * (x2 - x1) * (y1 - y2))
            / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
        ) * 0.5
        t3 = (
            ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
            / (
                4
                * (
                    (a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)
                ).sqrt()
                + eps
            )
            + eps
        ).log() * 0.5
        bd = (t1 + t2 + t3).clamp(eps, 100.0)
        hd = (1.0 - (-bd).exp() + eps).sqrt()
        return 1 - hd

    def _get_covariance_matrix(self, boxes):
        """
        Generating covariance matrix from obbs.

        Args:
            boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

        Returns:
            (torch.Tensor): Covariance metrixs corresponding to original rotated bounding boxes.
        """
        # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
        gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
        a, b, c = gbbs.split(1, dim=-1)
        cos = c.cos()
        sin = c.sin()
        cos2 = cos.pow(2)
        sin2 = sin.pow(2)
        return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


model = ONNX_model_container(model_path)
letterbox = LetterBox(imgsz)
postprocess = PostProcess()
for img in os.listdir(img_path):
    img = preprocess(
        cv2.imdecode(np.fromfile(f"{img_path}/{img}", dtype=np.uint8), cv2.IMREAD_COLOR)
    )
    output_data = model.run(img)

    boxes, classes, rads, scores = postprocess.main(output_data)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for score, rad, cl, (x, y, w, h) in zip(scores, rads, classes, boxes):
        degree = math.degrees(rad[0])
        points = cv2.boxPoints(((x, y), (w, h), degree)).astype(np.intp)
        cv2.drawContours(img, [points], -1, (255, 0, 0), 2)
        left, top = points.min(axis=0)
        cv2.putText(
            img,
            "{0} {1:.2f}".format(CLASSES[cl[0]], score[0]),
            (left, top - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    cv2.imshow("0", img)
    cv2.waitKey(1)
