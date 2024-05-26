import os
import cv2
import numpy as np
import onnxruntime

model_path = "./models/5.onnx"
img_path = r""
imgsz = 160
conf = 0.5
iou = 0.5
CLASSES = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "locater")


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
    def main(self, input_data):
        boxes, classes_conf = [], []
        defualt_branch = 3
        pair_per_branch = len(input_data) // defualt_branch
        for i in range(defualt_branch):
            boxes.append(self.box_process(input_data[pair_per_branch * i]))
            classes_conf.append(input_data[pair_per_branch * i + 1])

        # [n,c,h,w] -> [n*h*w,c]
        boxes = [self.sp_flatten(_v) for _v in boxes]
        classes_conf = [self.sp_flatten(_v) for _v in classes_conf]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)

        # filter according to threshold
        boxes, classes, scores = self.filter_boxes(boxes, classes_conf)

        # nms
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self.nms_boxes(b, s)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return [], [], []

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

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
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

        return xyxy

    def dfl(self, position):
        # Distribution Focal Loss (DFL)
        import torch

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

    def filter_boxes(self, boxes, box_class_probs):
        """Filter boxes with object threshold."""

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)
        _class_pos = np.where(class_max_score >= conf)

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]
        scores = class_max_score[_class_pos]

        return boxes, classes, scores

    def nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.
        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep


model = ONNX_model_container(model_path)
letterbox = LetterBox(imgsz)
postprocess = PostProcess()
for img in os.listdir(img_path):
    img = preprocess(
        cv2.imdecode(np.fromfile(f"{img_path}/{img}", dtype=np.uint8), cv2.IMREAD_COLOR)
    )
    output_data = model.run(img)

    boxes, classes, scores = postprocess.main(output_data)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for box, score, cl in zip(boxes, scores, classes):
        left, top, right, bottom = [int(_b) for _b in box]
        print(
            "%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], left, top, right, bottom, score)
        )
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(
            img,
            "{0} {1:.2f}".format(CLASSES[cl], score),
            (left, top - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    cv2.imshow("0", img)
    cv2.waitKey(0)
