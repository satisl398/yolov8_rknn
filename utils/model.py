from utils.common import letterbox, deletterbox
import os
import cv2
import numpy as np


class RKNN_model_container:
    def __init__(self, model_path, npuid):
        from rknnlite.api import RKNNLite
        rknn = RKNNLite(verbose=False)

        rknn.load_rknn(model_path)

        if npuid == 0:
            ret = rknn.init_runtime(target='rk3588', core_mask=RKNNLite.NPU_CORE_0)
        elif npuid == 1:
            ret = rknn.init_runtime(target='rk3588', core_mask=RKNNLite.NPU_CORE_1)
        elif npuid == 2:
            ret = rknn.init_runtime(target='rk3588', core_mask=RKNNLite.NPU_CORE_2)
        else:
            print('error npuid')
            exit()

        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)

        self.rknn = rknn

    def run(self, inputs):
        inputs = np.expand_dims(inputs, axis=0)
        res = self.rknn.inference(inputs=[inputs])
        return res


class ONNX_model_container:
    def __init__(self, model_path):
        import onnxruntime
        sess_option = onnxruntime.SessionOptions()
        sess_option.log_severity_level = 3
        self.sess = onnxruntime.InferenceSession(model_path, sess_option=sess_option,
                                                 providers=['CPUExecutionProvider'])

    def run(self, img):
        input_data = img.transpose((2, 0, 1))
        input_data = input_data[None].astype(np.float32)
        input_data = input_data / 255.
        input_dict = dict()
        input_dict[self.sess.get_inputs()[0].name] = input_data

        output_list = []
        for output in self.sess.get_outputs():
            output_list.append(output.name)

        res = self.sess.run(output_list, input_dict)
        return res


class BaseModel:
    def __init__(self, model_path: str, imgsz, conf, iou, classes: tuple, npuid=None):
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.model_path = model_path
        self.npuid = npuid
        self.model = self.load_model(model_path.rsplit('.', 1)[1])

    def load_model(self, type):
        if type == 'onnx':
            return ONNX_model_container(self.model_path)
        elif type == 'rknn':
            return RKNN_model_container(self.model_path, self.npuid)

    def run(self, img):
        h, w = img.shape[:2]
        img = self.preprocess(img)
        output_data = self.model.run(img)
        points, classes, scores = self.postprocess(output_data, w, h)
        return points, classes, scores

    def preprocess(self, img):
        img = letterbox(img, self.imgsz)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def postprocess(self, input_data, org_w, org_h):
        pass

    def dfl(self, position):
        # Distribution Focal Loss (DFL)
        n, c, h, w = position.shape
        p_num = 4
        mc = c // p_num
        y = position.reshape(n, p_num, mc, h, w)
        y = self.softmax(y, 2)
        acc_metrix = np.arange(mc).astype(float).reshape((1, 1, mc, 1, 1))
        y = (y * acc_metrix).sum(2)
        return y

    def softmax(self, x, axis=-1):
        # 计算每行的指数值
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        # 对每行进行归一化
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def sp_flatten(self, _in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    def filter_boxes(self, boxes, box_class_probs):
        """Filter boxes with object threshold.
        """
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)
        _class_pos = np.where(class_max_score >= self.conf)
        boxes = boxes[_class_pos]
        classes = classes[_class_pos]
        scores = class_max_score[_class_pos]

        return boxes, classes, scores


class DetectModel(BaseModel):
    def postprocess(self, input_data, org_w, org_h):
        dists, classes_conf = [], []
        defualt_branch = 3
        pair_per_branch = len(input_data) // defualt_branch
        for i in range(defualt_branch):
            dists.append(self.dfl(input_data[pair_per_branch * i]))
            classes_conf.append(input_data[pair_per_branch * i + 1])

        # [n,c,h,w] -> [n*h*w,c]
        boxes = [self.sp_flatten(self.dist2bbox(_v)) for _v in dists]
        classes_conf = [self.sp_flatten(_v) for _v in classes_conf]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)

        # filter according to threshold
        boxes, classes, scores = self.filter_boxes(boxes, classes_conf)
        # nms
        nxywhs, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self.nms(b, s)
            if len(keep) != 0:
                xyxys = b[keep]
                xywhs = np.concatenate([(xyxys[:, :2] + xyxys[:, 2:4]) / 2, xyxys[:, 2:4] - xyxys[:, :2]], axis=1)
                nxywhs.append([deletterbox(xywh, org_w, org_h, self.imgsz) for xywh in xywhs])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return [], [], []
        xywhs = np.concatenate(nxywhs)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return xywhs, classes, scores

    def dist2bbox(self, position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.imgsz // grid_h, self.imgsz // grid_w]).reshape((1, 2, 1, 1))

        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
        return xyxy

    def nms(self, boxes, scores):
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
            j = order[1:]
            keep.append(i)

            xx1 = np.maximum(x[i], x[j])
            yy1 = np.maximum(y[i], y[j])
            xx2 = np.minimum(x[i] + w[i], x[j] + w[j])
            yy2 = np.minimum(y[i] + h[i], y[j] + h[j])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[j] - inter)
            inds = np.where(ovr <= self.iou)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep


class OBBModel(BaseModel):
    def postprocess(self, input_data, org_w, org_h):
        max_wh = 7680

        dists, classes_conf, rads = [], [], []
        defualt_branch = 3
        pair_per_branch = len(input_data) // defualt_branch
        for i in range(defualt_branch):
            dists.append(self.dfl(input_data[pair_per_branch * i]))
            classes_conf.append(input_data[pair_per_branch * i + 1])
            rads.append((input_data[pair_per_branch * i + 2]-0.25) * np.pi)

        # [n,c,h,w] -> [n*h*w,c]
        xywhrs = [self.sp_flatten(self.dist2rbox(_v1, _v2)) for _v1, _v2 in zip(dists, rads)]
        classes_conf = [self.sp_flatten(_v) for _v in classes_conf]

        xywhrs = np.concatenate(xywhrs)
        classes_conf = np.concatenate(classes_conf)

        # filter according to threshold
        xywhrs, classes, scores = self.filter_boxes(xywhrs, classes_conf)

        # nms
        i = self.nms(np.concatenate([xywhrs[:, :2] + classes[:, None] * max_wh, xywhrs[:, 2:4], xywhrs[:, 4:]], axis=1),
                     scores)
        xywhrs = xywhrs[i]
        if len(xywhrs) == 0:
            return [], [], []

        xywhrs[:, :4] = [deletterbox(xywh, org_w, org_h, self.imgsz) for xywh in xywhrs[:, :4]]
        classes = classes[i]
        scores = scores[i]

        return xywhrs, classes, scores

    def dist2rbox(self, dists, rads):
        lt, rb = np.split(dists, 2, axis=1)
        cos, sin = np.cos(rads), np.sin(rads)
        xf, yf = np.split((rb - lt) / 2, 2, axis=1)
        x, y = xf * cos - yf * sin, xf * sin + yf * cos

        grid_h, grid_w = dists.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        stride = np.array([self.imgsz // grid_h, self.imgsz // grid_w]).reshape((1, 2, 1, 1))
        box_x, box_y = col + 0.5 + x, row + 0.5 + y
        box_xy = np.concatenate((box_x, box_y), axis=1)
        return np.concatenate([box_xy * stride, (rb + lt) * stride, rads], axis=1)

    def nms(self, boxes, scores):
        if len(boxes) == 0:
            return np.empty((0,), dtype=np.int8)
        sorted_idx = np.argsort(scores)[::-1]
        boxes = boxes[sorted_idx]
        ious = np.triu(self.batch_probiou(boxes, boxes), k=1)
        pick = np.nonzero(np.max(ious, axis=0) < self.iou)[0]

        return sorted_idx[pick]

    def batch_probiou(self, obb1, obb2, eps=1e-7):
        obb1 = obb1 if isinstance(obb1, np.ndarray) else obb1.numpy()
        obb2 = obb2 if isinstance(obb2, np.ndarray) else obb2.numpy()

        x1, y1 = np.split(obb1[..., :2], 2, axis=-1)
        x2, y2 = (x.squeeze(-1)[None] for x in np.split(obb2[..., :2], 2, axis=-1))
        a1, b1, c1 = self._get_covariance_matrix(obb1)
        a2, b2, c2 = (x.squeeze(-1)[None] for x in self._get_covariance_matrix(obb2))

        t1 = (
                     ((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) /
                     ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)
             ) * 0.25
        t2 = (
                     ((c1 + c2) * (x2 - x1) * (y1 - y2)) /
                     ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps)
             ) * 0.5
        t3 = np.log(
            ((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2) /
            (4 * ((a1 * b1 - c1 ** 2).clip(0) * (a2 * b2 - c2 ** 2).clip(0)) ** 0.5 + eps)
            + eps) * 0.5
        bd = np.clip(t1 + t2 + t3, eps, 100.0)
        hd = (1.0 - np.exp(-bd) + eps) ** 0.5
        return 1 - hd

    def _get_covariance_matrix(self, boxes):
        # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
        gbbs = np.concatenate((boxes[:, 2:4] ** 2 / 12, boxes[:, 4:]), axis=-1)
        a, b, c = np.split(gbbs, 3, axis=-1)
        cos = np.cos(c)
        sin = np.sin(c)
        cos2 = cos ** 2
        sin2 = sin ** 2
        return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


if __name__ == '__main__':
    from utils.common import plot_polygon

    model_path = '../models/7.onnx'
    img_path = '../images/7'
    # imgsz = 160
    imgsz = 640
    conf = 0.5
    # iou = 0.5
    iou = 0.2
    # CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'locater')
    CLASSES = ('target',)
    # model = DetectModel(model_path, imgsz, conf, iou, CLASSES)
    model = OBBModel(model_path, imgsz, conf, iou, CLASSES)
    frames_num = 0
    for img in os.listdir(img_path):
        img = cv2.imdecode(np.fromfile(f'{img_path}/{img}', dtype=np.uint8), cv2.IMREAD_COLOR)
        __s, classes, scores = model.run(img)

        # if len(__s) > 0:
        #     x, y, w, h = np.split(__s, [1, 2, 3], 1)
        #     pointss = np.concatenate([
        #         x - w / 2, y - h / 2,
        #         x - w / 2, y + h / 2,
        #         x + w / 2, y + h / 2,
        #         x + w / 2, y - h / 2
        #     ], axis=1)
        # else:
        #     pointss = []

        if len(__s) > 0:
            # xywhrs->xyxyxyxy
            x, y, w, h, r = np.split(__s, [1, 2, 3, 4], axis=1)
            sin_r, cos_r = np.sin(r), np.cos(r)
            pointss = np.concatenate([
                x - w / 2 * cos_r - h / 2 * sin_r,
                y - w / 2 * sin_r + h / 2 * cos_r,
                x + w / 2 * cos_r - h / 2 * sin_r,
                y + w / 2 * sin_r + h / 2 * cos_r,
                x + w / 2 * cos_r + h / 2 * sin_r,
                y + w / 2 * sin_r - h / 2 * cos_r,
                x - w / 2 * cos_r + h / 2 * sin_r,
                y - w / 2 * sin_r - h / 2 * cos_r
            ], axis=1)
        else:
            pointss = []

        texts = ['{0} {1:.2f}'.format(CLASSES[cl], score) for cl, score in zip(classes, scores)]
        img = plot_polygon(img, pointss, texts)
        cv2.imshow('0', cv2.resize(img, (640, 640)))
        cv2.waitKey(0)
        # break
