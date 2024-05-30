"""图像标注脚本, 生成yolo格式的标注文件"""

import cv2
import glob
import os
import numpy as np
import easydict
import sys
from itertools import chain

def colorstr(*input):
    *args, string = (
        input if len(input) > 1 else ("blue", "bold", input[0])
    )  # color arguments, string
    colors = {
        "black": "\033[30m",  # 字体颜色
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # 高亮字体颜色
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "bg_red": "\033[41m",  # 背景颜色
        "bg_green": "\033[42m",
        "bg_yellow": "\033[43m",
        "bg_blue": "\033[44m",
        "bg_magenta": "\033[45m",
        "bg_cyan": "\033[46m",
        "bg_white": "\033[47m",
        "end": "\033[0m",  # 属性重置
        "bold": "\033[1m",  # 加粗
        "underline": "\033[4m",  # 下划线
        "twinkle": "\033[5m",  # 闪烁，vscode终端不支持，bash/zsh支持
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def plt_bbox(
    img,
    box,
    line_thickness=None,
    label_format="{id}",
    txt_color=(255, 255, 255),
    box_color=[255, 0, 0],
):

    if isinstance(box, np.ndarray):
        box = box.tolist()

    tl = line_thickness or round(
        0.001 * (img.shape[0] + img.shape[1]) / 2
    )  # line/font thickness
    tl = max(2, tl)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, p1, p2, box_color, tl)
    if label_format:
        tf = max(tl - 1, 1)  # font thickness
        sf = tl / 3  # font scale

        id = int(box[4])
        label = label_format.format(id=id)

        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(img, p1, p2, box_color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            sf,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    return img


def compute_color_for_labels(label):
    color = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [128, 255, 0],
        [255, 128, 0],
    ]
    return color[label % len(color)]


def save_txt(txt_path, info, mode="w"):
    os.makedirs(os.path.split(txt_path)[0], exist_ok=True)

    txt_file = open(txt_path, mode)
    for line in info:
        txt_file.write(line + "\n")
    txt_file.close()


def read_txt(txt_path):
    txt_file = open(txt_path, "r")
    txt_data = []
    for line in txt_file.readlines():
        txt_data.append(line.replace("\n", ""))

    return txt_data


# 目标框标注程序
class CLabeled:
    img_format = [".jpg", ".png", ".webp",".bmp",".jpeg"]
    _help = f"""
--------------------------------------------------------------------------
                        Instructions
            
Opening the Software

    Method 1: Drag and drop a folder or file onto label.exe to open it.
    Method 2: Use cmd to open with the command label.exe "path".

Display Help:

    Press the H key on the keyboard.

Steps to Draw a Box:

    1. Press and hold the left mouse button at the top-left corner of the target.
    2. Drag the mouse to the bottom-right corner of the target.
    3. Release the left mouse button.
    
Change Box Classification:

    Method 1: Press the Q or E key near the center of the target box.
    Method 2: Double-click the left mouse button near the center of the target box.

Delete a Specific Box:

    Single-click the left mouse button around the target box.

Hide or Show Marked Boxes:

    Single-click the middle mouse button.

Switch Images:

    Press the A key to go back and the D key to go forward.

Statistics of Annotation History in the Folder:

    Press the N key on the keyboard.

Zoom in on the Image:

    Scroll the mouse wheel.

Drag the Image:

    When the image is zoomed in, press and hold the middle mouse button and drag.

Exit and Save Results:

    Press the ESC key on the keyboard.
--------------------------------------------------------------------------
   
"""

    def __init__(self, args):
        print(CLabeled._help)
        if os.path.isfile(args.image_folder):
            if os.path.splitext(args.image_folder)[-1] in CLabeled.img_format:
                self.image_folder = os.path.dirname(args.image_folder)
            else:
                raise TypeError(
                    f"{colorstr('red', 'bold', 'TypeError:')} {args.image_folder} is not file type in {CLabeled.img_format}"
                )
        else:
            self.image_folder = args.image_folder

        self.images_list = sorted(
            chain(
                *[
                    glob.glob(
                        os.path.join(self.image_folder, f"**{os.sep}*{f}"),
                        recursive=True,
                    )
                    for f in CLabeled.img_format
                ]
            )
        )
        self.total_image_number = 0
        self._compute_total_image_number()

        self.checkpoint_path = os.path.join(self.image_folder, f"checkpoint")

        self.current_label_index = 0
        if os.path.exists(self.checkpoint_path):
            self.read_checkpoint(self.checkpoint_path)
        if os.path.isfile(args.image_folder):
            self.current_label_index = self.images_list.index(args.image_folder)

        self.image = None
        self.current_image = None
        self.label_path = None

        self.boxes = list()
        self.classes = list()
        self.cls_num = args.category_num
        self.cur_class = 0

        self.width = None
        self.height = None

        self.windows_name = "image"
        self.mouse_position = (0, 0)
        self.show_label = True

        self.ix, self.iy = -1, -1
        self.region = None  # x1, y1, x2, y2
        self.drawing = False

        self.vis_mode = 0
        self.mouse_event=None

    def _encode_image(self, image):
        """
        根据region对图像进行裁剪
        """
        if self.region is None:
            return image
        return image[
            self.region[1] : self.region[3], self.region[0] : self.region[2]
        ].copy()

    def _encode_boxes(self, boxes):
        """
        根据region对boxes进行裁剪
        """
        if self.region is None:
            return boxes
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = (x1 * self.width - self.region[0]) / (self.region[2] - self.region[0])
            y1 = (y1 * self.height - self.region[1]) / (self.region[3] - self.region[1])
            x2 = (x2 * self.width - self.region[0]) / (self.region[2] - self.region[0])
            y2 = (y2 * self.height - self.region[1]) / (self.region[3] - self.region[1])
            new_boxes.append([x1, y1, x2, y2])
        return np.array(new_boxes)

    def _decode_boxes(self, boxes):
        """
        根据region对boxes进行解码
        """
        if self.region is None:
            return boxes
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = (x1 * (self.region[2] - self.region[0]) + self.region[0]) / self.width
            y1 = (y1 * (self.region[3] - self.region[1]) + self.region[1]) / self.height
            x2 = (x2 * (self.region[2] - self.region[0]) + self.region[0]) / self.width
            y2 = (y2 * (self.region[3] - self.region[1]) + self.region[1]) / self.height
            new_boxes.append([x1, y1, x2, y2])
        return np.array(new_boxes)

    def _encode_point(self, point):
        """
        根据region对point进行裁剪
        """
        if self.region is None:
            return point
        x, y = point
        x = x - self.region[0]
        y = y - self.region[1]
        return int(x), int(y)

    def _decode_point(self, point):
        """
        根据region对point进行解码
        """
        if self.region is None:
            return point
        x, y = point
        x = x + self.region[0]
        y = y + self.region[1]
        return int(x), int(y)

    def _reset(self):
        self.image = None
        self.current_image = None
        self.label_path = None
        self.boxes.clear()
        self.classes.clear()
        self.show_label = True
        self.current_box_num = 0

    def _compute_total_image_number(self):
        self.total_image_number = len(self.images_list)

    def _backward(self):
        self.current_label_index -= 1
        self.current_label_index = max(0, self.current_label_index)

    def _roi_limit(self, x, y):
        x, y = min(max(x, 0), self.width), min(max(y, 0), self.height)
        return x, y

    def change_box_category(self, num=1):

        if len(self.boxes):
            if len(self.boxes) > 1:
                current_point = np.array(
                    [
                        self.mouse_position[0] / self.width,
                        self.mouse_position[1] / self.height,
                    ]
                )
                current_center_point = (
                    np.array([box[0:2] for box in self.boxes])
                    + np.array([box[2:4] for box in self.boxes])
                ) / 2  # 中心点
                square1 = np.sum(np.square(current_center_point), axis=1)
                square2 = np.sum(np.square(current_point), axis=0)
                squared_dist = (
                    -2 * np.matmul(current_center_point, current_point.T)
                    + square1
                    + square2
                )
                sort_index = np.argsort(squared_dist)[0]
            else:
                sort_index = -1

            if 0 <= self.classes[sort_index] + num < self.cls_num:
                self.cur_class = int(self.classes[sort_index] + num)
            elif self.classes[sort_index] + num >= self.cls_num:
                self.cur_class = 0
            else:
                self.cur_class = self.cls_num - 1

            self.classes[sort_index] = self.cur_class

    def box_fix(self, xyxy):
        x_center = float(xyxy[0] + xyxy[2]) / 2
        y_center = float(xyxy[1] + xyxy[3]) / 2
        width = abs(xyxy[2] - xyxy[0])
        height = abs(xyxy[3] - xyxy[1])
        xywh_center = [x_center, y_center, width, height]
        return xywh_center

    def _draw_roi(self, event, x, y, flags, param, mode=True):
        x, y = self._decode_point((x, y))
        x, y = self._roi_limit(x, y)
        self.mouse_position = (x, y)
        self.mouse_event = (event,flags,x,y)
        if event == cv2.EVENT_LBUTTONDOWN:  # 按下鼠标左键

            self.ix, self.iy = x, y

        elif not self.drawing and event == cv2.EVENT_MBUTTONUP:
            # 按住鼠标中键进行移动，拖动region
            if self.ix == x and self.iy == y:
                self.show_label = not self.show_label

        elif not self.drawing and event == cv2.EVENT_MBUTTONDOWN:
            # 按住鼠标中键进行移动，拖动region
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键松开
            if abs(x - self.ix) > 3 and abs(y - self.iy) > 3:

                box = [
                    self.ix / self.width,
                    self.iy / self.height,
                    x / self.width,
                    y / self.height,
                ]
                box = [
                    max(min(box[0], box[2]), 0),
                    max(min(box[1], box[3]), 0),
                    min(max(box[0], box[2]), 1),
                    min(max(box[1], box[3]), 1),
                ]
                self.boxes.append(box)
                self.classes.append(self.cur_class)
            self.drawing = False

        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.change_box_category()

        elif event == cv2.EVENT_RBUTTONDOWN:  # 删除(中心点或左上点)距离当前鼠标最近的框

            if len(self.boxes):
                if len(self.boxes) > 1:
                    current_point = np.array([x / self.width, y / self.height])
                    current_center_point = (
                        np.array([box[0:2] for box in self.boxes])
                        + np.array([box[2:4] for box in self.boxes])
                    ) / 2  # 中心点
                    square1 = np.sum(np.square(current_center_point), axis=1)
                    square2 = np.sum(np.square(current_point), axis=0)
                    squared_dist = (
                        -2 * np.matmul(current_center_point, current_point.T)
                        + square1
                        + square2
                    )
                    sort_index = np.argsort(squared_dist)[0]

                else:
                    sort_index = -1
                del self.boxes[sort_index]
                del self.classes[sort_index]

        elif not self.drawing and event == cv2.EVENT_MOUSEWHEEL:
            # 滚轮向上放大图片，滚轮向下缩小图片
            if self.region is None:
                region = [0, 0, self.width, self.height]
            else:  
                region = self.region
            current_scale = (region[2] - region[0]) / self.width
            # 以鼠标位置为中心缩放，缩放比例为0.1，放大代表着缩小region，所以缩放比例为负数
            scale = current_scale * 0.9 if flags > 0 else current_scale * 1.1
            scale = max(0.1, min(1.0, scale))  # 最小缩放比例为1.0，最大为10倍
            # 以鼠标位置为中心缩放
            new_width = int(self.width * scale)
            new_height = int(self.height * scale)
            # x,y 为鼠标在原图中的位置，保证放大后鼠标在原图中的位置不变
            x1 = int(
                x - (x - region[0]) / (region[2] - region[0]) * new_width
            )
            y1 = int(
                y
                - (y - region[1]) / (region[3] - region[1]) * new_height
            )
            self.region = [
                x1,
                y1,
                x1 + new_width,
                y1 + new_height,
            ]

            # 保持region大小同时保持region在原图中
            if self.region[0] < 0:
                self.region[0] = 0
                self.region[2] = new_width
            if self.region[1] < 0:
                self.region[1] = 0
                self.region[3] = new_height
            if self.region[2] > self.width:
                self.region[2] = self.width
                self.region[0] = max(0, self.width - new_width)
            if self.region[3] > self.height:
                self.region[3] = self.height
                self.region[1] = max(0, self.height - new_height)
            
            if self.region==[0, 0, self.width, self.height]:
                self.region=None

        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            # 按住鼠标左键进行移动，画框
            self.drawing = True

        elif (
            not self.drawing
            and event == cv2.EVENT_MOUSEMOVE
            and flags == cv2.EVENT_FLAG_MBUTTON
        ):
            # 按住鼠标中键进行移动，拖动region
            if self.region is not None:
                offset_x = x - self.ix
                offset_y = y - self.iy
                if abs(offset_x) > 3 and abs(offset_y) > 3:
                    new_width = self.region[2] - self.region[0]
                    new_height = self.region[3] - self.region[1]
                    self.region = [
                        self.region[0] - offset_x,
                        self.region[1] - offset_y,
                        self.region[2] - offset_x,
                        self.region[3] - offset_y,
                    ]
                    # 保持region大小同时保持region在原图中
                    if self.region[0] < 0:
                        self.region[0] = 0
                        self.region[2] = new_width
                    if self.region[1] < 0:
                        self.region[1] = 0
                        self.region[3] = new_height
                    if self.region[2] > self.width:
                        self.region[2] = self.width
                        self.region[0] = max(0, self.width - new_width)
                    if self.region[3] > self.height:
                        self.region[3] = self.height
                        self.region[1] = max(0, self.height - new_height)



    def _draw_box_on_image(self, image):
        boxes, classes = self.boxes, self.classes
        for box, cls in zip(boxes, classes):
            x1, y1 = (int(image.shape[1] * box[0]), int(image.shape[0] * box[1]))
            x2, y2 = (int(image.shape[1] * box[2]), int(image.shape[0] * box[3]))
            color = compute_color_for_labels(int(cls))
            box = [x1, y1, x2, y2, int(cls)]
            image = plt_bbox(image, box, box_color=color)
        
        return image

    def xywh2xyxy(self, xywh):
        """[x, y, w, h]转为[xmin, ymin, xmax, ymax]"""
        xmin = xywh[0] - xywh[2] / 2
        ymin = xywh[1] - xywh[3] / 2
        xmax = xywh[0] + xywh[2] / 2
        ymax = xywh[1] + xywh[3] / 2
        xyxy = [xmin, ymin, xmax, ymax]

        return xyxy

    def read_label_file(self, label_file_path):

        boxes = []
        classes = []

        annotation = read_txt(label_file_path)
        print(annotation)
        for bbox in annotation:
            bbox = list(map(float, bbox.split()))
            boxes.append(self.xywh2xyxy(bbox[1:]))
            classes.append(bbox[0])

        self.boxes = boxes
        self.classes = classes
        self.current_box_num = len(boxes)

    def write_label_file(self, label_file_path):
        ann_boxes = []
        for box, cls in zip(self.boxes, self.classes):
            box = list(map(str, self.box_fix(box)))
            box.insert(0, str(int(cls)))
            ann_boxes.append(" ".join(box))

        save_txt(label_file_path, ann_boxes)

    def write_checkpoint(self, checkpoint_path):

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint_file = open(checkpoint_path, "w")
        checkpoint_file.writelines(str(self.current_label_index))

    def read_checkpoint(self, checkpoint_path):
        checkpoint_file = open(checkpoint_path, "r")
        for line in checkpoint_file.readlines():
            self.current_label_index = int(line.strip())
        checkpoint_file.close()

    def set_mode(self, value):
        if self.image is not None:
            image = self.image.copy()
            if value == 1: # 直方图均衡化
                image = ImageHistogram(image)
            elif value == 2:
                alpha = 1.5
                image = cv2.addWeighted(image, alpha, image, 0, 0)
            elif value == 3:
                alpha = 2
                image = cv2.addWeighted(image, alpha, image, 0, 0)
                image = cv2.addWeighted(image, alpha, image, 0, 0)
            elif value == 4:
                alpha = 0.6
                image = cv2.addWeighted(image, alpha, image, 0, 0)
            elif value==5: # 抗曝光
                image1 =  255-image
                image = np.minimum(image,image1)
                image = ImageHistogram(image)
            elif value==6: # 幂律变换
                image = powerLawTrans(image)
                
            self.temp = image
            self.vis_mode = value


    def copy_image(
        self,
    ):
        if self.vis_mode > 0:
            image = self.temp.copy()
        else:
            image = self.image.copy()

        return image

    def render(self):
        image = self.copy_image()
        if not self.mouse_event is None:
            event,flags,x,y = self.mouse_event


            if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
                # 按住鼠标左键进行移动，画框
                color = compute_color_for_labels(self.cur_class)
                cv2.rectangle(image, (self.ix, self.iy), (x, y), color, 2)
                
            elif event == cv2.EVENT_MOUSEMOVE:
                cv2.line(
                    image,
                    (x, 0),
                    (x, self.height),
                    (255, 0, 0),
                    2,
                    8,
                )
                cv2.line(image, (0, y), (self.width, y), (255, 0, 0), 2, 8)
                
        if self.show_label:
            self._draw_box_on_image(image)
        
        cv2.imshow(self.windows_name, self._encode_image(image))
            
    def run(self):

        print("Total Num: ", self.total_image_number)
        cv2.namedWindow(self.windows_name, cv2.WINDOW_NORMAL)

        cv2.createTrackbar("mode", self.windows_name, self.vis_mode, 10, self.set_mode)
        visited_image = set()
        labeled_index, labeled_box = self.current_label_index, 0
        save_info = False
        init = True
        while True:
            if self.current_label_index != labeled_index or save_info or init:
                if save_info:
                    self.write_label_file(self.label_path)

                    labeled_index = self.current_label_index
                    labeled_box = max(
                        labeled_box + len(self.boxes) - self.current_box_num, 0
                    )
                    print(
                        f"Visted Img Num: {len(visited_image)}; Img Num: {self.total_image_number}; Labeled Box num: {labeled_box}\n"
                    )
                    save_info = False

                self.region = None

                init = False

                self.write_checkpoint(self.checkpoint_path)
                self._reset()
                labeled_index = self.current_label_index
                image_path = self.images_list[labeled_index]
                visited_image.add(image_path)
                self.label_path = image_path2label_path(image_path)

                self.image = cv2.imdecode(
                    np.fromfile(image_path, dtype=np.uint8),
                    1,
                )
                if self.vis_mode > 0:
                    self.set_mode(self.vis_mode)

                if os.path.exists(self.label_path):
                    self.read_label_file(self.label_path)

                print(
                    f"Img ID: {labeled_index}\nImg Path: {image_path}\nLabel Path: {self.label_path}\n"
                )
                self.width = self.image.shape[1]
                self.height = self.image.shape[0]


            cv2.setMouseCallback(self.windows_name, self._draw_roi)
            self.render()
            key = cv2.waitKey(1)

            if key == ord("q") or key == ord("Q"):
                self.change_box_category(-1)
                
            elif key == ord("e") or key == ord("E"):
                self.change_box_category(1)
                
            elif key == ord("a") or key == ord("A"):  # backward
                self._backward()
                save_info = True
                
            elif key == ord("d") or key == ord("D"):  # forward
                self.current_label_index = min(
                    self.current_label_index + 1, self.total_image_number - 1
                )
                save_info = True
                
            elif key == ord("l") or key == ord("L"):  # del img
                os.remove(self.images_list[self.current_label_index])
                if os.path.exists(self.label_path):
                    os.remove(self.label_path)
                del self.images_list[self.current_label_index]
                self._compute_total_image_number()
                labeled_box = max(labeled_box - len(self.boxes), 0)
                self.current_label_index = min(
                    self.current_label_index, self.total_image_number - 1
                )


            elif key == ord("n") or key == ord("N"):
                labeled_box = statistics_box_num(self.images_list)
                
            elif key == ord("h") or key == ord("H"):
                print(CLabeled._help)
                
            elif key == 27:  # exit
                self.write_label_file(self.label_path)
                break
            

def powerLawTrans(image):
    image = np.power(image,0.4)
    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    image = cv2.convertScaleAbs(image)
    return image
    
            
def ImageHistogram(image):
    (b, g, r) = cv2.split(image)  
    rH = cv2.equalizeHist(r)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    image = cv2.merge((bH, gH, rH))
    return image  


def image_path2label_path(image_path):
    sa, sb = f"{os.sep}images{os.sep}", f"labels"
    if sa in image_path:
        label_path = os.path.join(
            image_path.rsplit(sa, 1)[0],
            sb,
            image_path.rsplit(sa, 1)[1].rsplit(".", 1)[0] + ".txt",
        )
    else:
        label_path = image_path.rsplit(".", 1)[0] + ".txt"
    return label_path


def statistics_box_num(image_list):
    box_num = 0
    has_labeled = 0
    nolabel = 0
    unlabel_num = 0
    for i, image_path in enumerate(image_list):
        print(f"{i}/{len(image_list)}", end="\r")
        label_path = image_path2label_path(image_path)
        if not os.path.exists(label_path):
            unlabel_num += 1
            continue
        annotation = read_txt(label_path)
        box_num += len(annotation)
        if len(annotation) > 0:
            has_labeled += 1
        else:
            nolabel += 1
    print(
        f"\nTotal Num: {len(image_list)}\nBox Num: {box_num}\nUnvisted Img Num: {unlabel_num}\nLabeled Img Num: {has_labeled}\nUnlabeled Img Num: {nolabel}\n"
    )
    return box_num


def main():
    if len(sys.argv) > 1:
        image_folder = sys.argv[1]
    else:

        print(CLabeled._help)
        input("Press Enter to exit...")
        return

    if not os.path.exists(image_folder):
        raise ValueError(f"{colorstr('red', 'bold', 'ValueError:')} {image_folder} does not exists! please check it !")

    category_num = 4
    args = easydict.EasyDict(
        {
            "image_folder": image_folder,
            "category_num": category_num,
        }
    )
    _app = CLabeled(args)
    _app.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        input("Press Enter to exit...")
