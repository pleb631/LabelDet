"""图像标注脚本, 生成yolo格式的标注文件"""

import cv2
import glob
import os
import numpy as np
import easydict
import sys
from itertools import chain
import time

from utils import *


class config_window:

    def show(self,main_window):
        if not cv2.getWindowProperty(main_window.config_name, cv2.WND_PROP_VISIBLE) < 1:
            return 0
        cv2.namedWindow(main_window.config_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(main_window.config_name, 320, 320)
        
        cv2.createTrackbar("vis-mode", main_window.config_name, main_window.vis_mode, 10, main_window.set_mode)
         
        cv2.createTrackbar("thickness", main_window.config_name, main_window.thickness, 5, main_window.set_thickness)
        cv2.createTrackbar("img-scale", main_window.config_name, main_window.imgscale_mode, 1, main_window.set_imgscale_mode)
        
        cv2.createTrackbar("show_num", main_window.config_name, main_window.show_boxnum, 1, main_window.set_show_boxnum)
        
        cv2.createTrackbar("cls-num", main_window.config_name, main_window.cls_num, main_window.cls_num, main_window.set_class_num)
        
class CLabeled:
    img_format = [".jpg", ".png", ".webp",".bmp",".jpeg"]
    _help = f"""
--------------------------------------------------------------------------
                        Instructions
            
Opening the Software

    Method 1: Drag and drop a folder or file onto label.exe to open it.
    Method 2: Use cmd to open with the command label.exe "path".

Display Help and Config:

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

        self.checkpoint_path = os.path.join(self.image_folder, "checkpoint")

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

        self.windows_name = "image"
        self.config_name = "config"
        self.mouse_position = (0, 0)
        self.show_label = True
        self.win_info=None

        self.ix, self.iy = -1, -1
        self.region = None  # x1, y1, x2, y2
        self.drawing = False

        self.vis_mode = 0
        self.thickness = 2
        self.imgscale_mode = 0
        self.show_boxnum = 1
        self.mouse_event=None
        
        self.reload=True
        self.copy_box = None

    def _encode_image(self, image):
        """
        根据region对图像进行裁剪
        """
        if self.region is None:
            return image
        return image[
            self.region[1] : self.region[3], self.region[0] : self.region[2]
        ]


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


    def change_box_category(self, num=1):

        if len(self.boxes):
            if len(self.boxes) > 1:
                new_unpad = self.win_info[6]
                current_point = np.array(
                    [
                        (self.mouse_position[0]-self.win_info[2]) / new_unpad[0],
                        (self.mouse_position[1]-self.win_info[3]) / new_unpad[1],
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



    def _draw_roi(self, event, x, y, flags, param, mode=True):
        x, y = self._decode_point((x, y))

        self.mouse_position = (x, y)
        self.mouse_event = (event,flags,x,y)

        (win_width, win_height,left,top,right,bottom,new_unpad,interpolation) = self.win_info
        
        if event == cv2.EVENT_LBUTTONDOWN:  # 按下鼠标左键
            
                self.ix, self.iy = x, y
        elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键松开
            #需要修改越界画框的问题和存储框
            if abs(x - self.ix) > 3 and abs(y - self.iy) > 3:

                box = [
                    (self.ix-left) / new_unpad[0],
                    (self.iy-top) / new_unpad[1],
                    (x-left) / new_unpad[0],
                    (y-top) / new_unpad[1],
                ]
                box = [
                    max(min(box[0], box[2],1), 0),
                    max(min(box[1], box[3],1), 0),
                    min(max(box[0], box[2],0), 1),
                    min(max(box[1], box[3],0), 1),
                ]
                if (box[2]-box[0])*win_width > 3 and (box[3]-box[1])*win_height:
                    self.boxes.append(box)
                    self.classes.append(self.cur_class)
            self.drawing = False

        elif not self.drawing and event == cv2.EVENT_MBUTTONUP:
        # 按住鼠标中键进行移动，拖动region
            if self.ix == x and self.iy == y:
                self.show_label = not self.show_label

        elif not self.drawing and event == cv2.EVENT_MBUTTONDOWN:
            # 按住鼠标中键进行移动，拖动region
            self.ix, self.iy = x, y
            
        elif not self.drawing and event == cv2.EVENT_MOUSEWHEEL:
            # 滚轮向上放大图片，滚轮向下缩小图片
            if self.region is None:
                region = [0, 0, win_width, win_height]
            else:  
                region = self.region
            current_scale = (region[2] - region[0]) / win_width
            # 以鼠标位置为中心缩放，缩放比例为0.1，放大代表着缩小region，所以缩放比例为负数
            scale = current_scale * 0.9 if flags > 0 else current_scale * 1.1
            scale = max(0.1, min(1.0, scale))  # 最小缩放比例为1.0，最大为10倍
            # 以鼠标位置为中心缩放
            new_width = int(win_width * scale)
            new_height = int(win_height * scale)
            # x,y 为鼠标在原图中的位置，保证放大后鼠标在原图中的位置不变
            x1 = int(
                x - (x - region[0]) / (region[2] - region[0]) * new_width
            )
            y1 = int(
                y
                - (y - region[1]) / (region[3] - region[1]) * new_height
            )
            
            x1 = max(x1,0)
            y1 = max(y1,0)
            x2 = min(x1 + new_width, win_width)
            y2 = min(y1 + new_height, win_height)
            self.region = [x1,y1,x2,y2]
 
            if self.region==[0, 0, win_width, win_height]:
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
                    region = [
                        self.region[0] - offset_x,
                        self.region[1] - offset_y,
                        self.region[2] - offset_x,
                        self.region[3] - offset_y,
                    ]
                    x1,y1,x2,y2 = region
                    x1 = max(x1,0)
                    y1 = max(y1,0)
                    x2 = min(x1 + new_width, win_width)
                    y2 = min(y1 + new_height, win_height)
                    self.region = [x1,y1,x2,y2]
            
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.change_box_category()
        elif event == cv2.EVENT_RBUTTONDOWN:  # 删除(中心点或左上点)距离当前鼠标最近的框

            if len(self.boxes):
                if len(self.boxes) > 1:
                    current_point = np.array([(x-left) / new_unpad[0], (y-top) / new_unpad[1]])
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
       

    def _draw_box_on_image(self, image,dw,dh,new_unpad):
        boxes, classes = self.boxes, self.classes
        for box, cls in zip(boxes, classes):
            x1, y1 = (int(new_unpad[0] * box[0]+dw), int(new_unpad[1] * box[1]+dh))
            x2, y2 = (int(new_unpad[0] * box[2]+dw), int(new_unpad[1] * box[3]+dh))
            color = compute_color_for_labels(int(cls))
            box = [x1, y1, x2, y2, int(cls)]
            image = plt_bbox(image, box, self.thickness,box_color=color)
        
        return image



    def read_label_file(self, label_file_path):

        boxes = []
        classes = []

        annotation = read_txt(label_file_path)
        print(annotation)
        for bbox in annotation:
            bbox = list(map(float, bbox.split()))
            boxes.append(xywh2xyxy(bbox[1:]))
            classes.append(bbox[0])

        self.boxes = boxes
        self.classes = classes
        self.current_box_num = len(boxes)

    def write_label_file(self, label_file_path):
        ann_boxes = []
        for box, cls in zip(self.boxes, self.classes):
            box = list(map(str, box_fix(box)))
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

    
    def set_thickness(self,value):
        self.thickness = max(value, 1)
        
    def set_imgscale_mode(self,value):
        self.imgscale_mode = value
        self.getScaleInfo(False)
        
    def set_show_boxnum(self,value):
        self.show_boxnum = value
        
    def set_class_num(self,value):
        self.cls_num = value
        
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

    def getScaleInfo(self,only_check_winsize=False):
        
        *_, win_width, win_height = cv2.getWindowImageRect(self.windows_name)
        
        if win_width<=0 or win_height<0:
            return 0 
        if only_check_winsize and not self.win_info is None:
            if self.win_info[:2]==(win_width, win_height):
                return 1
            
        ori_h,ori_w,_=self.image.shape
        r  = min(win_width/ori_w, win_height/ori_h)
        if r>1 and self.imgscale_mode!=0:
            r=1
            
        if r>1:
            interpolation=cv2.INTER_CUBIC
        else:
            interpolation=cv2.INTER_AREA
        new_unpad = int(round(ori_w * r)), int(round(ori_h * r))
        
        dw, dh = win_width - new_unpad[0], win_height - new_unpad[1]
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        self.win_info = (win_width, win_height,left,top,right,bottom,new_unpad,interpolation)
        
        return 0
        
        
    def render(self):
        image = self.copy_image()
        (win_width, win_height,left,top,right,bottom,new_unpad,interpolation) =    self.win_info
        im = cv2.resize(image, new_unpad,interpolation=interpolation)
        image = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114,114,114])  # add border
        
        
        
        if not self.mouse_event is None:
            event,flags,x,y = self.mouse_event
            x, y = min(max(x, left), win_width-right), min(max(y, top), win_height-bottom)

             
            if event == cv2.EVENT_MOUSEMOVE:
                if flags == cv2.EVENT_FLAG_LBUTTON:
                    # 按住鼠标左键进行移动，画框
                    color = compute_color_for_labels(self.cur_class)
                    cv2.rectangle(image, (self.ix, self.iy), (x, y), color, self.thickness)
                
                else:
                    cv2.line(
                        image,
                        (x, top),
                        (x, win_height-bottom),
                        (255, 0, 0),
                        self.thickness,
                        8,
                    )
                    cv2.line(image, (left, y), (win_width-right, y), (255, 0, 0), self.thickness, 8)
            if self.show_boxnum==1:
                cv2.putText(image, f"{len(self.boxes)}", (min(x+10,win_width),max(0,y-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
             
        if not self.mouse_event is None or self.reload:
            self.mouse_event = None
               
            if self.show_label:
                self._draw_box_on_image(image,left,top,new_unpad)
                          
            cv2.imshow(self.windows_name, self._encode_image(image))
            self.reload=False   

    def run(self):

        print("Total Num: ", self.total_image_number)
        cv2.namedWindow(self.windows_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windows_name, 640, 480)
        
        config = config_window()
        
        visited_image = set()
        labeled_index, labeled_box = self.current_label_index, 0
        save_info = False
        init = True
        last_ms=0
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
                self.reload=True
                self.getScaleInfo(False)
                if self.vis_mode > 0:
                    self.set_mode(self.vis_mode)

                if os.path.exists(self.label_path):
                    self.read_label_file(self.label_path)

                print(
                    f"Img ID: {labeled_index}\nImg Path: {image_path}\nLabel Path: {self.label_path}\n"
                )

            self.getScaleInfo(True)
            cv2.setMouseCallback(self.windows_name, self._draw_roi)
            self.render()
            
                        
            
            cur_ms = int(time.perf_counter() * 1000)
            if cur_ms-last_ms<30:
                t = max(30-(cur_ms-last_ms),1)
                time.sleep(t/1000)
                
            last_ms = int(time.perf_counter() * 1000)

            
            key = cv2.waitKey(1)& 0xFF

            if key == ord("q") or key == ord("Q"):
                self.change_box_category(-1)
                
            elif key == ord("e") or key == ord("E"):
                self.change_box_category(1)

            elif key == ord("c") or key == ord("C"):
                boxes, classes = self.boxes.copy(), self.classes.copy()
                self.copy_box = [boxes, classes]

            elif key == ord("v") or key == ord("V"):
                boxes, classes = self.copy_box
                self.boxes, self.classes = boxes.copy(), classes.copy()
                
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
                init = True


            elif key == ord("n") or key == ord("N"):
                labeled_box = statistics_box_num(self.images_list)
                
            elif key == ord("h") or key == ord("H"):
                print(CLabeled._help)
                config.show(self)
                
                
            elif key == 27:  # exit
                self.write_label_file(self.label_path)
                break
            

def main():
    if len(sys.argv) > 1:
        image_folder = sys.argv[1]
    else:
        # image_folder = r'E:\project\data'
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
