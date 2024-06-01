import os
import cv2
import numpy as np

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



def xywh2xyxy(xywh):
    """[x, y, w, h]转为[xmin, ymin, xmax, ymax]"""
    xmin = xywh[0] - xywh[2] / 2
    ymin = xywh[1] - xywh[3] / 2
    xmax = xywh[0] + xywh[2] / 2
    ymax = xywh[1] + xywh[3] / 2
    xyxy = [xmin, ymin, xmax, ymax]

    return xyxy


def box_fix(xyxy):
    x_center = float(xyxy[0] + xyxy[2]) / 2
    y_center = float(xyxy[1] + xyxy[3]) / 2
    width = abs(xyxy[2] - xyxy[0])
    height = abs(xyxy[3] - xyxy[1])
    xywh_center = [x_center, y_center, width, height]
    return xywh_center