import paddle
import cv2
import numpy as np
import argparse
import copy
import time
import os
import base64
from paddleocr import PaddleOCR


def allFilePath(rootPath, allFileList):
    # 读取文件夹内的文件，放到list
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            allFileList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFileList)


def letter_box(img, size=(640, 640)):
    # YOLO 前处理 letter_box操作
    h, w, _ = img.shape
    r = min(size[0] / h, size[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    new_img = cv2.resize(img, (new_w, new_h))
    left = int((size[1] - new_w) / 2)
    top = int((size[0] - new_h) / 2)
    img = cv2.copyMakeBorder(new_img, top, size[0] - new_h - top, left, size[1] - new_w - left, cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))
    return img, r, left, top


def pre_processing(img, img_size):
    img, r, left, top = letter_box(img, (img_size, img_size))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()  # bgr2rgb hwc2chw
    img = paddle.to_tensor(img).astype('float32') / 255.0
    img = img.unsqueeze(0)
    return img, r, left, top


def det_rec_plate(img, img_ori, ocr_model):
    result_list = []
    result = ocr_model.ocr(img, cls=True)

    for line in result:
        # 提取车牌信息
        for word_info in line:
            box = word_info[0]
            text = word_info[1][0]  # 识别出的文本
            score = word_info[1][1]  # 置信度

            # 转换成整型坐标
            rect = [int(coord) for sublist in box for coord in sublist]
            roi_img = img_ori[rect[1]:rect[5], rect[0]:rect[2]]
            result_dict = {
                'plate_no': text,
                'rect': rect,
                'detect_conf': score,
                'roi_height': roi_img.shape[0],
                'plate_type': 0  # 设置为0，实际应用中可根据需要设置
            }
            result_list.append(result_dict)

    return result_list


def draw_result(orgimg, dict_list):
    result_str = ""
    for result in dict_list:
        rect_area = result['rect']
        cv2.rectangle(orgimg, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[5]), (0, 0, 255), 2)  # 画框

        result_p = result['plate_no']
        result_str += result_p + " "

        labelSize = cv2.getTextSize(result_p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        if rect_area[0] + labelSize[0][0] > orgimg.shape[1]:
            rect_area[0] = int(orgimg.shape[1] - labelSize[0][0])

        orgimg = cv2.putText(orgimg, result_p, (rect_area[0], int(rect_area[1])),
                             cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    print(result_str)
    print("draw success")
    return orgimg


parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default=r'../temp/enhanced', help='source')  # 待识别图片路径
parser.add_argument('--output', type=str, default='../temp/recognition', help='source')  # 结果保存的文件夹
parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')  # 输入大小
args = parser.parse_args()


def main():
    save_path = args.output
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 加载 PaddleOCR 模型
    ocr_model = PaddleOCR(use_angle_cls=True, lang='ch')  # 'ch' 为中文识别，如果需要其他语言可更改

    file_list = []
    allFilePath(args.image_path, file_list)
    for pic_ in file_list:
        img = cv2.imread(pic_)
        img_ori = copy.deepcopy(img)
        result_list = det_rec_plate(img, img_ori, ocr_model)

        # 将结果画在图上
        ori_img = draw_result(img, result_list)
        save_img_path = os.path.join(save_path, "recgnition.png")  # 图片保存的路径

        # 保存处理后的图片
        cv2.imwrite(save_img_path, ori_img)
        print()
        # 编码为Base64
        _, img_encoded = cv2.imencode('.png', ori_img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        img_base64_with_prefix = f'data:image/png;base64,{img_base64}'

        # 你可以在这里返回或打印图像和车牌号
        print(result_list)
        return img_base64_with_prefix, result_list[0]['plate_no']


if __name__ == '__main__':
    main()
