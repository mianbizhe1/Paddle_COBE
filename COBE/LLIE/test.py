import cv2
import os.path as osp
import logging
import argparse
import paddle
import LLIE.options.options as option
import LLIE.utils.util as util
from LLIE.data import create_dataset, create_dataloader
from LLIE.models import create_model
import numpy as np

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='LOL_v2_syn.yml', help='Path to options YAML file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)


def main():
    # 创建模型
    print("创建模型")
    model = create_model(opt)
    print("加载数据集")
    # 加载数据集
    for phase, dataset_opt in opt['datasets'].items():
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt, opt, None)
    print("进行推理")
    # 进行推理
    for val_data in val_loader:
        model.feed_data(val_data)
        print("test之前")
        model.test()
        print("获取当前视觉效果")
        # 获取当前视觉效果
        visuals = model.get_current_visuals()
        print("准备写入")
        # 将结果张量转换为图像格式并保存
        rlt_img = util.tensor2img(visuals['rlt'])  # uint8
        cv2.imwrite("../temp/enhanced/enhanced.png", rlt_img)


if __name__ == '__main__':
    paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')  # 设置Paddle运行设备
    main()
