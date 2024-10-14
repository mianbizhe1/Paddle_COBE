import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os

import onnxruntime
import paddle
import requests
import base64
from SwinDir.models.network_swinir import SwinIR as net
from SwinDir.utils import util_calculate_psnr_ssim as util

import os
import onnxruntime as ort
import numpy as np
import cv2
import glob
import argparse
import base64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='lightweight_sr', help='classical_sr, lightweight_sr, real_sr, '
                                                                           'gray_dn, color_dn, jpeg_car, color_jpeg_car')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8')  # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=64, help='patch size used in training SwinIR. '
                                                                            'Just used to differentiate two different settings in Table 2 of the paper. '
                                                                            'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str, default='nasnet.onnx')
    parser.add_argument('--folder_lq', type=str, default='../temp/enhanced', help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=560,
                        help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    args = parser.parse_args()
    # set up ONNX Runtime inference session
    if os.path.exists(args.model_path):
        print(f'loading ONNX model from {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(
            os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)
        print("none")

    # 创建 ONNX Runtime 推理 session
    # ort_session = ort.InferenceSession(args.model_path)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):

        try:
            # read image
            imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
            img_lq = paddle.to_tensor(img_lq, dtype='float32')
            img_lq = paddle.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]],
                                      (2, 0, 1))  # HCW-BGR 转 CHW-RGB
            img_lq = paddle.to_tensor(img_lq, dtype='float32').unsqueeze(0)  # CHW-RGB 转 NCHW-RGB

            # 获取原始图像尺寸
            _, _, h_old, w_old = img_lq.shape

            # 计算 padding 大小以使图像尺寸成为 window_size 的倍数
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old

            # 沿高度方向和宽度方向进行翻转并拼接
            img_lq = paddle.concat([img_lq, paddle.flip(img_lq, axis=2)], axis=2)[:, :, :h_old + h_pad, :]
            img_lq = paddle.concat([img_lq, paddle.flip(img_lq, axis=3)], axis=3)[:, :, :, :w_old + w_pad]
            print("开始推理")

            output = div(img_lq, args, window_size)
            print("div done")
            # output = output.run(None, {input_name: img_lq})

            print(output[0].shape)
            print(output)

            # Resize output to match the original image size
            output = output[..., :h_old * args.scale, :w_old * args.scale]
        except Exception as e:
            print(f"Error during inference for image {imgname}: {e}")
            continue

        # save image
        output = output.squeeze().astype('float32').clip(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'{save_dir}/enhanced.png', output)
        _, img_encoded = cv2.imencode('.png', output)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        img_base64_with_prefix = f'data:image/png;base64,{img_base64}'
        return img_base64_with_prefix


def define_model(args):
    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
    param_key_g = 'params'

    pretrained_model = paddle.load(args.model_path)
    model.set_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model)

    return model


def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    if args.task in ['classical_sr', 'lightweight_sr']:
        save_dir = f'../temp/enhanced'
        folder = args.folder_lq
        border = args.scale
        window_size = 8

    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    img_gt = None
    img_lq = cv2.imread(f'{args.folder_lq}/enhanced.png', cv2.IMREAD_COLOR).astype(
        np.float32) / 255.
    return imgname, img_lq, img_gt


def div(img_lq, args, window_size):
    model_path = "nasnet.onnx"
    ort_session = onnxruntime.InferenceSession(model_path)
    if isinstance(img_lq, paddle.Tensor):  # 如果是 Paddle Tensor
        img_lq = img_lq.numpy()  # 转换为 NumPy 数组

    try:
        if args.tile is None:
            # test the image as a whole
            input_name = ort_session.get_inputs()[0].name
            output = ort_session.run(None, {input_name: img_lq})
        else:
            # test the image tile by tile
            b, c, h, w = img_lq.shape
            tile = min(args.tile, h, w)
            print("tile: ", tile)
            assert tile % window_size == 0, "tile size should be a multiple of window_size"
            tile_overlap = args.tile_overlap
            sf = args.scale

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            E = paddle.zeros((b, c, h * sf, w * sf), dtype=img_lq.dtype)
            W = paddle.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img_lq[:, :, h_idx:h_idx + tile, w_idx:w_idx + tile]
                    if isinstance(img_lq, paddle.Tensor):  # 如果是 Paddle Tensor
                        img_lq = img_lq.numpy()  # 转换为 NumPy 数组
                    input_name = ort_session.get_inputs()[0].name
                    try:
                        out_patch = ort_session.run(None, {input_name: in_patch})[0]  # 注意这里
                    except Exception as e:
                        print(f"Error during inference for patch at ({h_idx}, {w_idx}): {e}")
                        continue  # Skip this patch on error
                    out_patch = paddle.to_tensor(out_patch)
                    out_patch_mask = paddle.ones_like(out_patch)

                    E[:, :, h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch)
                    W[:, :, h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf].add_(out_patch_mask)
            output = E.divide(W)

        return output

    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # 或者返回一个默认值


if __name__ == '__main__':
    main()
