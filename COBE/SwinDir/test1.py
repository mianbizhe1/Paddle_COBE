import argparse
import os

import numpy as np
import onnxruntime
import paddle
import requests

upscale = 4
window_size = 8
height = 560
width = 560


# 加载 ONNX 模型
model_path = "nasnet.onnx"
if os.path.exists(model_path):
    print(f'loading model from {model_path}')
else:
    os.makedirs(os.path.dirname("./" + model_path), exist_ok=True)
    url = 'https://github.com/mianbizhe1/Paddle_COBE/blob/master/COBE/{}'.format(
        os.path.basename(model_path))
    r = requests.get(url, allow_redirects=True)
    print(f'downloading model {model_path}')
    open(model_path, 'wb').write(r.content)

session = onnxruntime.InferenceSession(model_path)


# 准备输入数据（NumPy 数组）
input_data = np.random.rand(1, 3, height, width).astype(np.float32)  # 示例输入



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
    output = div(input_data, args, 8)
    print(output)


if __name__ == '__main__':
    main()
