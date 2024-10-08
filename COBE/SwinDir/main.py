import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import paddle
import requests
import base64
from COBE.SwinDir.models.network_swinir import SwinIR as net
from COBE.SwinDir.utils import util_calculate_psnr_ssim as util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='lightweight_sr', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car, color_jpeg_car')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=64, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pdparams')
    parser.add_argument('--folder_lq', type=str, default='../temp/enhanced', help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    args = parser.parse_args()

    device = paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')

    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)

    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)
    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # read image
        imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = paddle.to_tensor(img_lq).astype('float32').unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with paddle.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.shape
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = paddle.concat([img_lq, paddle.flip(img_lq, [2])], axis=2)[:, :, :h_old + h_pad, :]
            img_lq = paddle.concat([img_lq, paddle.flip(img_lq, [3])], axis=3)[:, :, :, :w_old + w_pad]
            output = ban(img_lq, model, args, window_size)
            output = output[..., :h_old * args.scale, :w_old * args.scale]

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

def ban(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.shape
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = paddle.zeros([b, c, h*sf, w*sf], dtype=img_lq.dtype)
        W = paddle.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[:, :, h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = paddle.ones_like(out_patch)

                E[:, :, h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[:, :, h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.divide(W)

    return output

if __name__ == '__main__':
    main()
