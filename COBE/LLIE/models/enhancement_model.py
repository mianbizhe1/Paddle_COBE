import logging
from collections import OrderedDict

import paddle
import paddle.nn as nn
from paddle import DataParallel
print("paddle 部分完成")
import LLIE.models.networks as networks
import LLIE.models.lr_scheduler as lr_scheduler
from LLIE.models.base_model import BaseModel
from LLIE.models.loss import CharbonnierLoss, VGGLoss
from LLIE.models.archs.torch_rgbto import rgb_to_ycbcr, ycbcr_to_rgb

logger = logging.getLogger('base')

class enhancement_model(BaseModel):
    def __init__(self, opt):
        super(enhancement_model, self).__init__(opt)

        if opt['dist']:
            self.rank = paddle.distributed.get_rank()
        else:
            self.rank = -1  # non-distributed training
        train_opt = opt['train']

        # Define network and load pretrained models
        print("start load")
        self.netG = networks.define_G(opt)
        if opt['dist']:
            self.netG = paddle.DataParallel(self.netG)

        # Print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### Loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss()
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss()
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss()
            else:
                raise NotImplementedError(f'Loss type [{loss_type}] is not recognized.')
            self.l_pix_w = train_opt['pixel_weight']

            self.cri_pix_ill = nn.MSELoss(reduction='sum')
            self.cri_pix_ill2 = nn.MSELoss(reduction='sum')

            self.cri_vgg = VGGLoss()

            #### Optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if not v.stop_gradient:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning(f'Params [{k}] will not optimize.')

            self.optimizer_G = paddle.optimizer.Adam(
                learning_rate=train_opt['lr_G'],
                parameters=optim_params,
                weight_decay=wd_G,
                beta1=train_opt['beta1'],
                beta2=train_opt['beta2']
            )
            self.optimizers.append(self.optimizer_G)

            #### Schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer, train_opt['lr_steps'],
                            restarts=train_opt['restarts'],
                            weights=train_opt['restart_weights'],
                            gamma=train_opt['lr_gamma'],
                            clear_state=train_opt['clear_state']
                        )
                    )
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']
                        )
                    )
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs']
        self.nf = data['nf']
        if need_GT:
            self.real_H = data['GT']

    def set_params_lr_zero(self):
        # Fix normal module
        for group in self.optimizer_G._param_groups:
            group['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.clear_grad()
        dark = self.var_L
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = self.nf
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = paddle.abs(dark - light)

        mask = paddle.divide(light, noise + 0.0001)

        batch_size, _, height, width = mask.shape
        mask_max = paddle.max(mask.reshape([batch_size, -1]), axis=1).reshape([batch_size, 1, 1, 1])
        mask_max = mask_max.tile(repeat_times=[1, 1, height, width])
        mask = mask / (mask_max + 0.0001)

        mask = paddle.clip(mask, min=0, max=1.0)
        mask = mask.astype(paddle.float32)

        self.fake_H, self.fake_Amp, self.fake_H_s1, self.snr = self.netG(self.var_L)

        _, _, H, W = self.real_H.shape
        Y_real, Cb_real, Cr_real = rgb_to_ycbcr(self.real_H)
        tensor_cbcr_real = paddle.concat([Y_real, Cb_real, Cr_real], axis=1)
        image_fft = paddle.fft.fft2(tensor_cbcr_real, norm='backward')
        self.real_Amp = paddle.abs(image_fft)
        self.real_Pha = paddle.angle(image_fft)

        self.Cb_real = Cb_real
        self.Cr_real = Cr_real

        out_fft = paddle.fft.fft2(self.fake_H, norm='backward')
        self.fake_Pha = paddle.angle(out_fft)

        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_amp = self.l_pix_w * self.cri_pix(self.fake_Amp, self.real_Amp) * 0.01
        l_vgg = self.l_pix_w * self.cri_vgg(self.fake_H, self.real_H) * 0.1
        l_final = l_pix + l_amp + l_vgg
        l_final.backward()
        paddle.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=0.01)
        self.optimizer_G.step()

        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['l_amp'] = l_amp.item()
        self.log_dict['l_vgg'] = l_vgg.item()

    def test(self):
        self.netG.eval()
        print("1")
        with paddle.no_grad():
            self.var_L = self.var_L.astype(paddle.float32)  # 确保输入 tensor 是 float32 类型
            self.netG.eval()  # 切换网络到评估模式
            print("3")
            output = self.netG(self.var_L)  # 获取输出
            try:
                print("2")
                if isinstance(output, tuple) and len(output) == 4:
                    self.fake_H, self.fake_Amp, self.fake_H_s1, self.snr = output
                else:
                    print(output)
                    # 如果只返回一个值，则直接赋值给 fake_H
                    self.fake_H = output
                    self.fake_Amp = None
                    self.fake_H_s1 = None
                    self.snr = None
            except Exception as e:
                print(len(output))
                print(f"Error during forward pass: {e}")
            self.netG.train()  # 切换回训练模式
        print("start train")
        self.netG.train()



    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()

        out_dict['LQ'] = self.var_L.cpu()
        out_dict['rlt'] = self.fake_H.cpu()
        out_dict['rlt_s1'] = self.fake_H_s1.cpu()
        out_dict['rlt2'] = self.nf.cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.cpu()

        del self.real_H, self.nf, self.var_L, self.fake_H
        paddle.device.cuda.empty_cache()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__, self.netG._layers.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info(f'Network G structure: {net_struc_str}, with parameters: {int(n):,d}')

            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info(f'Loading model for G [{load_path_G}] ...')
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
