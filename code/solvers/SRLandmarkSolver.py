﻿import os
from collections import OrderedDict
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils  #chengjia
from tensorboardX import SummaryWriter #chengjia
# from networks.srfbn_hg_arch import FeedbackBlockCustom, FeedbackBlockHeatmapAttention, merge_heatmap_5


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as thutil
from torchvision.transforms import Normalize

from networks import create_model
from .base_solver import BaseSolver
from networks import init_weights
from utils import util


class SRLandmarkSolver(BaseSolver):
    def __init__(self, opt):
        super(SRLandmarkSolver, self).__init__(opt)
        self.train_opt = opt['solver']
        self.LR = self.Tensor()
        self.HR = self.Tensor()
        self.gt_heatmap = self.Tensor()
        self.SR = None
        self.hg_require_grad = True
        
        self.unnorm = Normalize(
            (-0.509 * opt['rgb_range'], -0.424 * opt['rgb_range'],
             -0.378 * opt['rgb_range']), (1.0, 1.0, 1.0))

        self.records = {
            'val_loss_pix': [],
            'val_loss_total': [],
            'psnr': [],
            'ssim': [],
            'lr': []
        }

        self.model = create_model(opt)

        if self.is_train:
            self.model.train()

            # set loss
            self.loss_dict = {}
            for k, v in self.train_opt['loss'].items():
                self.loss_dict[k] = {}
                loss_type = v['loss_type']
                if loss_type == 'l1':
                    self.loss_dict[k]['criterion'] = nn.L1Loss()
                elif loss_type == 'l2':
                    self.loss_dict[k]['criterion'] = nn.MSELoss()
                elif loss_type == 'AdapWingLoss':
                    self.loss_dict[k]['criterion'] = AdapWingLoss()
                else:
                    raise NotImplementedError(
                        '%d loss type [%s] is not implemented!' % (k,
                                                                   loss_type))
                self.loss_dict[k]['weight'] = v['weight']

            if self.use_gpu:
                for k in self.loss_dict.keys():
                    self.loss_dict[k]['criterion'] = self.loss_dict[k][
                        'criterion'].cuda()

            # set optimizer
            weight_decay = self.train_opt['weight_decay'] if self.train_opt[
                'weight_decay'] else 0
            optim_type = self.train_opt['type'].upper()
            if optim_type == "ADAM":
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.train_opt['learning_rate'],
                    weight_decay=weight_decay)
            else:
                raise NotImplementedError(
                    'Loss type [%s] is not implemented!' % optim_type)

            # set lr_scheduler
            if self.train_opt['lr_scheme'].lower() == 'multisteplr':
                self.scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer, self.train_opt['lr_steps'],
                    self.train_opt['lr_gamma'])
            else:
                raise NotImplementedError(
                    'Only MultiStepLR scheme is supported!')

        self.print_network()
        self.load()

        print(
            '===> Solver Initialized : [%s] || Use GPU : [%s]'
            % (self.__class__.__name__, self.use_gpu))
        if self.is_train:
            print("optimizer: ", self.optimizer)
            print("lr_scheduler milestones: %s   gamma: %f" %
                  (self.scheduler.milestones, self.scheduler.gamma))

    def _net_init(self, init_type='kaiming'):
        print('==> Initializing the network using [%s]' % init_type)
        init_weights(self.model, init_type)

    def feed_data(self, batch, need_HR=True, need_landmark=True):
        input = batch['LR']
        self.LR.resize_(input.size()).copy_(input)
        if need_HR:
            target = batch['HR']
            self.HR.resize_(target.size()).copy_(target)
        if need_landmark:
            gt_heatmap = batch['heatmap']

            self.gt_heatmap.resize_(gt_heatmap.size()).copy_(gt_heatmap)
            self.gt_landmark = batch['landmark']

    def train_step(self):

        self.model.train()
        self.optimizer.zero_grad()
        loss_pix = 0.0
        loss_align = 0.0

        SR_list, heatmap_list,merge_heatmap_list = self.model(self.LR)

        self.SR = SR_list
        self.heatmap_list = heatmap_list
        self.merge_heatmap= merge_heatmap_list
        self.SR = SR_list[-1]
        self.heatmap = heatmap_list[-1]
        self.merge_heatmap= merge_heatmap_list[-1]


        for step, SR in enumerate(SR_list):
            loss_pix += self.loss_dict['pixel']['weight'] * self.loss_dict[
                'pixel']['criterion'](SR, self.HR)
            loss_align += self.loss_dict['align']['weight'] * self.loss_dict[
                'align']['criterion'](heatmap_list[step], self.gt_heatmap)
        loss = loss_pix + loss_align    #


        loss.backward()
        self.optimizer.step()
        self.model.eval()

        return {
            'loss_pix': loss_pix.item(),
            'loss_align': loss_align.item(),
            'loss_total': loss.item()
        }

    def test(self):


        self.model.eval()
        with torch.no_grad():
            forward_func = self.model.forward
            SR_list, heatmap_list, merge_heatmap_list = forward_func(self.LR)


            self.SR_list = SR_list
            self.heatmap_list = heatmap_list
            self.merge_heatmap_list=merge_heatmap_list

            self.SR = SR_list[-1]
            self.heatmap = heatmap_list[-1]
            self.merge_heatmap=merge_heatmap_list[-1]

        self.model.train()
        if self.is_train:
            loss_pix = 0.0
            with torch.no_grad():
                for step, SR in enumerate(SR_list):
                    loss_pix += self.loss_dict['pixel']['weight'] * self.loss_dict[
                        'pixel']['criterion'](SR, self.HR)

                loss = loss_pix

            return {
                'loss_pix': loss_pix.item(),
                'loss_total': loss.item()
            } ,self.merge_heatmap


    def mod_HG_grad(self, requires_grad):
        if self.hg_require_grad != requires_grad:
            if isinstance(self.model, nn.DataParallel):
                for p in self.model.module.HG.parameters():

                    p.requires_grad=requires_grad
            else:
                for p in self.model.HG.parameters():

                    p.requires_grad=requires_grad
            self.hg_require_grad = requires_grad

    def calc_nme(self):
        '''
        calculate normalized mean error
        '''
        landmark = util.get_peak_points(self.heatmap.cpu().numpy()) * 4
        diff = landmark - self.gt_landmark.numpy()
        nme = util.calc_nme(landmark, self.gt_landmark.numpy())
        return nme
    
    def save_checkpoint(self, epoch, is_best):
        """
        save checkpoint to experimental dir
        """
        ckp = {
            'step': self.step,
            'epoch': epoch,
            'state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
            'best_step': self.best_step,
            'records': self.records
        }
        if is_best:
            filename = os.path.join(self.checkpoint_dir, 'best_ckp.pth')
            print('===> Saving best checkpoint (step %d) to [%s] ...]' % (self.step, filename))
        else:
            filename = os.path.join(self.checkpoint_dir, 'step_%07d_ckp.pth' % self.step)
            print('===> Saving last checkpoint to [%s] ...]' % filename)
        torch.save(ckp, filename)

    def load(self):
        """
        load or initialize network
        """
        def _load_func(m, d):
            if isinstance(m, nn.DataParallel):
                res = m.module.load_state_dict(d, strict=False) 
            else:
                res = m.load_state_dict(d, strict=False) 
            res_str = ''
            if len(res.missing_keys) != 0:
                res_str += 'missing_keys: ' + ', '.join(res.missing_keys)
            if len(res.unexpected_keys) != 0:
                res_str += 'unexpected_keys: ' + ', '.join(res.unexpected_keys)
            if len(res_str) == 0: # strictly fit
                res_str = 'Strictly loaded!'
            return res_str

        if (self.is_train
                and self.opt['solver']['pretrain']) or not self.is_train:
            model_path = self.opt['solver']['pretrained_path']
            if model_path is None:
                raise ValueError(
                    "[Error] The 'pretrained_path' does not declarate in *.json"
                )

            print('===> Loading model from [%s]...' % model_path)
            
            if self.is_train:
                raw_checkpoint = torch.load(model_path)
                if isinstance(raw_checkpoint, dict) and not isinstance(raw_checkpoint, OrderedDict):
                    if 'state_dict' in raw_checkpoint.keys():
                        checkpoint = raw_checkpoint['state_dict']
                    elif 'netG' in raw_checkpoint.keys():
                        checkpoint = raw_checkpoint['netG']
                    else:
                        raise KeyError("Model not in checkpoint keys: %s" % ', '.join(list(checkpoint.keys())))
                else: # state_dict
                    checkpoint = raw_checkpoint

                res_str = _load_func(self.model, checkpoint)
                print(res_str)

                if self.opt['solver']['pretrain'] == 'resume':
                    self.step = raw_checkpoint['step']
                    self.cur_epoch = raw_checkpoint['epoch']
                    self.optimizer.load_state_dict(raw_checkpoint['optimizer'])
                    self.best_pred = raw_checkpoint['best_pred']
                    self.best_step= raw_checkpoint['best_step']
                    self.records = raw_checkpoint['records']
            else:
                checkpoint = torch.load(model_path)
                if isinstance(checkpoint, dict) and not isinstance(checkpoint, OrderedDict):
                    if 'state_dict' in checkpoint.keys():
                        checkpoint = checkpoint['state_dict']
                    elif 'netG' in checkpoint.keys():
                        checkpoint = checkpoint['netG']
                    else:
                        raise KeyError("Model not in checkpoint keys: %s" % ', '.join(list(checkpoint.keys())))
                else: # state_dict
                    checkpoint = checkpoint

                res_str = _load_func(self.model, checkpoint)
                print(res_str)

        else:
            self._net_init()
            if 'HG_pretrained_path' in self.opt['solver'].keys():
                print('===> Loading Hourglass model from %s' %
                      self.opt['solver']['HG_pretrained_path'])
                HG_state_dict = torch.load(self.opt['solver']['HG_pretrained_path'])
                to_load = self.model.module.HG if isinstance(self.model, nn.DataParallel) \
                        else self.model.HG
                res_str = _load_func(to_load, HG_state_dict)
                print(res_str)
        if self.is_train:
            self.update_learning_rate()

    def get_current_visual(self, need_np=True, need_HR=True):
        """
        return LR image and SR list (HR) images
        """
        def _get_data(x):
            return self.unnorm(x[0]).data.float().cpu()
        out_dict = OrderedDict()
        out_dict['LR'] = _get_data(self.LR)
        if 'log_full_step' in self.opt['solver'].keys() and self.opt['solver']['log_full_step']:
            out_dict['SR'] = [_get_data(SR) for SR in self.SR_list]
        else:
            out_dict['SR'] = [_get_data(self.SR)]
            
        if need_np:
            out_dict['LR'] = util.Tensor2np(
                [out_dict['LR']], self.opt['rgb_range'])
            out_dict['SR'] = util.Tensor2np(
                out_dict['SR'], self.opt['rgb_range'])
        if need_HR:
            out_dict['HR'] = _get_data(self.HR)
            if need_np:
                out_dict['HR'] = util.Tensor2np([out_dict['HR']],
                                                self.opt['rgb_range'])[0]
        return out_dict

    def get_current_heatmap_pair(self):

        visuals = self.get_current_visual()
        HR = visuals['HR']
        SR = visuals['SR']
        heatmap_list = [np.squeeze(h.cpu().numpy()) for h in self.heatmap_list]
        heatmap_gt = np.squeeze(self.gt_heatmap.cpu().numpy())
        fig = util.plot_heatmap_compare(heatmap_list, heatmap_gt, SR, HR)
        return fig

    def get_current_landmark_pair(self):

        visuals = self.get_current_visual()
        HR = visuals['HR']
        SR = visuals['SR']
        landmark = [np.squeeze(np.array(util.get_peak_points(h.cpu().numpy()))) * 4 for h in self.heatmap_list]
        fig = util.plot_landmark_compare(landmark, SR, HR)
        return fig
    def save_current_visual(self, img_name):
        """
        save visual results for comparison
        """
        visuals = self.get_current_visual(need_np=False)
        visuals_list = [util.quantize(visuals['HR'].squeeze(0), self.opt['rgb_range'])]
        visuals_list.extend([
            util.quantize(s.squeeze(0), self.opt['rgb_range']) for s in visuals['SR']
        ])
        
        visual_images = torch.stack(visuals_list)
        visual_images = thutil.make_grid(visual_images, nrow=len(visuals_list), padding=5)
        visual_images = visual_images.byte().permute(1, 2, 0).numpy()
        save_dir = os.path.join(self.visual_dir, img_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        cv2.imwrite(
            os.path.join(save_dir, 'SR_step_%d.png' % (self.step)),
            visual_images[:, :, ::-1])

        fig = self.get_current_landmark_pair()
        fig.savefig(
            os.path.join(save_dir, 'Landmark_step_%d.png' % (self.step)))
        plt.close(fig)



        im = self.merge_heatmap[0].cuda().data.cpu().numpy()
        self.im = im
        # [c,h,w]->[h,w,c]
        im = np.transpose(self.im, [1, 2, 0])
        plt.figure()
        for i in range(1):
            ax = plt.subplot(1, 1, i + 1)
            plt.axis('off')
            plt.imshow(im[:, :, i])
        plt.savefig(os.path.join(save_dir, 'Fea_map_step_%d.png' % (self.step)))
        plt.close()

    def log_current_visual(self, tb_logger, img_name, current_step):
        """
        log visual results to tensorboard for comparison
        """
        visuals = self.get_current_visual(need_np=False)
        visuals_list = [util.quantize(visuals['HR'].squeeze(0), self.opt['rgb_range'])]
        visuals_list.extend([
            util.quantize(s.squeeze(0), self.opt['rgb_range']) for s in visuals['SR']
        ])
        
        visual_images = torch.stack(visuals_list)
        visual_images = thutil.make_grid(visual_images, nrow=len(visuals_list), padding=5, normalize=True, scale_each=True)
        tb_logger.add_image(
            img_name + '_SR', visual_images, global_step=current_step)

        fig = self.get_current_landmark_pair()
        tb_logger.add_figure(img_name + '_Landmark', fig, global_step=current_step)
        plt.close(fig)

    def get_current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def update_learning_rate(self):
        self.scheduler.step(self.step)
        if self.train_opt['release_HG_grad_step'] != None and self.step >= self.train_opt['release_HG_grad_step']:
            if self.hg_require_grad is not True:
                print("Release HG gradients!")
                self.mod_HG_grad(requires_grad=True)
        else:
            if self.hg_require_grad is not False:
                print("Fix HG gradients!")
                self.mod_HG_grad(requires_grad=False)

    def get_current_log(self):
        log = OrderedDict()
        log['step'] = self.step
        log['epoch'] = self.cur_epoch
        log['best_pred'] = self.best_pred
        log['best_step'] = self.best_step
        log['records'] = self.records
        return log

    def set_current_log(self, log):
        self.step = log['step']
        self.cur_epoch = log['epoch']
        self.best_pred = log['best_pred']
        self.best_step = log['best_step']
        self.records = log['records']

    def save_current_log(self):
        index = np.arange(start=1, stop=len(self.records['lr'])+1, step=1, dtype=int)
        index *= int(self.train_opt['val_freq'])
        data_frame = pd.DataFrame(
            data=self.records,
            index=index)
        data_frame.to_csv(
            os.path.join(self.records_dir, 'train_records.csv'),
            index_label='step')

    def print_network(self):
        """
        print network summary including module and number of parameters
        """
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel):
            net_struc_str = '{} - {}'.format(
                self.model.__class__.__name__,
                self.model.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.model.__class__.__name__)

        print("==================================================")
        print("===> Network Summary\n")
        net_lines = []
        line = s + '\n'
        # print(line)
        net_lines.append(line)
        line = 'Network structure: [{}], with parameters: [{:,d}]'.format(
            net_struc_str, n)
        print(line)
        net_lines.append(line)

        if self.is_train:
            with open(os.path.join(self.exp_root, 'network_summary.txt'),
                      'w') as f:
                f.writelines(net_lines)

        print("==================================================")
