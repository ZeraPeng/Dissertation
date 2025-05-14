import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import traceback
import sys
from data_cnn60 import AverageMeter, NTUDataLoaders
from s_model import (MLP, Decoder, Discriminator, Encoder, KL_divergence,
                   permute_dims, reparameterize, fuse_logits)

from model.get_part_feature import ModelMatch, SHIFTGCNModel
import ipdb
import logging
from util_char import *


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def parse_arg():
    # Arg Parser
    parser = argparse.ArgumentParser(description='View adaptive')
    parser.add_argument('--ss', type=int, help="split size")
    parser.add_argument('--st', type=str, help="split type")
    parser.add_argument('--dataset_path', type=str, help="dataset path")
    parser.add_argument('--dataset', type=str, help="dataset name  ")
    parser.add_argument('--wdir', type=str,
                        help="directory to save weights path")
    parser.add_argument('--le', type=str, help="language embedding model")
    parser.add_argument('--ve', type=str, help="visual embedding model")
    parser.add_argument('--phase', type=str, help="train or val")
    parser.add_argument('--num_classes', type=int, help="total classes")
    parser.add_argument('--num_cycles', type=int, help="no of cycles")
    parser.add_argument('--num_epoch_per_cycle', type=int,
                        help="number_of_epochs_per_cycle")
    parser.add_argument('--lr', type=float,
                        help="learning rate", default=0.0001)
    parser.add_argument('--latent_size', type=int, help="Latent dimension")
    parser.add_argument('--i_latent_size', type=int, required=True,
                        help="Instance Style Latent dimension")
    parser.add_argument('--mode', type=str, help="Mode")
    parser.add_argument('--load_epoch', type=int,
                        help="load epoch", default=None)
    parser.add_argument('--load_classifier', action='store_true')
    parser.add_argument('--load_vae', action='store_true')
    parser.add_argument('--tm', type=str, help='text mode')
    parser.add_argument("--batch_size", type=int,
                        default=64, help='batch size')
    parser.add_argument("--dis_step", type=int, default=10, help='dis step')

    parser.add_argument("--beta_x", type=float, default=None)
    parser.add_argument("--beta_y", type=float, default=None)

    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--alpha_p", type=float, default=0.5)

    parser.add_argument("--body_part", type=int, default=6)

    parser.add_argument("--askg_mode", type=str, default='vocab')   # or vanilla

    args = parser.parse_args()
    return args


args = parse_arg()
ss = args.ss
st = args.st
dataset = args.dataset
dataset_path = args.dataset_path
wdir = args.wdir
le = args.le
phase = args.phase
num_classes = args.num_classes
num_epochs = args.num_cycles
cycle_length = args.num_epoch_per_cycle
semantic_latent_size = args.latent_size
style_latent_size = args.i_latent_size
load_epoch = args.load_epoch
mode = args.mode
load_classifier = args.load_classifier
load_vae = args.load_vae
tm = args.tm
batch_size = args.batch_size
alpha = args.alpha
alpha_p = args.alpha_p
body_part = args.body_part

assert (args.beta_x is None and args.beta_y is None) or (
    args.beta_x is not None and args.beta_y is not None), "Both beta_x and beta_y should be provided or None"


def get_text_data(text_emb, target):
    target = target.to(text_emb.device)
    return text_emb[target]


def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_models(load_epoch, sequence_encoder, sequence_decoder, text_encoder, text_decoder):
    se_checkpoint = f'{wdir}/{le}/{tm}/se_{str(load_epoch)}.pth.tar'
    sd_checkpoint = f'{wdir}/{le}/{tm}/sd_{str(load_epoch)}.pth.tar'
    te_checkpoint = f'{wdir}/{le}/{tm}/te_{str(load_epoch)}.pth.tar'
    td_checkpoint = f'{wdir}/{le}/{tm}/td_{str(load_epoch)}.pth.tar'

    sequence_encoder.load_state_dict(torch.load(se_checkpoint)['state_dict'])
    sequence_decoder.load_state_dict(torch.load(sd_checkpoint)['state_dict'])
    text_encoder.load_state_dict(torch.load(te_checkpoint)['state_dict'])
    text_decoder.load_state_dict(torch.load(td_checkpoint)['state_dict'])


def train_one_cycle(cycle_num,
                    sequence_encoder, sequence_decoder, text_encoder, text_decoder, discriminator,
                    optimizer, dis_optimizer,
                    train_loader, device, text_emb):  # 0-10, 1700
    # text_emp.shape: [60, 4, 512]
    dis_step = args.dis_step
    # Loss
    mse_criterion = nn.MSELoss().to(device)
    bce_criterion = nn.BCELoss().to(device)
    cr_fact_iter = int(0.8 * len(train_loader))
    beta_iter = int(len(train_loader) / 3)
    for i, (global_feats, part_feats, target) in enumerate(train_loader):
        inputs = global_feats
        losses = AverageMeter()
        ce_loss_vals = []

        # models. set to training mode
        sequence_encoder.train()
        sequence_decoder.train()
        text_encoder.train()
        text_decoder.train()

        # hyper params. (for losses for the VAEs) beta_x: skeleton; beta_y: text
        if args.beta_x is None and args.beta_y is None:
            kld_loss_factor = max(
                (0.1 * (i - (len(train_loader) / 1700 * 1000)) / (len(train_loader) / 1700 * 3000)), 0)
            kld_loss_factor_2 = max(
                (0.1 * (i - cr_fact_iter) / (len(train_loader) / 1700 * 3000)), 0) * (cycle_num > 1)
        else:
            if i <= beta_iter:
                kld_loss_factor = 0
                kld_loss_factor_2 = 0
            else:
                kld_loss_factor = 1.5 * \
                    (float(i) / len(train_loader) - 1/3) * args.beta_x
                kld_loss_factor_2 = 1.5 * \
                    (float(i) / len(train_loader) - 1/3) * args.beta_y

        cross_alignment_loss_factor = 1 * (i > cr_fact_iter)
        
        _, channels, _ = text_emb.shape
        s = inputs.to(device, non_blocking=True)        # torch.Size([32, 256])
        s = s.repeat_interleave(channels, dim=0)      # Shape: [128, 256]

        t = target.to(device, non_blocking=True)
        t = get_text_data(text_emb, t).to(device, non_blocking=True)    # torch.Size([32, 4, 512])
        t = t.reshape(-1, 512)  # torch.Size([128, 512])
        t = t.to(dtype=list(text_encoder.parameters())[0].dtype)

        smu, slv, ismu, islv = sequence_encoder(s, instance_style=True, type=type)      
        sz = reparameterize(smu, slv)   # [128,96]
        isz = reparameterize(ismu, islv)    # [128,8]
        sout = sequence_decoder(torch.cat([sz, isz], dim=-1))   # [128, 256]

        tmu, tlv = text_encoder(t)
        tz = reparameterize(tmu, tlv)
        tout = text_decoder(tz)
        
        sfromt = sequence_decoder(torch.cat([tz, isz], dim=-1))
        tfroms = text_decoder(sz)

        # ELBO Loss
        loss_rss = mse_criterion(s, sout)
        loss_rtt = mse_criterion(t, tout)
        loss_kld_s = KL_divergence(smu, slv).to(device)
        loss_kld_is = KL_divergence(ismu, islv).to(device)
        loss_kld_t = KL_divergence(tmu, tlv).to(device)

        # Cross Alignment Loss
        loss_rst = mse_criterion(s, sfromt)
        loss_rts = mse_criterion(t, tfroms)

        # MI Loss, minimizes the mutual information between isz and sz
        # ref: https://github.com/uqzhichen/SDGZSL/blob/b9dba96d536b69ddbf03b1eff27f62c280c518f8/train.py#L174C9-L174C9
        trained_dis = False
        dis_step -= 1
        if dis_step == 0:
            dis_step = args.dis_step
            discriminator.train()
            # gen targets
            B = sz.shape[0]
            ones = torch.ones(B, 1).to(sz.device)
            zeros = torch.zeros(B, 1).to(sz.device)

            # train discriminator with skeleton branch
            dis_sz = reparameterize(smu, slv)
            dis_isz = reparameterize(ismu, islv)
            original_batch = torch.cat([dis_sz, dis_isz], dim=-1)

            perm_sz, perm_isz = permute_dims(dis_sz, dis_isz)
            perm_batch = torch.cat([perm_sz, perm_isz], dim=-1)

            original_batch_pred = discriminator(original_batch)
            perm_batch_pred = discriminator(perm_batch)
            loss_s_dis = (bce_criterion(original_batch_pred, ones) +
                        bce_criterion(perm_batch_pred, zeros)) / 2

            # train discriminator with text branch
            dis_tz = reparameterize(tmu, tlv)
            dis_isz = reparameterize(ismu, islv)
            original_batch = torch.cat([dis_tz, dis_isz], dim=-1)

            perm_tz, perm_isz = permute_dims(dis_tz, dis_isz)
            perm_batch = torch.cat([perm_tz, perm_isz], dim=-1)

            original_batch_pred = discriminator(original_batch)
            perm_batch_pred = discriminator(perm_batch)
            loss_t_dis = (bce_criterion(original_batch_pred, ones) +
                        bce_criterion(perm_batch_pred, zeros)) / 2

            loss_dis = (loss_s_dis + loss_t_dis) / 2
            scaled_loss_dis = kld_loss_factor_2 * loss_dis
            dis_optimizer.zero_grad()
            scaled_loss_dis.backward(retain_graph=True)
            dis_optimizer.step()

            acc_dis = float(torch.sum(original_batch_pred > 0.5) +
                            torch.sum(perm_batch_pred < 0.5)) / (2 * B)
            trained_dis = True

        discriminator.eval()
        original_batch = torch.cat([sz, isz], dim=-1)
        loss_tc = torch.mean(discriminator(original_batch))
        scaled_loss_tc = loss_tc * kld_loss_factor_2

        loss = loss_rss + loss_rtt
        loss -= kld_loss_factor * (loss_kld_s + loss_kld_is) + \
            kld_loss_factor_2 * loss_kld_t
        loss += cross_alignment_loss_factor * (loss_rst + loss_rts)
        loss += scaled_loss_tc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        ce_loss_vals.append(loss.cpu().detach().numpy())

        log_dict = {
            "factors/kld_loss_factor": kld_loss_factor,
            "factors/kld_loss_factor_2": kld_loss_factor_2,
            "factors/cross_alignment_loss_factor": cross_alignment_loss_factor,
            "factors/cycle_num": cycle_num,

            'train_vae/loss': losses.val,
            'train_vae/s_recons': loss_rss.item(),
            'train_vae/t_recons': loss_rtt.item(),
            'train_vae/s_kld': loss_kld_s.item(),
            'train_vae/is_kld': loss_kld_is.item(),
            'train_vae/t_kld': loss_kld_t.item(),
            'train_vae/s_crecons': loss_rst.item(),
            'train_vae/t_crecons': loss_rts.item(),
            'train_vae/tc_loss': loss_tc.item(),
        }
        if trained_dis:
            log_dict.update({
                'train_vae/dis_loss': loss_dis.item(),
                'train_vae/dis_acc': acc_dis
            })
    return log_dict

def train_one_cycle_with_zsl(cycle_num,
                    sequence_encoder, sequence_decoder, text_encoder, text_decoder, discriminator,
                    optimizer, dis_optimizer,
                    train_loader, device, text_emb,
                    zsl_loader=None, unseen_inds=None, unseen_text_emb=None):  # Add ZSL parameters
    # text_emp.shape: [60, 4, 512]
    dis_step = args.dis_step
    # Loss
    mse_criterion = nn.MSELoss().to(device)
    bce_criterion = nn.BCELoss().to(device)
    ce_criterion = nn.CrossEntropyLoss().to(device)
    cr_fact_iter = int(0.8 * len(train_loader))
    beta_iter = int(len(train_loader) / 3)
    
    # Initialize ZSL variables if data is provided
    zsl_enabled = zsl_loader is not None and unseen_inds is not None and unseen_text_emb is not None
    zsl_alignment_loss_total = 0
    zsl_batch_count = 0

    zsl_iter = iter(zsl_loader)
    
    for i, (global_feats, part_feats, target) in enumerate(train_loader):
        inputs = global_feats
        losses = AverageMeter()
        ce_loss_vals = []

        # models. set to training mode
        sequence_encoder.train()
        sequence_decoder.train()
        text_encoder.train()
        text_decoder.train()

        # hyper params. (for losses for the VAEs) beta_x: skeleton; beta_y: text
        if args.beta_x is None and args.beta_y is None:
            kld_loss_factor = max(
                (0.1 * (i - (len(train_loader) / 1700 * 1000)) / (len(train_loader) / 1700 * 3000)), 0)
            kld_loss_factor_2 = max(
                (0.1 * (i - cr_fact_iter) / (len(train_loader) / 1700 * 3000)), 0) * (cycle_num > 1)
        else:
            if i <= beta_iter:
                kld_loss_factor = 0
                kld_loss_factor_2 = 0
            else:
                kld_loss_factor = 1.5 * \
                    (float(i) / len(train_loader) - 1/3) * args.beta_x
                kld_loss_factor_2 = 1.5 * \
                    (float(i) / len(train_loader) - 1/3) * args.beta_y

        cross_alignment_loss_factor = 1 * (i > cr_fact_iter)
        
        _, channels, _ = text_emb.shape
        s = inputs.to(device, non_blocking=True)        # torch.Size([32, 256])
        s = s.repeat_interleave(channels, dim=0)      # Shape: [128, 256]

        t = target.to(device, non_blocking=True)
        t = get_text_data(text_emb, t).to(device, non_blocking=True)    # torch.Size([32, 4, 512])
        t = t.reshape(-1, 512)  # torch.Size([128, 512])
        t = t.to(dtype=list(text_encoder.parameters())[0].dtype)

        smu, slv, ismu, islv = sequence_encoder(s, instance_style=True, type=type)      
        sz = reparameterize(smu, slv)   # [128,96]
        isz = reparameterize(ismu, islv)    # [128,8]
        sout = sequence_decoder(torch.cat([sz, isz], dim=-1))   # [128, 256]

        tmu, tlv = text_encoder(t)
        tz = reparameterize(tmu, tlv)
        tout = text_decoder(tz)
        
        sfromt = sequence_decoder(torch.cat([tz, isz], dim=-1))
        tfroms = text_decoder(sz)

        # ELBO Loss
        loss_rss = mse_criterion(s, sout)
        loss_rtt = mse_criterion(t, tout)
        loss_kld_s = KL_divergence(smu, slv).to(device)
        loss_kld_is = KL_divergence(ismu, islv).to(device)
        loss_kld_t = KL_divergence(tmu, tlv).to(device)

        # Cross Alignment Loss
        loss_rst = mse_criterion(s, sfromt)
        loss_rts = mse_criterion(t, tfroms)

        # MI Loss, minimizes the mutual information between isz and sz
        # ref: https://github.com/uqzhichen/SDGZSL/blob/b9dba96d536b69ddbf03b1eff27f62c280c518f8/train.py#L174C9-L174C9
        trained_dis = False
        dis_step -= 1
        if dis_step == 0:
            dis_step = args.dis_step
            discriminator.train()
            # gen targets
            B = sz.shape[0]
            ones = torch.ones(B, 1).to(sz.device)
            zeros = torch.zeros(B, 1).to(sz.device)

            # train discriminator with skeleton branch
            dis_sz = reparameterize(smu, slv)
            dis_isz = reparameterize(ismu, islv)
            original_batch = torch.cat([dis_sz, dis_isz], dim=-1)

            perm_sz, perm_isz = permute_dims(dis_sz, dis_isz)
            perm_batch = torch.cat([perm_sz, perm_isz], dim=-1)

            original_batch_pred = discriminator(original_batch)
            perm_batch_pred = discriminator(perm_batch)
            loss_s_dis = (bce_criterion(original_batch_pred, ones) +
                        bce_criterion(perm_batch_pred, zeros)) / 2

            # train discriminator with text branch
            dis_tz = reparameterize(tmu, tlv)
            dis_isz = reparameterize(ismu, islv)
            original_batch = torch.cat([dis_tz, dis_isz], dim=-1)

            perm_tz, perm_isz = permute_dims(dis_tz, dis_isz)
            perm_batch = torch.cat([perm_tz, perm_isz], dim=-1)

            original_batch_pred = discriminator(original_batch)
            perm_batch_pred = discriminator(perm_batch)
            loss_t_dis = (bce_criterion(original_batch_pred, ones) +
                        bce_criterion(perm_batch_pred, zeros)) / 2

            loss_dis = (loss_s_dis + loss_t_dis) / 2
            scaled_loss_dis = kld_loss_factor_2 * loss_dis
            dis_optimizer.zero_grad()
            scaled_loss_dis.backward(retain_graph=True)
            dis_optimizer.step()

            acc_dis = float(torch.sum(original_batch_pred > 0.5) +
                            torch.sum(perm_batch_pred < 0.5)) / (2 * B)
            trained_dis = True

        discriminator.eval()
        original_batch = torch.cat([sz, isz], dim=-1)
        loss_tc = torch.mean(discriminator(original_batch))
        scaled_loss_tc = loss_tc * kld_loss_factor_2

        # Origin VAE loss
        loss = loss_rss + loss_rtt
        loss -= kld_loss_factor * (loss_kld_s + loss_kld_is) + \
            kld_loss_factor_2 * loss_kld_t
        loss += cross_alignment_loss_factor * (loss_rst + loss_rts)
        loss += scaled_loss_tc

        # Prepare text features for ZSL
        if zsl_enabled:
            t_z_list = []
            with torch.no_grad():
                text_encoder.eval()
                n_t = unseen_text_emb.to(device).float()
                a, b, _ = n_t.shape
                n_t = n_t.reshape(-1, n_t.shape[-1])
                t_tmu, t_tlv = text_encoder(n_t)
                t_z = reparameterize(t_tmu, t_tlv)
                t_z = t_z.reshape(a, b, t_z.shape[-1])
                t_z_list.append(t_z)
        # ZSL alignment loss integration
        current_zsl_loss = torch.tensor(0.0, device=device)
        if zsl_enabled:
            try:
                zsl_global_feats, zsl_part_feats, zsl_target = next(zsl_iter)
            except StopIteration:
                zsl_iter = iter(zsl_loader)
                zsl_global_feats, zsl_part_feats, zsl_target = next(zsl_iter)
            
            target_mapped = map_target(unseen_inds, zsl_target.to(device), device)
            
            zsl_s = zsl_global_feats.to(device)
            with torch.no_grad():
                nt_smu, _ = sequence_encoder(zsl_s)
                score_alignment = compute_alignment_score(nt_smu, t_z_list[0])
            
            current_zsl_loss = ce_criterion(score_alignment, target_mapped)
            zsl_alignment_loss_total += current_zsl_loss.item()
            zsl_batch_count += 1
            zsl_loss_weight = 0.3
            loss += zsl_loss_weight * current_zsl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        ce_loss_vals.append(loss.cpu().detach().numpy())

        log_dict = {
            "factors/kld_loss_factor": kld_loss_factor,
            "factors/kld_loss_factor_2": kld_loss_factor_2,
            "factors/cross_alignment_loss_factor": cross_alignment_loss_factor,
            "factors/cycle_num": cycle_num,

            'train_vae/loss': losses.val,
            'train_vae/s_recons': loss_rss.item(),
            'train_vae/t_recons': loss_rtt.item(),
            'train_vae/s_kld': loss_kld_s.item(),
            'train_vae/is_kld': loss_kld_is.item(),
            'train_vae/t_kld': loss_kld_t.item(),
            'train_vae/s_crecons': loss_rst.item(),
            'train_vae/t_crecons': loss_rts.item(),
            'train_vae/tc_loss': loss_tc.item(),
        }
        
        # Add ZSL logs
        if zsl_enabled and zsl_batch_count > 0:
            log_dict.update({
                'train_vae/zsl_alignment_loss': zsl_alignment_loss_total / zsl_batch_count,
                'train_vae/current_zsl_loss': current_zsl_loss.item()
            })
            
        if trained_dis:
            log_dict.update({
                'train_vae/dis_loss': loss_dis.item(),
                'train_vae/dis_acc': acc_dis
            })
    
    # Evaluate ZSL performance at the end of each cycle if needed
    evaluate_zsl_each_cycle = False
    if zsl_enabled and evaluate_zsl_each_cycle:
        with torch.no_grad():
            names = ['model']
            vae_dict = {
                'model': {
                    'sequence_encoder': sequence_encoder,
                    'text_encoder': text_encoder
                }
            }
            alpha = 1.0
            alpha_p = 1.0
            zsl_accuracy = zsl_test_for_vae(names, vae_dict, zsl_loader, None, unseen_inds, unseen_text_emb, alpha, alpha_p, device)
            log_dict.update({
                'eval/zsl_accuracy': zsl_accuracy
            })
    return log_dict

def save_model(epoch, sequence_encoder, sequence_decoder, text_encoder, text_decoder, optimizer):
    se_checkpoint = f'{wdir}/{le}/{tm}/se_{str(epoch)}.pth.tar'
    sd_checkpoint = f'{wdir}/{le}/{tm}/sd_{str(epoch)}.pth.tar'
    te_checkpoint = f'{wdir}/{le}/{tm}/te_{str(epoch)}.pth.tar'
    td_checkpoint = f'{wdir}/{le}/{tm}/td_{str(epoch)}.pth.tar'

    save_checkpoint({'epoch': epoch + 1,
                    'state_dict': sequence_encoder.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, se_checkpoint)
    save_checkpoint({'epoch': epoch + 1,
                    'state_dict': sequence_decoder.state_dict(),
                    }, sd_checkpoint)
    save_checkpoint({'epoch': epoch + 1,
                    'state_dict': text_encoder.state_dict(),
                    }, te_checkpoint)
    save_checkpoint({'epoch': epoch + 1,
                    'state_dict': text_decoder.state_dict(),
                    }, td_checkpoint)

def save_all_model(epoch, part_models):
    part_models_checkpoint = f'{wdir}/{le}/{tm}/{str(epoch)}_vae_models.pth.tar'
    model_checkpoints = {}
    for part_name, model_dict in part_models.items():
        model_checkpoints[part_name] = {}
        for name, model in model_dict.items():
            if name in ['sequence_encoder', 'sequence_decoder', 'text_encoder', 'text_decoder', 'optimizer']:
                model_checkpoints[part_name][name] = model.state_dict()
    save_checkpoint({'epoch': epoch + 1,
                     'state_dict': model_checkpoints}, part_models_checkpoint)

def map_target(u_ids, target, device):
    t = target
    t = t.cpu().numpy()
    position_map = {value: index for index, value in enumerate(u_ids)}
    target_mapped = [position_map[item] for item in t] 
    target_mapped = torch.tensor(target_mapped).to(device)
    return target_mapped


def zsl_alignment(names, vae_dict, zsl_loader, val_loader, unseen_inds, unseen_text_emb, alpha, alpha_p, device, logging=logging):
    if len(names) == 1:
        alpha = 1.0
        alpha_p = 1.0
    beta = (1.0 - alpha)/len(names)
    beta_p = (1.0 - alpha_p)/len(names)

    loss_weights = {}
    pred_weights = {}
    for i, name in enumerate(names):
        if i == 0:
            loss_weights[name] = alpha
            pred_weights[name] = alpha_p 
        else:
            loss_weights[name] = beta
            pred_weights[name] = beta_p


    # prepare target and stext features
    y = torch.tensor(range(ss)).to(device)      # ss=5 here

    t_z_list = []
    for i, name in enumerate(vae_dict):
        text_encoder = vae_dict[name]['text_encoder']
        text_encoder.eval()
        n_t = unseen_text_emb[i].to(device).float()        # 5, 512
        a, b, _ = n_t.shape
        n_t = n_t.reshape(-1, n_t.shape[-1])
        t_tmu, t_tlv = text_encoder(n_t)
        t_z = reparameterize(t_tmu, t_tlv) 
        t_z = t_z.reshape(a, b, t_z.shape[-1])     # t_z.shape: torch.Size([5, 4, 96])
        
        t_z_list.append(t_z)        # [num_types(xaa, xao), [num_classes, num_prompts, feature_dim]]

    criterion = nn.CrossEntropyLoss().to(device) 

    logging.info('Starting training stage...')
    c_loss = {}
    for i, name in enumerate(names):  
        c_loss[name] = 0

    count = 0
    num = 0
    final_preds = []
    tars = []
    pred_stream_dict = {}
    pred_stream_count = {}
    u_inds = torch.from_numpy(unseen_inds)
    for name in names:
        pred_stream_count[name] = 0
    for (global_feats, part_feats, target) in zsl_loader:    # inp: data of current patch. target: ground truth
        pred_t_dict = {}
        confidence_dict = {}
        stream_t_pred = {}
        target_mapped = map_target(unseen_inds, target, device)
        score_alignment = [0 for i in range(0, len(names))]
        loss_alignment = [0 for i in range(0, len(names))]
        for i, name in enumerate(names):
            t_s = global_feats.to(device)
            sequence_encoder = vae_dict[name]['sequence_encoder']
            nt_smu, t_slv = sequence_encoder(t_s)
            
            score_alignment[i] = compute_alignment_score(nt_smu, t_z_list[i])
            loss_alignment[i] = criterion(score_alignment[i], target_mapped)

            pred_t_dict[name], confidence_dict[name] = predict_from_alignment_scores(score_alignment[i])

        weighted_t_pred = sum(pred_t_dict[name] * pred_weights[name] for name in names)
        fused_t_pred = torch.round(weighted_t_pred).long()          
        fused_t_pred = fused_t_pred.cpu()
        final_preds.append(u_inds[fused_t_pred])
        tars.append(target)
        count += torch.sum(u_inds[fused_t_pred] == target)
        num += len(target)

        for i, name in enumerate(names):
            pred_stream_dict[name] = []
            stream_t_pred[name] = torch.round(pred_t_dict[name]).long()
            stream_t_pred[name] = stream_t_pred[name].cpu()
            pred_stream_dict[name].append(u_inds[stream_t_pred[name]])
            pred_stream_count[name] += torch.sum(u_inds[stream_t_pred[name]] == target)
    
    zsl_acc_list = []
    zsl_accuracy = float(count)/num
    zsl_acc_list.append(zsl_accuracy)
    zsl_stream_acc_dict = {}
    logging.info(f'=== total acc: {zsl_accuracy} ===')
    zsl_stream_acc_dict['total'] = zsl_accuracy
    for i, name in enumerate(names):
        zsl_acc_stream = float(pred_stream_count[name])/num
        logging.info(f"{name}: {zsl_acc_stream}")
        zsl_acc_list.append(zsl_acc_stream)
        zsl_stream_acc_dict[name] = zsl_acc_stream
    zsl_acc_best = max(zsl_acc_list)
    return zsl_stream_acc_dict, zsl_acc_best

def train_classifier(names, vae_dict, zsl_loader, val_loader, unseen_inds, unseen_text_emb, alpha, alpha_p, device):
    if len(names) == 1:
        alpha = 1.0
        alpha_p = 1.0
    beta = 1.0 - alpha
    beta_p = 1.0 - alpha_p

    loss_weights = {}
    pred_weights = {}
    for i, name in enumerate(names):
        if i == 0:
            loss_weights[name] = alpha
            pred_weights[name] = alpha_p 
        else:
            loss_weights[name] = beta
            pred_weights[name] = beta_p

    # Init all the classifiers 
    clf_dict = {}
    clf_optimizer = {}
    for name in names:
        clf_dict[name] = MLP([semantic_latent_size,  ss]).to(device)
        clf_optimizer[name] = optim.Adam(clf_dict[name].parameters(), lr=0.001)

    if load_classifier == True:
        cls_checkpoint = f'{wdir}/{le}/{tm}/classifiers.pth.tar'
        clf_load_dict = torch.load(cls_checkpoint, weights_only=False)
    else:
        # use text features to train the classifier
        with torch.no_grad():   
            # Global classifier
            y = torch.tensor(range(ss)).to(device)      # ss=5 here
            y = y.repeat([500])

            t_z_list = []
            for i, name in enumerate(vae_dict):
                text_encoder = vae_dict[name]['text_encoder']
                text_encoder.eval()
                n_t = unseen_text_emb[i].to(device).float()        # 5, 512
                n_t = n_t.repeat([500, 1, 1])
                a, b, _ = n_t.shape
                n_t = n_t.reshape(-1, n_t.shape[-1])
                t_tmu, t_tlv = text_encoder(n_t)
                t_z = reparameterize(t_tmu, t_tlv) 
                t_z = t_z.reshape(a, b, t_z.shape[-1])     # t_z.shape: torch.Size([2500, 4, 96])
                
                t_z_list.append(t_z)


        criterion = nn.CrossEntropyLoss().to(device) 
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='training_log.txt',
            filemode='w'  # 'w' to overwrite, 'a' to append
        )
        logging.info('Starting training stage...')
        c_loss = {}
        for i, name in enumerate(names):  
            c_loss[name] = 0
        for c_e in range(300):  # training cycle
            pred_dict = {}
            out_dict = {}
            for i, name in enumerate(names):
                clf_dict[name].train()
                t_z = t_z_list[i]
                _, fg_num, _ = t_z.shape
                fg_pred_list = []   # fine-grained
                fg_out = []
                for j in range(0, fg_num):
                    out = clf_dict[name](t_z[:,j,:])
                    c_loss[name] += criterion(out, y).item()
                    pred = torch.argmax(out, -1).float()
                    fg_pred_list.append(pred)
                    fg_out.append(out)
                pred_dict[name] = sum(fg_pred_list) / fg_num
                out_dict[name] = fg_out
            
            # weighted_pred = pred_dict["global"] * pred_weights['global'] + pred_dict['part'] * pred_weights['part']

            # total loss
            # total_c_loss = c_loss["global"] * alpha + c_loss['part'] * (1.0 - alpha)
            if all(isinstance(c_loss[name], float) for name in names):
                c_loss_tensors = {}
                for i, name in enumerate(names):
                    t_z = t_z_list[i]
                    _, fg_num, _ = t_z.shape
                    c_loss_tensors[name] = sum(criterion(clf_dict[name](t_z[:,j,:]), y) for j in range(fg_num))
                total_c_loss = sum(c_loss_tensors[name] * loss_weights[name] for name in names)
            else:
                total_c_loss = sum(c_loss[name] * loss_weights[name] for name in names)

            # BP
            for name in names:
                clf_optimizer[name].zero_grad()
            total_c_loss.backward()
            for name in names:
                clf_optimizer[name].step()

            train_acc_dict = {}
            for name in names:
                train_acc_dict[name] = float(torch.sum(y == torch.round(pred_dict[name]))) / (ss * 500)
            
            weighted_pred = sum(pred_dict[name] * pred_weights[name] for name in names)
            final_pred = torch.round(weighted_pred).long()  
            final_acc = float(torch.sum(y == final_pred)) / (ss * 500)              
            train_acc_dict['final'] = final_acc

            # print(f"epoch {c_e} === global_c_acc: {global_c_acc:.2f}. part_c_acc: {part_c_acc}. final_acc: {final_acc:.2f}.")
            # pred_list = [p.cpu() for p in pred_list]
            # pred_list = torch.stack(pred_list, dim=0)
            # pred_result = torch.sum(pred_list * weights[:, None], dim=0)
            # pred_result = pred_result.round().to(device)
            # pred_acc = float(torch.sum(y == pred_result))/(ss*500)

            # print(f"Training ... prediction accuracy: {pred_acc}.")
            # global_out = global_out.unsqueeze(1)
            # part_out_list_stacked = torch.stack(part_out_list, dim=1)   # torch.Size([2500, 6, 5])
            # global_part_out = torch.cat([global_out, part_out_list_stacked], dim=1)

            # print(f"Training... {c_e+1} global_c_acc: {global_c_acc}, part_c_acc: {part_c_acc}")
        logging.info('Training stage completed.')

        # global_part_out_cpu = global_part_out.cpu().numpy()
        # np.save(global_part_out_cpu, "global_part_out.npy")
        # print('Text embedding trained out (sample) saved.')
    
    # use skeleton features to do the actual classification
    logging.info("Starting ZSL Classification...")
    u_inds = torch.from_numpy(unseen_inds)
    final_embs = []
    with torch.no_grad():       # evaluate on zsl test set
        for name in names:
            vae_dict[name]['sequence_encoder'].eval()
            clf_dict[name].eval()

        count = 0
        num = 0
        final_preds = []
        tars = []
        pred_stream_dict = {}
        pred_stream_count = {}
        for name in names:
            pred_stream_count[name] = 0
        for (global_feats, part_feats, target) in zsl_loader:    # inp: data of current patch. target: ground truth
            pred_t_dict = {}
            stream_t_pred = {}

            for i, name in enumerate(names):
                t_s = global_feats.to(device)
                sequence_encoder = vae_dict[name]['sequence_encoder']
                nt_smu, t_slv = sequence_encoder(t_s)

                final_embs.append(nt_smu)
                t_out = clf_dict[name](nt_smu)         # torch.Size([32, 5])        
                t_pred = torch.argmax(t_out, -1).float()
                pred_t_dict[name] = t_pred

            weighted_t_pred = sum(pred_t_dict[name] * pred_weights[name] for name in names)

            fused_t_pred = torch.round(weighted_t_pred).long()          
            fused_t_pred = fused_t_pred.cpu()
            final_preds.append(u_inds[fused_t_pred])
            tars.append(target)
            count += torch.sum(u_inds[fused_t_pred] == target)
            num += len(target)

            for i, name in enumerate(names):
                pred_stream_dict[name] = []
                stream_t_pred[name] = torch.round(pred_t_dict[name]).long()
                stream_t_pred[name] = stream_t_pred[name].cpu()
                pred_stream_dict[name].append(u_inds[stream_t_pred[name]])
                pred_stream_count[name] += torch.sum(u_inds[stream_t_pred[name]] == target)

    zsl_accuracy = float(count)/num
    for i, name in enumerate(names):
        zsl_acc_stream = float(pred_stream_count[name])/num
    final_embs = np.array([j.cpu().numpy() for i in final_embs for j in i])
    p = [j.item() for i in final_preds for j in i]
    t = [j.item() for i in tars for j in i]
    p = np.array(p)
    t = np.array(t)

    # val_out_embs = []
    # val_out_logits = []
    # with torch.no_grad():       # evaluating on gzsl test set
    #     sequence_encoder.eval()
    #     for i, part_name in enumerate(part_names):
    #         part_models[part_name]['sequence_encoder'].eval()
    #         clf_dict[part_name].eval()
    #     clf_dict["global"].eval()
    #     gzsl_count = 0
    #     gzsl_num = 0
    #     gzsl_preds = []
    #     gzsl_tars = []
    #     loader = val_loader if phase == 'train' else zsl_loader
    #     for (global_feats, part_feats, target) in loader:   
    #         gzsl_pred_list = []     
    #         t_s = global_feats.to(device)
    #         t_smu, t_slv = sequence_encoder(t_s)    
    #         global_t_out = clf_dict["global"](t_smu)  
    #         gzsl_pred_list.append(torch.argmax(global_t_out, -1).cpu()) 
            
    #         t_s_part = part_feats.to(device)
    #         part_t_out_list = []
    #         for i, part_name in enumerate(part_names):
    #             part_clf = clf_dict[part_name]
    #             part_se = part_models[part_name]['sequence_encoder']
    #             nt_smu_part, t_slv_part = part_se(t_s_part[:,i,:])
    #             part_t_out = part_clf(nt_smu_part)
    #             part_t_out_list.append(part_t_out)
    #             gzsl_pred_list.append(torch.argmax(part_t_out, -1).cpu())

    #         global_t_out = global_t_out.unsqueeze(1)
    #         part_t_out_list_stacked = torch.stack(part_t_out_list, dim=1)   # torch.Size([2500, 6, 5])
    #         global_part_t_out = torch.cat([global_t_out, part_t_out_list_stacked], dim=1)
    #         val_out_logits.append(global_part_t_out)
    #         val_out_embs.append(F.softmax(global_part_t_out, 1))
            
    #         gzsl_pred_list = [p.cpu() for p in gzsl_pred_list]
    #         gzsl_pred_list = torch.stack(gzsl_pred_list, dim=0)
    #         gzsl_pred_result = torch.sum(gzsl_pred_list * best_weights[:, None], dim=0)
    #         gzsl_pred_result = torch.round(gzsl_pred_result).to(torch.long)

    #         gzsl_preds.append(u_inds[gzsl_pred_result])
    #         gzsl_tars.append(target)
    #         gzsl_count += torch.sum(u_inds[gzsl_pred_result] == target)
    #         num += len(target)

    # val_out_logits = np.array([j.cpu().numpy() for i in val_out_logits for j in i])
    # val_out_embs = np.array([j.cpu().numpy() for i in val_out_embs for j in i])  # "ztest_out.npy"
    
    # return zsl_accuracy, val_out_embs, val_out_logits, clf_dict
    return zsl_accuracy, clf_dict


def get_seen_zs_embeddings(clf_dict, sequence_encoder, part_models, val_loader, part_names, device, unseen_inds):
    final_embs = []
    out_val_embeddings = []
    u_inds = torch.from_numpy(unseen_inds)
    with torch.no_grad():
        sequence_encoder.eval()
        clf_dict['global'].eval()
        for i, part_name in enumerate(part_names):
            part_models[part_name]['sequence_encoder'].eval()
            clf_dict[part_name].eval()
        count = 0
        num = 0
        preds = []
        tars = []
        pred_t_list = []
        for (global_feats, part_feats, target) in val_loader:
            t_s = global_feats.to(device)   # torch.Size([32, 256])
            nt_smu, t_slv = sequence_encoder(t_s)   # torch.Size([32, 96]) encoded skeleton latent embeddings. In Encoder forward(): nt_smu -> "mu", t_slv -> "logvar"
            final_embs.append(nt_smu)
            global_t_out = clf_dict["global"](nt_smu)         # torch.Size([32, 5])        
            pred_t_list.append(torch.argmax(global_t_out, -1).cpu())
            # part
            t_s_part = part_feats.to(device)
            part_t_out_list = []
            for i, part_name in enumerate(part_names):
                part_clf = clf_dict[part_name]
                part_se = part_models[part_name]['sequence_encoder']
                nt_smu_part, t_slv_part = part_se(t_s_part[:,i,:])
                part_t_out = part_clf(nt_smu_part)
                part_t_out_list.append(part_t_out)
                pred_t_list.append(torch.argmax(part_t_out, -1).cpu())
            
            global_t_out = global_t_out.unsqueeze(1)
            part_t_out_list_stacked = torch.stack(part_t_out_list, dim=1)   # torch.Size([2500, 6, 5])
            global_part_t_out = torch.cat([global_t_out, part_t_out_list_stacked], dim=1)
            out_val_embeddings.append(F.softmax(global_part_t_out, 1))

    out_val_embeddings = np.array([j.cpu().numpy()
                                  for i in out_val_embeddings for j in i])          # "val_out.npy"
    return out_val_embeddings


def save_classifier(cls):
    cls_checkpoint = f'{wdir}/{le}/{tm}/classifier.pth.tar'
    save_checkpoint({'state_dict': cls.state_dict()}, cls_checkpoint)

def save_clf_dict(cls_dict):
    cls_checkpoints = f'{wdir}/{le}/{tm}/classifiers.pth.tar'
    model_checkpoints = {}
    for name, model in cls_dict.items():
        model_checkpoints[name] = model.state_dict()  
    save_checkpoint({'state_dict': model_checkpoints}, cls_checkpoints)

def get_part_names(body_part):
    if body_part == 6:
        part_names = ["head", "hand", "arm", "hip", "leg", "foot"]
    elif body_part == 4:
        part_names = ["head", "upper limbs", "hip", "lower limbs"]
    elif body_part == 2:
        part_names = ["upper body", "lower body"]
    return part_names

def init_vaes(names, vis_emb_input_size, semantic_latent_size, style_latent_size, text_emb_input_size, device):
    vae_dict = {}
    for name in names:
        sequence_encoder = Encoder(
            [vis_emb_input_size, semantic_latent_size + style_latent_size], style_latent_size).to(device)
        sequence_decoder = Decoder(
            [semantic_latent_size + style_latent_size, vis_emb_input_size]).to(device)
        text_encoder = Encoder(
            [text_emb_input_size, semantic_latent_size]).to(device)
        text_decoder = Decoder(
            [semantic_latent_size, text_emb_input_size]).to(device)

        # Discriminator
        discriminator = Discriminator(
            semantic_latent_size + style_latent_size).to(device)

        # Optimizer
        params = []
        for model in [sequence_encoder, sequence_decoder, text_encoder, text_decoder]:
            params += list(model.parameters())
        optimizer = optim.Adam(params, lr=args.lr)
        dis_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr)
        
        vae_dict[name] = {
            "sequence_encoder": sequence_encoder,
            "sequence_decoder": sequence_decoder,
            "text_encoder": text_encoder,
            "text_decoder": text_decoder,
            "discriminator": discriminator,
            "optimizer": optimizer,
            "dis_optimizer": dis_optimizer
        }
    return vae_dict 

def load_semantic_emb(source, device):
    if num_classes == 60:
            text = source[:60]
    else:
        text = source[:120]
    text_emb = text / torch.norm(text, dim=1, keepdim=True)
    text_emb = text_emb.to(device, non_blocking=True)

    return text_emb

def main():
    # Embedding Dim
    if args.ve == 'shift':
        vis_emb_input_size = 256
    elif args.ve == 'posec3d':
        vis_emb_input_size = 512
    elif args.ve == 'stgcn':
        vis_emb_input_size = 256
    else:
        raise ValueError('Unknown visual embedding model')
    text_emb_input_size = 1024
    askg_mode = args.askg_mode

    seed = 5
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda")

    if not os.path.exists(f'{wdir}/{le}/{tm}'):
        os.makedirs(f'{wdir}/{le}/{tm}')

    # DataLoader
    ntu_loaders = NTUDataLoaders(dataset_path, 'max', 1)
    train_loader = ntu_loaders.get_train_loader(
        batch_size, 0)
    zsl_loader = ntu_loaders.get_val_loader(batch_size, 0)
    val_loader = ntu_loaders.get_test_loader(batch_size, 0)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        filename=f'char_logs/ntu{num_classes}_{ss}_{st}_noaug.txt',
        filemode='w'  # 'w' to overwrite, 'a' to append
    )
    names = ['whole', 'xaa', 'xao']
    logging.info(f'semantic embedding types: {names}')
    if phase == 'val':
        unseen_inds = np.sort(
            np.load(f'resources/label_splits/{dataset}/{st}v{str(ss)}_0.npy'))
        seen_inds = np.load(
            f'resources/label_splits/{dataset}/{st}s{str(num_classes - ss - ss)}_0.npy')
    else:
        unseen_inds = np.sort(
            np.load(f'resources/label_splits/{dataset}/{st}u{str(ss)}.npy'))
        seen_inds = np.load(
            f'resources/label_splits/{dataset}/{st}s{str(num_classes - ss)}.npy')

    
    aug = False
    c_text_emb = []
    c_unseen_text_emb = []
    if aug:
        prefix = 'aug'
    else:
        prefix = 'no_aug'
    if 'whole' in names:
        tml = tm.split('_')
        tfl = [torch.from_numpy(
            np.load(f'resources/text_feats/{args.dataset}/{le}/{m}_{num_classes}.npy')) for m in tml]
        text_feat = torch.concat(tfl, dim=-1)
        text_emb_input_size = text_feat.size(-1)
        text_emb = text_feat / torch.norm(text_feat, dim=1, keepdim=True)
        text_emb = text_emb.to(device, non_blocking=True)
        text_emb = text_emb.unsqueeze(dim=1)
        c_text_emb.append(text_emb)
        c_unseen_text_emb.append(text_emb[unseen_inds, :])
    if 'xaa' in names:
        xaa_source = torch.load(f'ASKG/data/{askg_mode}/{prefix}/xaa_text_feats_xprompt_ntu.tar', weights_only=True)
        xaa_text_emb = load_semantic_emb(xaa_source, device)
        c_text_emb.append(xaa_text_emb)
        c_unseen_text_emb.append(xaa_text_emb[unseen_inds,:,:])
    if 'xao' in names:
        xao_source = torch.load(f'ASKG/data/{askg_mode}/{prefix}/xao_text_feats_xprompt_ntu.tar', weights_only=True)
        xao_text_emb = load_semantic_emb(xao_source, device)
        c_text_emb.append(xao_text_emb)
        c_unseen_text_emb.append(xao_text_emb[unseen_inds,:,:])

    vae_dict = init_vaes(names, vis_emb_input_size, semantic_latent_size, style_latent_size, text_emb_input_size, device)
    # ========== Training ==========
    best = 0
    for epoch in range(num_epochs):
        # ===== Train Cross-Alignment Module =====
        if load_vae == True:
            vae_checkpoint = f'{wdir}{le}/{tm}/se_16999_vae_models.pth.tar'
            vae_load_dict = torch.load(vae_checkpoint, weights_only=False)
            # # global
            # sequence_encoder.load_state_dict(vae_load_dict['state_dict']['global']['sequence_encoder'])
            # text_encoder.load_state_dict(vae_load_dict['state_dict']['global']['text_encoder'])
            # text_decoder.load_state_dict(vae_load_dict['state_dict']['global']['text_decoder'])
            # sequence_decoder.load_state_dict(vae_load_dict['state_dict']['global']['sequence_decoder'])
            # # part
            # for i, part_name in enumerate(part_names):
            #     for model_name, model in part_models.items():
            #         if model_name in ['sequence_encoder', 'sequence_decoder', 'text_encoder', 'text_decoder', 'optimizer']:
            #             model.load_state_dict(vae_load_dict['state_dict'][part_name][model_name])
        else:
            for i, name in enumerate(names):
                # train_one_cycle(epoch,
                #                 vae_dict[name]['sequence_encoder'], vae_dict[name]['sequence_decoder'],
                #                 vae_dict[name]['text_encoder'], vae_dict[name]['text_decoder'], 
                #                 vae_dict[name]['discriminator'], vae_dict[name]['optimizer'], vae_dict[name]['dis_optimizer'], 
                #                 train_loader, device, c_text_emb[i])
                train_one_cycle_with_zsl(epoch,
                                vae_dict[name]['sequence_encoder'], vae_dict[name]['sequence_decoder'],
                                vae_dict[name]['text_encoder'], vae_dict[name]['text_decoder'], 
                                vae_dict[name]['discriminator'], vae_dict[name]['optimizer'], vae_dict[name]['dis_optimizer'], 
                                train_loader, device, c_text_emb[i], zsl_loader=zsl_loader, unseen_inds=unseen_inds, unseen_text_emb=c_unseen_text_emb[i])
            if phase == 'train':
                save_all_model(cycle_length*(epoch+1)-1, vae_dict) 
    
        # ===== Train Classifier =====
        # zsl_acc, clf_dict = train_classifier(names, vae_dict, zsl_loader, val_loader, unseen_inds, c_unseen_text_emb, alpha, alpha_p, device)
        zsl_acc_dict, zsl_acc = zsl_alignment(names, vae_dict, zsl_loader, val_loader, unseen_inds, c_unseen_text_emb, alpha, alpha_p, device, logging=logging)

        if (zsl_acc > best):
            best = zsl_acc
            # save_clf_dict(clf_dict)
            print('---------------------')
            print(
                f'zsl_accuracy increased to {best :.2%} on cycle ', epoch)
            logging.info(f'Cycle {epoch+1} best zsl_acc: {best :.2%}. Components acc: {zsl_acc_dict}')
        #     print('checkpoint saved')
            # if phase == 'train':
            #     np.save(
            #         f'{wdir}/{le}/{tm}/MSF_{str(ss)}_r_gzsl_zs.npy', val_out_embs)
            # else:
            #     np.save(
            #         f'{wdir}/{le}/{tm}/MSF_{str(ss)}_r_unseen_zs.npy', val_out_embs)    # "ztest_out.npy"
            #     seen_zs_embeddings = get_seen_zs_embeddings(
            #         clf_dict, sequence_encoder, part_models, val_loader, device, unseen_inds)
            #     np.save(
            #         f'{wdir}/{le}/{tm}/MSF_{str(ss)}_r_seen_zs.npy', seen_zs_embeddings)        # "val_out.npy"
    return best


if __name__ == "__main__":
    main()
