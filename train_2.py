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

unseen_classes = [10, 11, 19, 26, 56]   # ntu60_55/5_split
# unseen_classes = [4,19,31,47,51]   # ablation study ntu60 split1
# unseen_classes = [12,29,32,44,59]   # ablation study ntu60 split2
# unseen_classes = [7,20,28,39,58]   # ablation study ntu60 split3
# unseen_classes = [3,5,9,12,15,40,42,47,51,56,58,59]  # ntu60_48/12_split
# unseen_classes = [4,13,37,43,49,65,88,95,99,106]  # ntu120_110/10_split
# unseen_classes = [5,9,11,16,18,20,22,29,35,39,45,49,59,68,70,81,84,87,93,94,104,113,114,119]  # ntu120_96/24_split
seen_classes = list(set(range(60))-set(unseen_classes))  # ntu60
train_label_dict = {}
for idx, l in enumerate(seen_classes):
    tmp = [0] * len(seen_classes)
    tmp[idx] = 1
    train_label_dict[l] = tmp
test_zsl_label_dict = {}
for idx, l in enumerate(unseen_classes):
    tmp = [0] * len(unseen_classes)
    tmp[idx] = 1
    test_zsl_label_dict[l] = tmp
test_gzsl_label_dict = {}
for idx, l in enumerate(range(60)):
    tmp = [0] * 60
    tmp[idx] = 1
    test_gzsl_label_dict[l] = tmp


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
    parser.add_argument('--tm', type=str, help='text mode')
    parser.add_argument("--batch_size", type=int,
                        default=64, help='batch size')
    parser.add_argument("--dis_step", type=int, default=10, help='dis step')

    parser.add_argument("--beta_x", type=float, default=None)
    parser.add_argument("--beta_y", type=float, default=None)
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
tm = args.tm
batch_size = args.batch_size

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
                    train_loader, device, text_emb, type='global'):  # 0-10, 1700
    print(f"training {type} VAE.")
    dis_step = args.dis_step
    # Loss
    mse_criterion = nn.MSELoss().to(device)
    bce_criterion = nn.BCELoss().to(device)
    ipdb.set_trace()
    cr_fact_iter = int(0.8 * len(train_loader))
    beta_iter = int(len(train_loader) / 3)
    for i, (inputs, target) in enumerate(train_loader):
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
        
        s = inputs.to(device, non_blocking=True)
        t = target.to(device, non_blocking=True)
        t = get_text_data(text_emb, t).to(device, non_blocking=True)

        smu, slv, ismu, islv = sequence_encoder(s, instance_style=True, type=type)      
        sz = reparameterize(smu, slv)
        isz = reparameterize(ismu, islv)
        sout = sequence_decoder(torch.cat([sz, isz], dim=-1))

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
    return


def save_model(epoch, sequence_encoder, sequence_decoder, text_encoder, text_decoder, optimizer, type="global"):
    if type == "global":
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
        
    elif type == "part":
        for i in range(0,6):
            save_checkpoint({'epoch': epoch + 1,
                            'state_dict': sequence_encoder[i].state_dict(),
                            }, f'{wdir}/{le}/{tm}/se_{str(epoch)}_part_{i}.pth.tar')
            save_checkpoint({'epoch': epoch + 1,
                            'state_dict': sequence_decoder[i].state_dict(),
                            }, f'{wdir}/{le}/{tm}/sd_{str(epoch)}_part_{i}.pth.tar')
            save_checkpoint({'epoch': epoch + 1,
                            'state_dict': text_encoder[i].state_dict(),
                            }, f'{wdir}/{le}/{tm}/te_{str(epoch)}_part_{i}.pth.tar')
            save_checkpoint({'epoch': epoch + 1,
                            'state_dict': text_decoder[i].state_dict(),
                            }, f'{wdir}/{le}/{tm}/td_{str(epoch)}_part_{i}.pth.tar')




def train_classifier(text_encoder, sequence_encoder, p_text_encoder_list, p_sequence_encoder_list, zsl_loader, val_loader, unseen_inds, unseen_text_emb, device):
    clf = MLP([semantic_latent_size, ss]).to(device)  # MLP classifier. semantic_laten_size, ss (step size) are args.    
    part_clf_list = []
    for i in range(0,6):
        part_clf = MLP([semantic_latent_size, ss]).to(device)
        part_clf_list.append(part_clf)
    if load_classifier == True:
        cls_checkpoint = f'{wdir}/{le}/{tm}/classifier.pth.tar'
        clf.load_state_dict(torch.load(cls_checkpoint)['state_dict'])
    else:
        # Reference: https://github.com/cseeyangchen/STAR. /model/shiftgcn_match_ntu.py

        # load the semantic attributes
        # attribute_features_dict = torch.load('/DATA3/cy/STAR/data/text_feature/ntu_spatial_temporal_attribute_feature_dict_gpt35.tar')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        action_descriptions = torch.load('text_feature/ntu_semantic_part_feature_dict_gpt35_6part_512.tar')

        # load part language description
        part_language = []
        for i, part_name in enumerate(["head", "hand", "arm", "hip", "leg", "foot"]):
            part_language.append(action_descriptions[i+1].unsqueeze(1))
        part_language1 = torch.cat(part_language, dim=1).cuda(device)      # part_language1.shape: torch.Size([120, 6, 768]) [action label, body part, text embeddings]

        unseen_label_ind = range(ss)        # 这里是 unseen_label 的 index, 现在设置为与下面的 y 保持一致. 需要检查 SA-DVAE 中的 unseen_class 是怎么设置的
        # 这里可能需要把 unseen_label_ind 转成 有 item() 的数据, 具体是什么还需要看一下 (可能: label序号, label名称, label名称的embeddings)
        part_language = torch.cat([part_language1[l,:,:].unsqueeze(0) for l in unseen_label_ind], dim=0)   
        part_language_seen = part_language1[seen_classes]   

        label_language = torch.cat([action_descriptions[0][l].unsqueeze(0) for l in unseen_label_ind], dim=0).cuda(device)  # label_language.shape: torch.Size([5, 768])

        # use text features to train the classifier
        cls_optimizer = optim.Adam(clf.parameters(), lr=0.001) # SGD or Adam
        with torch.no_grad():   
            y = torch.tensor(range(ss)).to(device)      # ss=5 here
            y = y.repeat([500])

            text_encoder.eval()
            n_t = unseen_text_emb.to(device).float()
            n_t = n_t.repeat([500, 1])
            t_tmu, t_tlv = text_encoder(n_t)
            t_z = reparameterize(t_tmu, t_tlv)
            
            n_pl = part_language.float()
            n_pl = n_pl.repeat([500, 1, 1])

            t_z_pl = []
            for i in range(6):
                part_te = p_text_encoder_list[i]
                t_tmu_pl, t_tlv_pl = part_te(n_pl[:,i,:])
                t_z_pl_i = reparameterize(t_tmu_pl, t_tlv_pl)
                t_z_pl.append(t_z_pl_i)
            t_z_pl = torch.stack(t_z_pl, dim=1)

        criterion_global = nn.CrossEntropyLoss().to(device) 
        criterion_part = nn.CrossEntropyLoss().to(device)   # 更改

        clf_train_history = {
                "c_acc": [],
                "c_loss": [],
                "global_c_loss": [],
                "part_loss": []
            }
        for c_e in range(300):  # training cycle
            clf.train()

            # global        
            global_out = clf(t_z)
            global_c_loss = criterion_global(global_out, y)
            # part 
            part_out_list = []
            for i in range(6):  # 6 parts
                part_clf = part_clf_list[i]
                part_out = part_clf(t_z_pl[:,i,:])       # 这里的 classifier 需要换吗?
                part_out_list.append(part_out)

            # fuse logits
            part_fused_output = fuse_logits(part_out_list, fusion_type="logsumexp")
            part_c_loss = criterion_part(part_fused_output, y)

            # global and part losses -> fuse -> total loss
            c_loss = global_c_loss + part_c_loss    # 还需要一个 balance factor
            # c_loss = global_c_loss
            cls_optimizer.zero_grad()
            c_loss.backward()
            cls_optimizer.step()
            c_acc = float(torch.sum(y == torch.argmax(global_out, -1)))/(ss*500)

            # print(f"Training... {c_e+1} c_acc: {c_acc}, c_loss: {c_loss}")

    # use skeleton features to do the actual classification
    clf.eval()

    u_inds = torch.from_numpy(unseen_inds)
    final_embs = []
    with torch.no_grad():       # evaluate on zsl test set
        sequence_encoder.eval()
        clf.eval()
        count = 0
        num = 0
        preds = []
        tars = []

        # import class "ModelMatch" from STAR to finegrain global feature into part features.
        finegrain_model = ModelMatch(num_class=60,
                         num_point=25,
                         num_person=2,
                         graph="graph.ntu_rgb_d.Graph",
                         graph_args={'labeling_mode': 'spatial'},
                         in_channels=3)
        for (inp, target) in zsl_loader:    # inp: data of current patch. target: ground truth
            t_s = inp.to(device)
            nt_smu, t_slv = sequence_encoder(t_s)   # encoded skeleton latent embeddings. In Encoder forward(): nt_smu -> "mu", t_slv -> "logvar"
            
            part_language = torch.cat([part_language1[l,:,:].unsqueeze(0) for l in target], dim=0)
            part_language_seen = part_language1[seen_classes]
            label_language = torch.cat([action_descriptions[0][l].unsqueeze(0) for l in target], dim=0).cuda(device)


            final_embs.append(nt_smu)
            t_out = clf(nt_smu)                     # t_out: contains logits output by clf (MLP)
            pred = torch.argmax(t_out, -1).cpu()
            preds.append(u_inds[pred])
            tars.append(target)
            count += torch.sum(u_inds[pred] == target)
            num += len(target)
    zsl_accuracy = float(count)/num
    final_embs = np.array([j.cpu().numpy() for i in final_embs for j in i])
    p = [j.item() for i in preds for j in i]
    t = [j.item() for i in tars for j in i]
    p = np.array(p)
    t = np.array(t)

    val_out_embs = []
    with torch.no_grad():       # evaluating on gzsl test set
        sequence_encoder.eval()
        clf.eval()
        gzsl_count = 0
        gzsl_num = 0
        gzsl_preds = []
        gzsl_tars = []
        loader = val_loader if phase == 'train' else zsl_loader
        for (inp, target) in loader:        
            t_s = inp.to(device)
            t_smu, t_slv = sequence_encoder(t_s)    
            t_out = clf(t_smu)                      
            val_out_embs.append(F.softmax(t_out, 1))
            pred = torch.argmax(t_out, -1).cpu()
            gzsl_preds.append(u_inds[pred])
            gzsl_tars.append(target)
            gzsl_count += torch.sum(u_inds[pred] == target)
            num += len(target)

    val_out_embs = np.array([j.cpu().numpy() for i in val_out_embs for j in i])

    return zsl_accuracy, val_out_embs, clf


def get_seen_zs_embeddings(clf, sequence_encoder, val_loader, device, unseen_inds):
    final_embs = []
    out_val_embeddings = []
    u_inds = torch.from_numpy(unseen_inds)
    with torch.no_grad():
        sequence_encoder.eval()
        clf.eval()
        count = 0
        num = 0
        preds = []
        tars = []
        for (inp, target) in val_loader:
            t_s = inp.to(device)
            t_smu, t_slv = sequence_encoder(t_s)
            final_embs.append(t_smu)
            t_out = clf(t_smu)
            out_val_embeddings.append(F.softmax(t_out, dim=1))
            pred = torch.argmax(t_out, -1).cpu()
            preds.append(u_inds[pred])
            tars.append(target)
            count += torch.sum(u_inds[pred] == target)
            num += len(target)

    out_val_embeddings = np.array([j.cpu().numpy()
                                  for i in out_val_embeddings for j in i])
    return out_val_embeddings


def save_classifier(cls):
    cls_checkpoint = f'{wdir}/{le}/{tm}/classifier.pth.tar'
    save_checkpoint({'state_dict': cls.state_dict()}, cls_checkpoint)


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
    
    tml = tm.split('_')
    tfl = [torch.from_numpy(
        np.load(f'resources/text_feats/{args.dataset}/{le}/{m}_{num_classes}.npy')) for m in tml]
    text_feat = torch.concat(tfl, dim=-1)
    text_emb_input_size = text_feat.size(-1)
    text_emb = text_feat / torch.norm(text_feat, dim=1, keepdim=True)
    text_emb = text_emb.to(device, non_blocking=True)

    unseen_text_emb = text_emb[unseen_inds, :]
    print("global language embeddings loaded.")

    action_descriptions = torch.load('text_feature/ntu_semantic_part_feature_dict_gpt35_6part_512.tar')

    # load part language description
    part_language = []
    for i, part_name in enumerate(["head", "hand", "arm", "hip", "leg", "foot"]):
        part_language.append(action_descriptions[i+1].unsqueeze(1))
    part_language1 = torch.cat(part_language, dim=1).cuda(device)      # part_language1.shape: torch.Size([120, 6, 768]) [action label, body part, text embeddings]

    if num_classes == 60:
        part_text_feat = part_language1[:60]
    
    part_text_emb = part_text_feat / torch.norm(part_text_feat, dim=1, keepdim=True)
    part_text_emb = part_text_emb.to(device, non_blocking=True)

    print("part language embeddings loaded.")

    # VAE: variational autoencoders
    # global
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

    # part
    p_sequence_encoder_list = []
    p_sequence_decoder_list = []
    p_text_encoder_list = []
    p_text_decoder_list = []
    p_discriminator_list = []
    p_optimizer_list = []
    p_dis_optimizer_list = []
    for i in range(0,6):
        p_sequence_encoder = Encoder(
            [vis_emb_input_size, semantic_latent_size + style_latent_size], style_latent_size).to(device)
        p_sequence_decoder = Decoder(
            [semantic_latent_size + style_latent_size, vis_emb_input_size]).to(device)
        p_text_encoder = Encoder(
            [text_emb_input_size, semantic_latent_size]).to(device)
        p_text_decoder = Decoder(
            [semantic_latent_size, text_emb_input_size]).to(device)

        # Discriminator
        p_discriminator = Discriminator(
            semantic_latent_size + style_latent_size).to(device)

        # Optimizer
        p_params = []
        for model in [p_sequence_encoder, p_sequence_decoder, p_text_encoder, p_text_decoder]:
            p_params += list(model.parameters())
        p_optimizer = optim.Adam(p_params, lr=args.lr)
        p_dis_optimizer = optim.Adam(p_discriminator.parameters(), lr=args.lr)
        
        p_sequence_encoder_list.append(p_sequence_encoder)
        p_sequence_decoder_list.append(p_sequence_decoder)
        p_text_encoder_list.append(p_text_encoder)
        p_text_decoder_list.append(p_text_decoder)
        p_discriminator_list.append(p_discriminator)
        p_optimizer_list.append(p_optimizer)
        p_dis_optimizer_list.append(p_dis_optimizer)

    # Training
    best = 0
    for epoch in range(num_epochs):
        train_one_cycle(epoch,
                        sequence_encoder, sequence_decoder, text_encoder, text_decoder, discriminator,
                        optimizer, dis_optimizer,
                        train_loader, device, text_emb, type='global')     # train_loader: Data_Loader from data_cnn60.py            
        if phase == 'train':
            save_model(cycle_length*(epoch+1)-1, sequence_encoder,
                        sequence_decoder, text_encoder, text_decoder, optimizer)
        # for i in range(0,6):
        #     train_one_cycle(epoch,
        #                     p_sequence_encoder_list[i], p_sequence_decoder_list[i], p_text_encoder_list[i], 
        #                     p_text_decoder_list[i], p_discriminator_list[i], p_optimizer_list[i], p_dis_optimizer_list[i],
        #                     train_loader, device, part_text_emb[:,i,:], type='part')
        # if phase == 'train':
        #     save_model(cycle_length*(epoch+1)-1, p_sequence_encoder_list,
        #                 p_sequence_encoder_list, p_text_encoder_list, p_text_decoder_list, p_optimizer_list, type='part') 
        zsl_acc, val_out_embs, clf = train_classifier(
            text_encoder, sequence_encoder, p_text_encoder_list, p_sequence_encoder_list, zsl_loader, val_loader, unseen_inds, unseen_text_emb, device)
        if (zsl_acc > best):
            best = zsl_acc
            save_classifier(clf)
            print('---------------------')
            print(
                f'zsl_accuracy increased to {best :.2%} on cycle ', epoch)
            print('checkpoint saved')
            if phase == 'train':
                np.save(
                    f'{wdir}/{le}/{tm}/MSF_{str(ss)}_r_gzsl_zs.npy', val_out_embs)
            else:
                np.save(
                    f'{wdir}/{le}/{tm}/MSF_{str(ss)}_r_unseen_zs.npy', val_out_embs)
                seen_zs_embeddings = get_seen_zs_embeddings(
                    clf, sequence_encoder, val_loader, device, unseen_inds)
                np.save(
                    f'{wdir}/{le}/{tm}/MSF_{str(ss)}_r_seen_zs.npy', seen_zs_embeddings)
                


if __name__ == "__main__":
    main()
