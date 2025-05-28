import sys

sys.path.append('../../')

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda import amp      # 混合精度训练

import numpy as np
import argparse
import tqdm

import os
import pathlib
import random
import yaml
from copy import deepcopy

from Models.SemanticSegment.realseg import BiBranchSeg
from evaluate_seg import validation
from Metrics.SemanticSegment.SemanticSegLoss import SegLoss,OhemCrossEntropy,pixel_acc
from Tools.utils_od import eval_config
from Tools.utils import select_device,initseed,CosineDecayLR,de_parallel
from Tools.tricks import ModelEMA
from Datasets.SemanticSegment.DatasetAPI import data_construct
from Tools.OptimPolicy.SamOptimizer import *

FILE = pathlib.Path(__file__).resolve()
ROOT = FILE.parents[2]
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

def parse_opt():
    parser = argparse.ArgumentParser()


    parser.add_argument('--saved_dir', default=f"{ROOT}/Running/SemanticSegment/Cityscapes/exp1", help='save to project/name')
    parser.add_argument('--weight', type=str, default="", help='pretraining saved file')
    parser.add_argument('--resume', type=str, default="", help='pretraining saved file')
    parser.add_argument('--dataset', type=str, default="../../Datasets/SemanticSegment/Cityscapes/Cityscapes.yaml", help='pretraining saved file')


    parser.add_argument('--is_coco', action='store_true', default=False, help='using CoCo dataset to train')
    parser.add_argument('--finetune',action='store_true', default=False, help='finetune model,learning rate descent 50%')

    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--device', default="0", help='cuda device,i.e. 0,1,2 or cpu')
    parser.add_argument('--sync_bn', action='store_true', help='only use to DDP,to enable sync batch norm')

    parser.add_argument('--epochs', type=int,default=200, help='training epoch for model')
    parser.add_argument('--batch_size', type=int,default=8, help='getting batch size in dataset')
    parser.add_argument('--image_size',nargs='+',type=int,default=(1024,1024),help='image size which feed into model')
    parser.add_argument('--workers',type=int,default=2,help='the number worker of dataset func')
    opt = parser.parse_args()
    print(opt)
    return opt


def train(args,device):
    batch_size,epochs,config_path,weight,saved_dir,workers,image_size = args.batch_size,args.epochs,\
                                                                    args.dataset,args.weight,\
                                                                    args.saved_dir,args.workers,np.array(args.image_size)

    # --------------------------------------- define ------------------------------------------
    start_epoch = 0
    saved_dir = pathlib.Path(saved_dir)
    if not saved_dir.exists(): saved_dir.mkdir(parents=True, exist_ok=True)
    # device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

    # 读取数据集配置文件,生成数据集
    with open(config_path, encoding='ascii', errors='ignore') as f:
        yaml_file = yaml.safe_load(f)  # model dict
    with open(f"{saved_dir}/config.yaml", 'w', encoding='ascii') as f:
        keep = ['optim_hyp','info_dict','augment_hyp']
        for k,v in yaml_file.items():
            if k in keep:
                yaml.safe_dump({k:v}, f)

    num_cls = yaml_file['num_cls']
    loader_hyp = yaml_file['loader_hyp']
    with_cuda = device != 'cpu'
    initseed(1+RANK)      # 初始化随机种子

    # ---------------------------------------------- 模型构建 ----------------------------------------
    if weight:
        pkl = torch.load(weight, map_location='cpu')
        model = BiBranchSeg(yaml_file['architecture'],is_debug=False).to(device)
        model.load_state_dict(pkl)
    else:
        model = BiBranchSeg(yaml_file['architecture'],is_debug=False).to(device)

    dataset_path = yaml_file['train_data'] if os.path.isabs(yaml_file['train_data']) else f"{ROOT}/Datasets/{yaml_file['train_data']}"
    val_dataset_path = yaml_file['val_data'] if os.path.isabs(yaml_file['val_data']) else f"{ROOT}/Datasets/{yaml_file['val_data']}"
    print("Current dataset path: ",dataset_path)

    if yaml_file.get('pseudo_data') and yaml_file['pseudo_data']!='':
        dataset_path = [dataset_path,yaml_file['pseudo_data'] if os.path.isabs(yaml_file['pseudo_data']) else f"{ROOT}/Datasets/{yaml_file['pseudo_data']}"]
    dataset,dataloader = data_construct(dataset_path,yaml_file['suffix'],image_size,num_cls,batch_size,32,ignore_label=yaml_file['ignore_label'],
                                        istraining=True,loader_hyp=loader_hyp,is_norm=True,workers=workers,rank=LOCAL_RANK)
    if RANK in [-1,0]:
        val_dataset,val_dataloader = data_construct(val_dataset_path,yaml_file['suffix'],image_size,num_cls,batch_size*2,32,
                                                    ignore_label=yaml_file['ignore_label'],istraining=False,loader_hyp=loader_hyp,is_norm=True,workers=workers)


    nb = len(dataloader)    # 记录batch数量 num_batch


    # ----------------------------------------------- 损失与优化配置读取 -------------------------------------

    info_dict = yaml_file['info_dict']
    info_dict = eval_config(info_dict, yaml_file)

    loss_seg = SegLoss(info_dict,dataset.ignore_label,dataset.class_weights).to(device)  # 分割损失
    loss_ohem = OhemCrossEntropy(dataset.ignore_label,0.9,min_kept=131072,weight=dataset.class_weights).to(device)

    # ------------------------------------------- trick ------------------------------------------



    optim_hyp = yaml_file['optim_hyp']
    lr = optim_hyp['lr']
    lr_decay_rate = optim_hyp['lr_decay_rate']

    ema = ModelEMA(model)
    optim_bs = 64  # 一次优化的批大小
    accumulate = max(round(optim_bs / batch_size), 1)
    warmup_iter = 3 * nb  # 预热迭代次数
    optim_hyp['weight_decay'] *= batch_size * accumulate / optim_bs     # 权重衰减率缩放

    if optim_hyp['optimizer_type'] == 'Adam':
        optim = torch.optim.Adam(model.parameters(),lr=lr,betas=(0.937, 0.999))
    elif optim_hyp['optimizer_type'] == 'SGD':
        lr = lr  # SGD比Adam高一个量级
        optim = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.937, weight_decay=optim_hyp['weight_decay'])
    else:
        print(f"optimizer setting error: not {optim_hyp['optimizer_type']}, please select Adam or SGD in config:{config_path}")
        return None


    scheduler = CosineDecayLR(optim,
                              T_max=epochs * nb,
                              lr_init=lr,
                              lr_min=lr*lr_decay_rate,
                              warmup=warmup_iter)


    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        ema.ema.load_state_dict(checkpoint['ema'])
        ema.updates = checkpoint['updates']
        optim.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
    else:   model.half().float()    # 半精度训练

    if with_cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        if args.sync_bn:
            # SyncBatchNorm
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    #训练到最后20epochs时需要关闭mosaic数据增强

    bestmAP = 0.0
    last_opt_step = -1               # 最近一次梯度更新位置
    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=with_cuda)
    for epoch in range(start_epoch,epochs):
        if RANK != -1:
            dataloader.sampler.set_epoch(epoch)
        par = enumerate(dataloader)
        if RANK in [-1,0]:
            par = tqdm.tqdm(par,total=nb)
        loss_mean = torch.zeros(2,device=device)
        # loss_supervision_mean = torch.zeros(1,device=device)        # 3对应深度监督层数
        model.train()
        optim.zero_grad()
        for i,(samples, targets, paths, shapes) in par:
            cur_iter_loc = nb * epoch + i
            # 学习率更新
            scheduler.step(cur_iter_loc)
            if cur_iter_loc < warmup_iter:
                accumulate = max(1, np.interp(cur_iter_loc, [0,warmup_iter], [1, optim_bs / batch_size]).round())
                # for j, x in enumerate(optim.param_groups):
                #     # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                #     x['lr'] = np.interp(cur_iter_loc, [0,warmup_iter], [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                #     if 'momentum' in x:
                #         x['momentum'] = np.interp(cur_iter_loc, [0,warmup_iter], [0.8, 0.937])

            image_batch = samples.decompose()[0].to(device,non_blocking=True)
            targets = targets.to(device)

            with amp.autocast(enabled=with_cuda):
                pred,pred_deep = model(image_batch)
                # loss_stardard = loss_seg(pred, targets)
                loss_stardard = loss_ohem(pred, targets)
                loss_deep = loss_seg(pred_deep, targets)
                pixel_acc_stardrad = pixel_acc(pred,targets)
                loss = loss_stardard + loss_deep
                loss_detail = torch.cat([loss_stardard.detach().unsqueeze(0),loss_deep.detach()])
                if RANK != -1:
                    loss *= WORLD_SIZE      # gradient averaged between devices in DDP mode

            # loss.backward()
            scaler.scale(loss).backward()

            if cur_iter_loc-last_opt_step>=accumulate:
                scaler.step(optim)
                scaler.update()
                # optim.step()
                optim.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = cur_iter_loc


            if RANK in [-1, 0]:
                loss_mean = (loss_mean * i + loss_detail) / (i + 1)
                memory = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                par.set_description("Now epoch:{},Memory:{},batch:{}/{} seg loss:{:.5f} deep loss:{:.5f} Cur Acc:{:.5f} learning_rate:{:.5f}"
                                 .format(epoch,memory,i,nb,loss_mean[0], loss_mean[1],pixel_acc_stardrad,optim.param_groups[0]['lr']))
            # 多尺度训练
            if optim_hyp['multi_scale_train'] and (i + 1) % 10 == 0:
                dataset.img_size = random.choice(range(10, 20)) * 32
                print("Change the Scale for t"
                      "raining with the {} scale images".format(dataset.img_size))
        # scheduler.step()
        print(f"epoch:{epoch}/{epochs} Loss mean: Seg loss:{loss_mean[0]},Deep Supervise loss:{loss_mean[1]}")

        if saved_dir:
            torch.save({
                'model': model.state_dict(),
                'ema': ema.ema.state_dict(),
                'updates': ema.updates,
                'optimizer': optim.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'epoch': epoch
            }, f"{saved_dir}/model_resume.pt")

        if RANK in [-1,0]:
            if (epoch+1)%1 == 0:
                ema.update_attr(model,['strides'])
                metrics_output = validation(
                    model=ema.ema,
                    dataloader=val_dataloader,
                    compute_loss=loss_seg,
                    device=device,
                    cls_info=yaml_file,
                    padding_mode=val_dataset.padding_mode)
                cur_mAP = metrics_output[0]

                if cur_mAP>bestmAP:
                    bestmAP = cur_mAP
                    torch.save(deepcopy(ema.ema.state_dict()), f"{saved_dir}/model_best.pt")
                torch.save(deepcopy(model.state_dict()), f"{saved_dir}/model_last.pt")

                with open(f"{saved_dir}/train.log",'a') as f:
                    f.write(
                        f"epoch:{epoch}/{epochs} Loss mean: seg loss:{loss_mean[0] / len(dataloader)}")
                    f.write(f"Current MAP:{cur_mAP},Best MAP:{bestmAP}\n")

    torch.save(de_parallel(model).state_dict(), f"{saved_dir.absolute()}/model.pkl")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_opt()

    device = select_device(args.device, batch_size=args.batch_size)
    if LOCAL_RANK != -1:
        print(WORLD_SIZE)
        assert args.batch_size % WORLD_SIZE == 0, f'--batch-size {args.batch_size} must be multiple of WORLD_SIZE {WORLD_SIZE}'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
    train(args,device)
    if WORLD_SIZE > 1 and RANK == 0:
        print('Destroying process group... ')
        dist.destroy_process_group()