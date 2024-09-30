#!/usr/bin/env python3
import logging
import pdb
import os
import datetime
import random

import PIL
import torch.nn.functional as F 
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torchvision import transforms
from torch.utils.data import DataLoader
from checkpoint import Checkpoint
from deepbio.utils import send_slack_message
from tensorboardX import SummaryWriter

import my_folder
from dataset_train_random import Dataset, Dataset_mpp
from model import load_model
import utils 
import warnings
warnings.filterwarnings('ignore')

logging = logging.getLogger("prostate")
logging.setLevel("INFO")


def multi_tensor_collate_fn(batch):
    # 배치의 첫 번째 샘플을 가져와 텐서들의 개수를 확인
    first_sample = batch[0]
    pdb.set_trace()
    if isinstance(first_sample, torch.Tensor):
        return torch.stack(batch, dim=0)
    
    elif isinstance(first_sample, (tuple, list)):
        # 각 요소를 재귀적으로 stack하여 처리
        return [torch.stack([sample[i] for sample in batch], dim=0) for i in range(len(first_sample))]

    else:
        raise TypeError("Batch elements must be either torch.Tensor or tuple/list of torch.Tensor.")


def resize_output(outputs, NEP_list): 
    result = [] 
    for output, is_NEP in zip(outputs,NEP_list) : 
        if not is_NEP: 
            output = output[:256,:256]
        output = F.interpolate(output.unsqueeze(0), 1024, mode='bilinear')
        result.append(output)
    return torch.cat(result, dim=0)

def train(args, _round, resume_check):
    print('111')
    if args.patch_size != 1024 or args.mag != 100: 
        raise NotImplementedError
    writer = SummaryWriter(log_dir=args.log_dir)
    train_transforms = transforms.Compose([
                           transforms.ColorJitter(0.2, 0.5, 0.5, 0.1), # (0.1, 0.3, 0.3, 0.05)
                           my_folder.RandomGaussianBlur(),
                           my_folder.RandomAdditiveGaussianNoise()])

    use_cuda = torch.cuda.is_available()
    # Prepare data.
    logging.info("Preparing Data..")
    val_loader = DataLoader(Dataset_mpp(PatchDir=args.data_dir, Train=False, debug=args.debug_mode,
                                    patch_size=args.patch_size, mag=args.mag, mask_downsample=args.mask_downsample),
                            batch_size=args.batch_size*2, pin_memory=True, num_workers=args.num_workers,
                            shuffle=False, drop_last=False)


    if args.inference : 
        train_loader = val_loader 
    else : 
        train_dataset = Dataset_mpp(PatchDir=args.data_dir, Train=True, PatchTransform=train_transforms,
                                RandomCrop=True, RandomMpp=True,  RotateFlipTransform=True, debug=args.debug_mode,
                                imagenet_dir = args.imagenet_dir, imagenet_prob = args.imagenet_prob,
                                patch_size=args.patch_size, mag=args.mag, mask_downsample=args.mask_downsample)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                        shuffle=True, drop_last=True,  pin_memory=True) # , collate_fn = multi_tensor_collate_fn
        

    if args.train_iter <= 0 : 
        args.train_iter = len(train_loader)
    if args.val_iter <= 0 : 
        args.val_iter = len(val_loader)

    # Prepare model.
    logging.info("Preparing Model..")
    best_score = 0
    start_epoch = 1
     
    torch.cuda.empty_cache()
    net = load_model(args.model_name, args.class_num)
    params = list(net.parameters())

    n_params = 0 
    for param in params : 
        n_params += torch.numel(param)
    logging.info(f'Number of parameters: {n_params}')

    if args.optimizer == 'SGD' : 
        optimizer = SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=1e-4, eps=1e-4 if args.half_precision else 1e-8)
    elif args.optimizer == 'AdamW' : 
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-4, eps=1e-4 if args.half_precision else 1e-8)
    else : 
        raise ValueError(f'Unknown optimizer: {args.optimizer}')
    
    scheduler = utils.CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=args.train_iter*args.epochs, max_lr = args.learning_rate, 
                                              min_lr=args.learning_rate/1000, warmup_steps=args.warmup_steps)

    checkpoint = Checkpoint(net, optimizer if resume_check else None)

    if args.checkpoint_file:
        try : 
            checkpoint.load(args.checkpoint_file)
        except KeyError : 
            checkpoint = Checkpoint(net)
            checkpoint.load(args.checkpoint_file)

        if resume_check: 
            best_score = checkpoint.best_score.cpu()
            start_epoch = checkpoint.epoch + 1
        logging.info(f'Checkpoint loaded from {args.checkpoint_file}.')

    
    elif args.inference: 
        raise Exception("Inference without loading checkpoint?")


    if args.reset_epoch:
        best_score = 0
        start_epoch = 1

    if use_cuda:
        net.cuda()
        cudnn.benchmark = True
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    


    if args.loss_function == "bce":
        criterion = utils.my_loss_bce(args.cancer_weight) 
    elif args.loss_function == "dice":
        criterion = utils.DiceLoss() 
    else:
        raise RuntimeError(f"Unknown loss function: {args.loss_function}")


    # Run training.
    logging.info("Training start!")
    start_time = datetime.datetime.now()

    if args.half_precision : 
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, args.epochs + 1):

        # Train epoch.
        net.train(True)

        train_loss = []
        train_aux_loss = [] 
        batch_end_time = datetime.datetime.now()

        train_data_iter = iter(train_loader)

        for idx in range(args.train_iter) : 
            if args.inference: 
                break 
            
            try :
                inputs, targets, is_NEP = next(train_data_iter)  
            except StopIteration : 
                train_data_iter = iter(train_loader)
                inputs, targets, is_NEP = next(train_data_iter) 
            
            if random.random() < args.cutmix_prob: 
                inputs, targets = utils.cutmix(inputs, targets)

        
            # Measure time spent in loading data.
            batch_start_time = datetime.datetime.now()
            targets = targets.cuda()
            # Wrap data.


            inputs = inputs.cuda()

            if random.random() < args.mixup_prob: 
                inputs, targets = utils.mixup(inputs, targets)

            # Forward pass.

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled = args.half_precision ): 
                outputs = net(inputs)
                if isinstance(outputs, tuple): 
                    outputs, aux_loss = outputs
                    aux_loss = aux_loss.mean()
                else : 
                    aux_loss = torch.zeros(1).to(outputs.device)
                outputs = F.interpolate(outputs, 1024, mode='bilinear')
                loss = criterion(outputs, targets) 
                
            optimizer.zero_grad()
            scaler.scale(loss+ aux_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            

            # Calculate metrics.
            batch_loss = loss.cpu().data.item()
            batch_aux_loss = aux_loss.cpu().data.item()
            # Measure time spent in computation.
            batch_end_time = datetime.datetime.now()
            compute_elapsed = batch_end_time - batch_start_time

            train_loss.append(batch_loss)
            train_aux_loss.append(batch_aux_loss)

            if idx == 0 : 
                torchvision.utils.save_image(inputs, 'train_sample_input.jpg')
                torchvision.utils.save_image(targets, 'train_sample_target.jpg')
                vis = inputs*0.5 + torchvision.transforms.Resize((inputs.size(-2), inputs.size(-1)))(targets)*0.5
                torchvision.utils.save_image(vis, 'train_sample.jpg')
                try: 
                    torchvision.utils.save_image(torch.nn.functional.sigmoid(outputs), 'train_sample_output.jpg')
                except : 
                    pdb.set_trace()

            # Log.
            if idx % 100 == 0 :
                logging.info((f"Training, epoch #{epoch}/{args.epochs}, "
                            f"batch #{idx + 1}/{args.train_iter}: "
                            f"loss {np.mean(train_loss):.3f}, "
                            f"auxiliary loss {np.mean(train_aux_loss):.3f}, "
                            f"Current lr: {optimizer.param_groups[0]['lr']}, "
                            f"compute total {str(compute_elapsed)[-9:-4]}s"))
            scheduler.step()
            
            if args.debug_mode and  idx == 10 : 
                break 
        # Validate epoch.

        net.eval()

        val_loss = []
        cf_matrix_val = 0
        batch_end_time = datetime.datetime.now()
        global_count = 0 
        
        val_data_iter = iter(val_loader)
        for idx in range(args.val_iter) : 
            
            try :
                inputs, targets, is_NEP = next(val_data_iter)  
            except StopIteration : 
                val_data_iter = iter(val_loader)
                inputs, targets, is_NEP = next(val_data_iter) 

            if idx == 0 : 
                torchvision.utils.save_image(inputs[0], 'val_input_sample.jpg')

            batch_start_time = datetime.datetime.now()
            
            # Wrap data.
            inputs = inputs.cuda()
            targets = targets.cuda()

            # Forward pass.
            with torch.autograd.no_grad():
            
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled = args.half_precision): 
                    outputs = net(inputs)

                    if isinstance(outputs, tuple): 
                        outputs, aux_loss = outputs
                        aux_loss = aux_loss.mean()
                    else : 
                        aux_loss = torch.zeros(1).to(outputs.device)

                    outputs = resize_output(outputs, is_NEP)
                    loss = criterion(outputs, targets) 

            # Calculate metrics.
            batch_loss = loss.cpu().data.item()
            try : 
                batch_cf_matrix = utils.calculate_cf_matrix(outputs, targets)
            except Exception: 
                pdb.set_trace()

            # Measure time spent in computation.
            batch_end_time = datetime.datetime.now()
            compute_elapsed = batch_end_time - batch_start_time
            # Log.

            val_loss.append(batch_loss)
            cf_matrix_val = cf_matrix_val + batch_cf_matrix

            if idx == 0 : 
                torchvision.utils.save_image(inputs, 'val_sample_input.jpg')
                torchvision.utils.save_image(targets, 'val_sample_target.jpg')
                vis = inputs*0.5 + torchvision.transforms.Resize((inputs.size(-2), inputs.size(-1)))(targets)*0.5
                torchvision.utils.save_image(vis, 'val_sample.jpg')
                try: 
                    torchvision.utils.save_image(torch.nn.functional.sigmoid(outputs), 'val_sample_output.jpg')
                except : 
                    pdb.set_trace()
            if idx % 100 == 0 :
                logging.info((f"Validating, epoch #{epoch}/{args.epochs}, "
                            f"batch {idx + 1}/{args.val_iter}: "
                            f"loss {np.mean(val_loss):.2f}. "
                            f"compute {str(compute_elapsed)[-9:-4]}s"))

            if args.debug_mode and  idx == 10 : 
                break 

            if args.inference: 
                for input, target, output in zip(inputs, targets, outputs): 
                    # input: (3, 512, 512) float32 0~1
                    # target: (1, 128, 128) float32 0~1 
                    # output: (1, 128, 128) float16 -inf ~ inf
                    v1 = torch.nn.functional.interpolate(input.unsqueeze(0), scale_factor=0.25).squeeze(0)
                    v2 = target.expand((3, -1, -1))
                    v3 = torch.nn.functional.sigmoid(output).float().expand((3, -1, -1))
                    v = torch.cat([v1, v2, v3], dim=2)
                    torchvision.utils.save_image(v, os.path.join(args.visualize_path,f'visualize_{global_count}.jpg'))
                    global_count += 1 



        # Calculate metrics.


        logging.info("Calculate metrics.")
        elapsed = datetime.datetime.now() - start_time
        dice_coeff, accuracy, iou = utils.calculate_metrics(cf_matrix_val)

        if args.inference: 
            print(f'Dice: {dice_coeff:.3f}, IoU: {iou:.3f}, Acc: {accuracy:.3f}')
            pdb.set_trace()
            raise Exception

        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)

        log = "\n".join([
            f"{args.model_name} {_round}th on auto-hardmining process 2nd fold",
            f"Epoch #{epoch}/{args.epochs}",
            f"```",
            f"# time",
            f"- elapsed: {str(elapsed)}",
            f"# parameters",
            f"- current learning rate: {optimizer.param_groups[0]['lr']}",
            "\n".join([f"- {k}: {v}" for k, v in vars(args).items()]),
            f"# training",
            f"- loss: {avg_train_loss:.3f}",
            f"# validation",
            f"- loss: {avg_val_loss:.3f}",
            f"- dice coeff.: {dice_coeff:.3f}",
            f"- iou: {iou:.3f}",
            f"- pixel acc.: {accuracy:.3f}",
            f"- confusion matrix: \n {cf_matrix_val}",
            f"```"])

        try : 
            writer.add_scalar('train_loss', avg_train_loss, epoch )
            writer.add_scalar('val_loss', avg_val_loss, epoch )
            writer.add_scalar('pixel_acc', accuracy, epoch )
            writer.add_scalar('dice_coeff', dice_coeff, epoch )
            writer.add_scalar('iou', iou, epoch )
            print(log)
        except Exception as e: 
            print(e)
            pdb.set_trace()
        # Save checkpoint.
        logging.info("Save checkpoint.")

        score = dice_coeff
        checkpoint = Checkpoint(net, optimizer, epoch, args)

        checkpoint.save(os.path.join(args.log_dir, f"{args.ckpt_name}.{epoch}.pth.tar"))

        if score > best_score:

            checkpoint.save(os.path.join(args.log_dir, f"{args.ckpt_name}.best.pth.tar"))
            best_score = score
            print("Saving...")
        # Log epoch.
        send_slack_message(log)

    logging.info(log)
    logging.info("Finished training!")
 