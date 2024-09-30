#!/usr/bin/env python3
import logging
import argparse
import os
import multiprocessing
import logging

from logging.handlers import QueueHandler, QueueListener
from deepbio.utils import get_logger

from train import train
from signal import signal, SIGPIPE, SIG_DFL

signal(SIGPIPE, SIG_DFL)



def logger_init():
    q = multiprocessing.Queue()
    handler = logging.StreamHandler()
    full_format = logging.Formatter(
                 '%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)')
    handler.setFormatter(full_format)
    ql = QueueListener(q, handler)
    ql.start()

    return ql, q


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__) 
    # model & checkpoint
    parser.add_argument("--model_name", default="DETR_S_MultiHead_Register",)
    parser.add_argument("--ckpt_name", default="DETR_S",)
    parser.add_argument("--checkpoint_file", default=None,)
    parser.add_argument("--resume", default=False, action="store_true",)
    parser.add_argument("--class_num", type=int, default=1)

    # training schedule 
    parser.add_argument("--batch_size", type=int, default=40,)    
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--optimizer", default='AdamW', choices=['SGD','AdamW'])
    parser.add_argument("--log_dir", default=None,)
    parser.add_argument("--num_workers", type=int, default=24,)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_iter", type=int, default=10000)
    parser.add_argument("--val_iter", type=int, default=0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--half_precision", default=True, action="store_false",)


    # Data                        
    parser.add_argument("--data_dir", default=None, 
                        help=("The directory containing dataset."))
    parser.add_argument("--patch_size", type=int, default=1024)
    parser.add_argument("--mag", type=int, default=100)
    parser.add_argument("--imagenet_dir", default=None,
                        help=("The directory containing Imagenet dataset dataset."))
    parser.add_argument("--imagenet_prob", default=0.1, type=float)
    parser.add_argument("--cancer_weight", default=2, type=float)
    parser.add_argument("--cutmix_prob", type=float, default=1.0)
    parser.add_argument("--mixup_prob", type=float, default=1.0)

    # Logging & detail 
    parser.add_argument("--loss_function", default="dice")
    parser.add_argument("--debug", default=False, action="store_true",
                        help="Print debug messages.")
    parser.add_argument("--reset_epoch", default=False, action="store_true",
                        help="Reset start epoch.")
    parser.add_argument("--error_recipient", default=None,
                        help=("The Slack username of the person to be notified"
                              " in case of error."))   
    parser.add_argument("--debug_mode", default=False, action="store_true",)
    parser.add_argument("--inference", default=False, action="store_true",)
    parser.add_argument("--visualize_path", default='./visualize', action="store_true",)

    args = parser.parse_args()

    import datetime 
    args.mask_downsample = 8 if ('EfficientViT' in args.model_name) else 4
    args.log_dir = os.path.join(args.log_dir, str(datetime.datetime.now()))
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.visualize_path, exist_ok=True)

    with open(os.path.join(args.log_dir, 'args.txt'), 'w') as file:
        for arg_name in vars(args):
            arg_value = getattr(args, arg_name)
            file.write(f"{arg_name}: {arg_value}\n")

    print(args)
    logger = get_logger(name="prostate", print_level="INFO",
                        file_level="INFO", slack_level=None,
                        error_recipient=args.error_recipient)

    resume_check = args.resume

    q_listener, q = logger_init()
    _round = 0 
    logger.info(f'round is {_round}')
    
    train(args, _round, resume_check)


    resume_check = False

    q_listener.stop()



