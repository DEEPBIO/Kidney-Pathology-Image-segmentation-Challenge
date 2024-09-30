#!/usr/bin/env python3
"""Provide a unified API to Ellie.
"""
import logging
import torch

logger = logging.getLogger("deepbio.ellie-pytorch")

def load_model(model_name, class_num):
    torch.backends.cudnn.benchmark = True

    # if model_name == "UNet":
    #     model = UNet(class_num)

    # elif model_name == "FusionNet":
    #     model = FusionGenerator(3, class_num, 16)
    
    # elif model_name == "Deeplab_v3_plus":
    #     model = deeplab_v3_plus(class_num)

    # elif model_name == "Deeplab_bilie":
    #     model = deeplab_bilie(class_num)

    # elif model_name == "Deeplab_bilie2":
    #     model = deeplab_bilie_ver2(class_num)

    # elif model_name == "Deeplab_v3_plus_attention":
    #     model = deeplab_v3_plus_attention(class_num)

    # elif model_name == 'DETR_L' : 
    #     from .DETR import DETR_L
    #     model = DETR_L()

    # elif model_name == 'DETR_L_multi' : 
    #     from .DETR import DETR_L_multi
    #     model = DETR_L_multi()

    # elif model_name == 'DETR_L_MultiHead' : 
    #     from .DETR import DETR_L_MultiHead
    #     model = DETR_L_MultiHead(class_num)

    # elif model_name == 'DETR_L_MultiHead_SoftMoE_Register' : 
    #     from .DETR import DETR_L_MultiHead_SoftMoE_Register
    #     model = DETR_L_MultiHead_SoftMoE_Register(class_num)

    # elif model_name == 'DETR_L_MultiHead_Register' : 
    #     from .DETR import DETR_L_MultiHead_Register
    #     model = DETR_L_MultiHead_Register(class_num)

    # elif model_name == 'DETR_B_MultiHead_Register' : 
    #     from .DETR import DETR_B_MultiHead_Register
    #     model = DETR_B_MultiHead_Register(class_num)

    if model_name == 'DETR_S_MultiHead_Register' : 
        from .DETR import DETR_S_MultiHead_Register
        model = DETR_S_MultiHead_Register(class_num)

    # elif model_name == 'EfficientViT_L1':
    #     from .EfficientViT_seg import create_seg_model
    #     model = create_seg_model(name='l1', dataset='ade20k', pretrained=True, weight_url='checkpoint/efficientvit_l1_ade20k.pt')

    # elif model_name == 'EfficientViT_L2':
    #     from .EfficientViT_seg import create_seg_model
    #     model = create_seg_model(name='l2', dataset='ade20k', pretrained=True, weight_url='checkpoint/efficientvit_l2_ade20k.pt')
    elif model_name == "segnext":
        from .segnext import _SegNeXt
        model = _SegNeXt('tiny', class_num=1)
    else:
        raise NameError("Model Name Error")

    return model
