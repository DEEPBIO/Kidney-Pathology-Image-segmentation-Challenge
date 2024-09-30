#!/usr/bin/env python3
import torch 
def load_model(model_name, class_num):
    torch.backends.cudnn.benchmark = True
    if model_name == 'DETR_S_MultiHead_Register' : 
        from .DETR import DETR_S_MultiHead_Register
        model = DETR_S_MultiHead_Register(class_num)
    else:
        raise NameError("Model Name Error")

    return model
