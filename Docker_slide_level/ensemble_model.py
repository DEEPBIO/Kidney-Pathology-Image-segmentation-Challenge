import os
import torch
import torch.nn as nn
from const import *
class ensemble_model(nn.Module) : 
    def __init__(self, models) : 
        super().__init__() 
        self.models = nn.ModuleList(models)
    
    def forward(self, x) : 
        outputs = torch.stack([model(x) for model in self.models], dim=0)
        outputs = torch.nn.functional.interpolate(outputs.squeeze(2), size=(INFERENCE_OUTPUT_SIZE * TARGET_RESIZE_RATIO,INFERENCE_OUTPUT_SIZE * TARGET_RESIZE_RATIO), mode='bilinear')
        outputs = torch.nn.functional.sigmoid(outputs)
        outputs = outputs.mean(0)
        outputs = (outputs>THRESHOLD).type(torch.float)
        return outputs

def prepare_model(model_name, model_path="/mnt/nfs4/users/old_data/90_user/users/iypaik/KPIs_240722_934/DETR_S.2.pth.tar", device='cuda'):
    from model import load_model
    model = load_model(model_name, class_num=1)
    ckpt = torch.load(model_path)['model_state']
    model.load_state_dict(ckpt)
    model.to(device=device)
    return torch.nn.DataParallel(model)

def load_model(ckpt_folder):
    model_list = []
    
    for filename in os.listdir(ckpt_folder):
        if filename.endswith("pth.tar"):
            model = prepare_model("DETR_S_MultiHead_Register", model_path=os.path.join(ckpt_folder, filename))
            model_list.append(model)
    print(len(model_list))
    model = ensemble_model(model_list)
    return model
