import torch 
import torch.nn as nn
import torch.nn.functional as F
import warnings
import pdb 

# https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet50
def resnet50_pretrained(calc_layer4=False): 
    from torchvision.models import resnet50
    model = resnet50("IMAGENET1K_V2")

    if calc_layer4 : 

        class forward_wrap(nn.Module) : 
            def __init__(self, model) : 
                super().__init__()
                self.model = model 
            
            def forward(self, x) : 

                x = self.model.conv1(x)
                x = self.model.bn1(x)
                x = self.model.relu(x)
                x = self.model.maxpool(x)

                features = [] 
                x = self.model.layer1(x) # (B, 256, 128, 128)
                features.append(x)
                x = self.model.layer2(x) # (B, 512, 64, 64)
                features.append(x)
                x = self.model.layer3(x) # (B, 1024, 32, 32)
                features.append(x)
                x = self.model.layer4(x) # (B, ?, 16, 16)
                features.append(x)
                return features 

        return forward_wrap(model) 
    else : 
        class forward_wrap(nn.Module) : 
            def __init__(self, model) : 
                super().__init__()
                self.model = model 
            
            def forward(self, x) : 

                x = self.model.conv1(x)
                x = self.model.bn1(x)
                x = self.model.relu(x)
                x = self.model.maxpool(x)

                features = [] 
                x = self.model.layer1(x) # (B, 256, 128, 128)
                features.append(x)
                x = self.model.layer2(x) # (B, 512, 64, 64)
                features.append(x)
                x = self.model.layer3(x) # (B, 1024, 32, 32)
                features.append(x)

                return features 

        return forward_wrap(model) 



def vit_l_pretrained(): 
    from torchvision.models import vit_l_16
    model = vit_l_16("IMAGENET1K_SWAG_E2E_V1")
    class forward_wrap(nn.Module) : 
        def __init__(self, model) : 
            super().__init__()
            self.model = model 
        
        def forward(self, x) : 
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = self.model.encoder(x)
            return x 

    return forward_wrap(model) 

def vit_l_pretrained_moe(): 
    try: 
        from .vision_transformer_moe import vit_l_16
    except ImportError: 
        from vision_transformer_moe import vit_l_16

    model = vit_l_16("IMAGENET1K_SWAG_E2E_V1") # need to insert the value in order to override the internal parameter
    state_dict = torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/vit_l_16_swag-4f3808c9.pth")
    result = model.load_state_dict(state_dict, strict=False)
    #print(result)

    class forward_wrap(nn.Module) : 
        def __init__(self, model) : 
            super().__init__()
            self.model = model 
        
        def forward(self, x) : 
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x, aux_loss = self.model.encoder(x)
            return x, aux_loss

    return forward_wrap(model) 


def vit_l_pretrained_soft_moe_register(): 
    try: 
        from .vision_transformer_soft_moe_register import vit_l_16
    except ImportError: 
        from vision_transformer_soft_moe_register import vit_l_16

    model = vit_l_16("IMAGENET1K_SWAG_E2E_V1") # need to insert the value in order to override the internal parameter
    state_dict = torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/vit_l_16_swag-4f3808c9.pth")
    result = model.load_state_dict(state_dict, strict=False)
    print(result)

    class forward_wrap(nn.Module) : 
        def __init__(self, model) : 
            super().__init__()
            self.model = model 
        
        def forward(self, x) : 
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.model.class_token.expand(n, 1, -1)
            batch_register_token = self.model.register_token.expand(n, 4, -1)
            x = torch.cat([batch_register_token, batch_class_token, x], dim=1)

            x = self.model.encoder(x)
            return x

    return forward_wrap(model) 

def vit_l_pretrained_register(): 
    try: 
        from .vision_transformer_register import vit_l_16
    except ImportError: 
        from vision_transformer_register import vit_l_16

    model = vit_l_16("IMAGENET1K_SWAG_E2E_V1") # need to insert the value in order to override the internal parameter
    state_dict = torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/vit_l_16_swag-4f3808c9.pth")
    result = model.load_state_dict(state_dict, strict=False)
    print(result)

    class forward_wrap(nn.Module) : 
        def __init__(self, model) : 
            super().__init__()
            self.model = model 
        
        def forward(self, x) : 
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.model.class_token.expand(n, 1, -1)
            batch_register_token = self.model.register_token.expand(n, 4, -1)
            x = torch.cat([batch_register_token, batch_class_token, x], dim=1)

            x = self.model.encoder(x)
            return x

    return forward_wrap(model) 



def vit_b_pretrained_register(): # dim=768
    try: 
        from .vision_transformer_register import vit_b_16
    except ImportError: 
        from vision_transformer_register import vit_b_16

    model = vit_b_16("IMAGENET1K_SWAG_E2E_V1") # need to insert the value in order to override the internal parameter
    state_dict = torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/vit_b_16_swag-9ac1b537.pth")
    result = model.load_state_dict(state_dict, strict=False)
    print(result)

    class forward_wrap(nn.Module) : 
        def __init__(self, model) : 
            super().__init__()
            self.model = model 
        
        def forward(self, x) : 
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.model.class_token.expand(n, 1, -1)
            batch_register_token = self.model.register_token.expand(n, 4, -1)
            x = torch.cat([batch_register_token, batch_class_token, x], dim=1)

            x = self.model.encoder(x)
            return x

    return forward_wrap(model) 


def vit_s_register(): # dim=768
    try: 
        from .vision_transformer_register import VisionTransformer
    except ImportError: 
        from vision_transformer_register import VisionTransformer

    model = VisionTransformer(image_size=512, patch_size=16, num_layers=12, num_heads=6, hidden_dim=384, mlp_dim=1536,)

    class forward_wrap(nn.Module) : 
        def __init__(self, model) : 
            super().__init__()
            self.model = model 
        
        def forward(self, x) : 
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.model.class_token.expand(n, 1, -1)
            batch_register_token = self.model.register_token.expand(n, 4, -1)
            x = torch.cat([batch_register_token, batch_class_token, x], dim=1)

            x = self.model.encoder(x)
            return x

    return forward_wrap(model) 

def vit_l_pretrained_glu(): 
    try: 
        from .vision_transformer_GLU import vit_l_16
    except ImportError: 
        from vision_transformer_GLU import vit_l_16

    model = vit_l_16("IMAGENET1K_SWAG_E2E_V1") # need to insert the value in order to override the internal parameter
    state_dict = torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/vit_l_16_swag-4f3808c9.pth")
    result = model.load_state_dict(state_dict, strict=False)
    print(result)

    class forward_wrap(nn.Module) : 
        def __init__(self, model) : 
            super().__init__()
            self.model = model 
        
        def forward(self, x) : 
            n = x.shape[0]

            # Expand the class token to the full batch
            batch_class_token = self.model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)

            x = self.model.encoder(x)
            return x 


    return forward_wrap(model) 


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)
class Upsample(nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)

# from: https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/decode_heads/setr_up_head.py#L11
class SETRUPHead(nn.Module):
    """Naive upsampling head and Progressive upsampling head of SETR.

    Naive or PUP head of `SETR  <https://arxiv.org/pdf/2012.15840.pdf>`_.

    Args:
        norm_layer (dict): Config dict for input normalization.
            Default: norm_layer=dict(type='LN', eps=1e-6, requires_grad=True).
        num_convs (int): Number of decoder convolutions. Default: 1.
        up_scale (int): The scale factor of interpolate. Default:4.
        kernel_size (int): The kernel size of convolution when decoding
            feature information from backbone. Default: 3.
        init_cfg (dict | list[dict] | None): Initialization config dict.
            Default: dict(
                     type='Constant', val=1.0, bias=0, layer='LayerNorm').
    """

    def __init__(self, in_channels=1024, channels=256, num_classes=1, 
                 num_convs=2,
                 up_scale=2,
                 kernel_size=3,):

        assert kernel_size in [1, 3], 'kernel_size must be 1 or 3.'

        super().__init__()

        assert isinstance(in_channels, int)

        self.norm = torch.nn.LayerNorm(in_channels)

        self.up_convs = nn.ModuleList()
        for _ in range(num_convs):
            self.up_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=1, padding = int(kernel_size - 1) // 2), 
                    nn.BatchNorm2d(channels), 
                    nn.GELU(), 
                    Upsample(
                        scale_factor=up_scale,
                        mode='bilinear',
                        align_corners=None)))
            in_channels = channels
        self.cls_seg = nn.Sequential(
            nn.Dropout(0.1), 
            nn.Conv2d(channels, num_classes, kernel_size=1)
        )

    def forward(self, x):

        #n, c, h, w = x.shape
        #x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
        x = self.norm(x) # (n, h*w, c)
        n, hw, c = x.shape 
        h = round(hw**.5)
        x = x.transpose(1, 2).reshape(n, c, h, h).contiguous()

        for up_conv in self.up_convs:
            x = up_conv(x)
        out = self.cls_seg(x)
        return out


class DETR_L(nn.Module): 
    def __init__(self, num_classes=1): 
        super().__init__()
        self.resnet = resnet50_pretrained()
        self.vit = vit_l_pretrained() 
        self.head = SETRUPHead(in_channels=1024, channels=256, num_classes= num_classes)
    def forward(self, x) : 
        features = self.resnet(x)
        x = features[2] # (B, 1024, 32, 32)
        B, C, H, W = x.shape
        x = x.view(B,C,-1).contiguous() # (B, 1024, 32*32)
        x = x.permute(0,2,1) # (B, 32*32, 1024)
        x = self.vit(x) # (B, 32*32+1, 1024)
        cls_token, x = torch.split(x, [1,1024], dim=1) # (B, 1, 1024), (B, 32*32, 1024)
        x = self.head(x)
        x = F.sigmoid(x)
        return x 


class DETR_L_multi(nn.Module): 
    def __init__(self, num_classes=1): 
        super().__init__()
        self.resnet = resnet50_pretrained(calc_layer4=True)
        self.vit = vit_l_pretrained() 
        original_pos_embedding = self.vit.model.encoder.pos_embedding.data # (1, 1025, 1024), first one is for cls_token. 
        new_pos_embedding = torch.empty(1, 1+1024+256, 1024).normal_(std=0.02)
        new_pos_embedding[:,:1025] = original_pos_embedding
        self.vit.model.encoder.pos_embedding = nn.Parameter(new_pos_embedding)
        self.head = SETRUPHead(in_channels=1024, channels=256, num_classes= num_classes)
        self.linear = nn.Sequential(
                    nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding = 0),
                    nn.GELU(),)
        
    def forward(self, x) : 
        features = self.resnet(x)
        x1 = features[2] # (B, 1024, 32, 32)
        B, C, H, W = x1.shape
        x1 = x1.view(B,C,-1).contiguous() # (B, 1024, 32*32)
        x1 = x1.permute(0,2,1) # (B, 32*32, 1024)

        x2 = features[3] # (B, 2048, 16, 16)
        x2 = self.linear(x2) #(B, 1024, 16, 16)
        B, C, H, W = x2.shape
        x2 = x2.view(B,C,-1).contiguous() # (B, 1024, 16*16)
        x2 = x2.permute(0,2,1) # (B, 16*16, 1024)


        x = torch.concatenate([x1,x2], dim=1) # (B, 1024+256, 1024)

        x = self.vit(x) # (B, 1+1024+256, 1024)
        cls_token, x, _ = torch.split(x, [1,1024, 256], dim=1) # (B, 1, 1024), (B, 32*32, 1024)
        x = self.head(x)
        x = F.sigmoid(x)
        return x 



class DETR_L_MoE(nn.Module): 
    def __init__(self, num_classes=1): 
        super().__init__()
        self.resnet = resnet50_pretrained()
        self.vit = vit_l_pretrained_moe() 
        self.head = SETRUPHead(in_channels=1024, channels=256, num_classes= num_classes)
    def forward(self, x) : 
        features = self.resnet(x)
        x = features[2] # (B, 1024, 32, 32)
        B, C, H, W = x.shape
        x = x.view(B,C,-1).contiguous() # (B, 1024, 32*32)
        x = x.permute(0,2,1) # (B, 32*32, 1024)
        x, aux_loss = self.vit(x) # (B, 32*32+1, 1024)
        cls_token, x = torch.split(x, [1,1024], dim=1) # (B, 1, 1024), (B, 32*32, 1024)
        x = self.head(x)
        #x = F.sigmoid(x)
        return x, aux_loss



class DETR_L_MultiHead(nn.Module): 
    def __init__(self, num_classes=1): 
        super().__init__()
        self.resnet = resnet50_pretrained()
        self.vit = vit_l_pretrained() 
        self.heads = nn.ModuleList([SETRUPHead(in_channels=1024, channels=256, num_classes= 1) for _ in range(num_classes)])
    def forward(self, x) : 
        features = self.resnet(x)
        x = features[2] # (B, 1024, 32, 32)
        B, C, H, W = x.shape
        x = x.view(B,C,-1).contiguous() # (B, 1024, 32*32)
        x = x.permute(0,2,1) # (B, 32*32, 1024)
        x = self.vit(x) # (B, 32*32+1, 1024)
        cls_token, x = torch.split(x, [1,1024], dim=1) # (B, 1, 1024), (B, 32*32, 1024)
        outputs = [head(x) for head in self.heads]
        x = torch.cat(outputs, dim=1)
        #x = F.sigmoid(x)
        return x 




class DETR_L_MultiHead_SoftMoE_Register(nn.Module): 
    def __init__(self, num_classes=1): 
        super().__init__()
        self.resnet = resnet50_pretrained()
        self.vit = vit_l_pretrained_soft_moe_register() 
        self.heads = nn.ModuleList([SETRUPHead(in_channels=1024, channels=256, num_classes= 1) for _ in range(num_classes)])
    def forward(self, x) : 
        features = self.resnet(x)
        x = features[2] # (B, 1024, 32, 32)
        B, C, H, W = x.shape
        x = x.view(B,C,-1).contiguous() # (B, 1024, 32*32)
        x = x.permute(0,2,1) # (B, 32*32, 1024)
        x = self.vit(x) # (B, 32*32+1, 1024)
        register_token, cls_token, x = torch.split(x, [4,1,1024], dim=1) # (B, 1, 1024), (B, 32*32, 1024)
        outputs = [head(x) for head in self.heads]
        x = torch.cat(outputs, dim=1)
        #x = F.sigmoid(x)
        return x 

class DETR_L_MultiHead_Register(nn.Module): 
    def __init__(self, num_classes=1): 
        super().__init__()
        self.resnet = resnet50_pretrained()
        self.vit = vit_l_pretrained_register() 
        self.heads = nn.ModuleList([SETRUPHead(in_channels=1024, channels=256, num_classes= 1) for _ in range(num_classes)])
    def forward(self, x) : 
        features = self.resnet(x)
        x = features[2] # (B, 1024, 32, 32)
        B, C, H, W = x.shape
        x = x.view(B,C,-1).contiguous() # (B, 1024, 32*32)
        x = x.permute(0,2,1) # (B, 32*32, 1024)
        x = self.vit(x) # (B, 5+32*32, 1024)
        register_token, cls_token, x = torch.split(x, [4,1,1024], dim=1) # (B, 1, 1024), (B, 32*32, 1024)
        outputs = [head(x) for head in self.heads]
        x = torch.cat(outputs, dim=1)
        #x = F.sigmoid(x)
        return x 

class DETR_B_MultiHead_Register(nn.Module): 
    def __init__(self, num_classes=1): 
        super().__init__()
        self.resnet = resnet50_pretrained()
        self.vit = vit_b_pretrained_register() 
        self.heads = nn.ModuleList([SETRUPHead(in_channels=768, channels=256, num_classes= 1) for _ in range(num_classes)])
        self.linear = nn.Linear(1024, 768)
    def forward(self, x) : 
        features = self.resnet(x)
        x = features[2] # (B, 1024, 32, 32)
        B, C, H, W = x.shape
        x = x.view(B,C,-1).contiguous() # (B, 1024, 32*32)
        x = x.permute(0,2,1) # (B, 32*32, 1024)
        x = self.linear(x) # (B, 32*32, 768)
        x = self.vit(x) # (B, 5+32*32, 768)
        register_token, cls_token, x = torch.split(x, [4,1,1024], dim=1) 
        outputs = [head(x) for head in self.heads]
        x = torch.cat(outputs, dim=1)
        #x = F.sigmoid(x)
        return x 


class DETR_S_MultiHead_Register(nn.Module): 
    def __init__(self, num_classes=1): 
        super().__init__()
        self.resnet = resnet50_pretrained()
        self.vit = vit_s_register() 
        self.heads = nn.ModuleList([SETRUPHead(in_channels=384, channels=256, num_classes= 1) for _ in range(num_classes)])
        self.linear = nn.Linear(1024, 384)
    def forward(self, x) : 
        features = self.resnet(x)
        x = features[2] # (B, 1024, 32, 32)
        B, C, H, W = x.shape
        x = x.view(B,C,-1).contiguous() # (B, 1024, 32*32)
        x = x.permute(0,2,1) # (B, 32*32, 1024)
        x = self.linear(x) # (B, 32*32, 768)
        x = self.vit(x) # (B, 5+32*32, 768)
        register_token, cls_token, x = torch.split(x, [4,1,H*W], dim=1) 
        outputs = [head(x) for head in self.heads]
        x = torch.cat(outputs, dim=1)
        #x = F.sigmoid(x)
        return x 

if __name__ == '__main__' : 
    import pdb 
    print(13)
    model = DETR_S_MultiHead_Register().cuda() 
    optimizer = torch.optim.SGD(model.parameters(), lr=100, momentum=0.9, weight_decay=1e-4)
    #model = torch.nn.DataParallel(model)
    x = torch.randn(6,3,1024,1024).cuda()
    #print(model.vit.model.encoder.layers[0].mlp.gate.w_gating)
    with torch.autocast(device_type='cuda', dtype=torch.float16): 
        y = model(x) 
        loss = y.mean()
        loss.backward()
        optimizer.step()
    #print(model.vit.model.encoder.layers[0].mlp.gate.w_gating)
    print(y.shape)
    pdb.set_trace()
