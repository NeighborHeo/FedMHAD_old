# %%
# import sys
# sys.path.append('/home/suncheol/code/FedTest/FedMAD2')
# from mymodels.resnet8 import *
# from mymodels.vit_models import *
from .resnet8 import *
from .vit_models import *

# class mymodel():
#     def __init__(self, modelname = 'resnet8', num_classes=10, device=False, **kwargs):
#         super().__init__()
#         if modelname == 'resnet8':
#             self.model = ResNet8(num_classes=num_classes, **kwargs)
#         elif modelname == 'vit_tiny':
#             self.model = vit_tiny_patch16_224(num_classes=num_classes, **kwargs)
#         else:
#             raise ValueError('modelname should be resnet8 or vit_tiny')
        
#         if len(gpu)>1:
#              model = nn.DataParallel(model, device_ids=gpu)
    
    
#     def forward(self, x):
#         return self.model(x = x)
    
#     def get_attention_maps(self, x):
#         return None

def define_model(modelname, num_classes, **kwargs):
    if modelname == 'resnet8':
        model = ResNet8(num_classes=num_classes, **kwargs)
    elif modelname == 'vit_tiny_patch16_224':
        model = vit_tiny_patch16_224(num_classes=num_classes, **kwargs)
    elif modelname == 'vit_small_patch16_224':
        model = vit_small_patch16_224(num_classes=num_classes, **kwargs)
    elif modelname == 'vit_base_patch16_224':
        model = vit_base_patch16_224(num_classes=num_classes, **kwargs)
    else:
        raise ValueError('modelname should be resnet8 or vit_tiny')
    
    gpu = [int(i) for i in range(torch.cuda.device_count())]
    if len(gpu)>=1:
        model = nn.DataParallel(model, device_ids=gpu)
        
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model

if __name__ == '__main__':
    model = define_model(modelname='vit_tiny_patch16_224', num_classes=20)
    import utils.utils as utils
    #state_dict_path = '/home/suncheol/code/FedTest/FedMAD2/checkpoints/pascal_voc2012/vit_tiny_patch16_224_multilabel_only/a1.0+sd1+e300+b128+lkl+slmha/model-0.pth'
    state_dict_path = '/home/suncheol/code/FedTest/FedMAD2/checkpoints/pascal_voc2012/vit_tiny_patch16_224_multilabel_only/a1.0+sd1+e300+b128+lkl+slmha/model-0.pth'
    
    utils.load_dict(state_dict_path, model)
# %%
