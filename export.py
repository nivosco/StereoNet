from models import stereonet
import torch
from collections import OrderedDict


model = stereonet(1, "subtract").eval().cuda()
state_dict = torch.load('checkpoint_finetune_kitti15.tar')
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if 'module.' in k:
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict, strict=False)
x = torch.randn(1, 3, 224, 224, requires_grad=True).cuda()
y = torch.randn(1, 3, 224, 224, requires_grad=True).cuda()
out = model(x,y)
torch.onnx.export(model, (x,y), 'stereonet.onnx', opset_version=13, training=torch.onnx.TrainingMode.PRESERVE, do_constant_folding=False, export_params=True)
