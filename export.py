from models.stereonet import StereoNet
import torch

model = StereoNet(1, "subtract").eval().cuda()
x = torch.randn(1, 3, 224, 224, requires_grad=True).cuda()
y = torch.randn(1, 3, 224, 224, requires_grad=True).cuda()
out = model(x,y)
torch.onnx.export(model, (x,y), 'stereonet.onnx', opset_version=13, training=torch.onnx.TrainingMode.PRESERVE, do_constant_folding=False, export_params=True)
