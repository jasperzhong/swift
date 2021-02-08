import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

print(torch.__version__)

model = torch.jit.load("body_tracking_fp32.pt")
model.eval()
script_model = torch.jit.script(model)
script_model_vulkan = optimize_for_mobile(script_module, backend='Vulkan')
script_model_vulkan.save("body_tracking_fp32_opt.pt")
