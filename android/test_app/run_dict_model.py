import torch.nn as nn
import torch

print (torch.version.__version__)

MODEL_PATH = "app/src/main/assets/test_dict_str_tensor.pt"

module = torch.jit.load(MODEL_PATH)

d = {}
for i in range(5):
    d[str(i)] = torch.rand((1, 320, 320))
print("input:{}".format(d))
output = module.forward(d)
print("-"*80)
print("output:{}".format(output))