import torch.nn as nn
import torch

print (torch.version.__version__)

OUTPUT_DIR = "app/src/main/assets/"

class WrapRPN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features):
        # type: (Dict[str, Tensor]) -> Dict[str, Tensor]
        output : Dict[str, torch.Tensor] = {}
        i = 0
        for key in features.keys():
            output[str(i)] = features[key]
            i += 1
        return output

module = WrapRPN()
module.eval()

script_module = torch.jit.script(module)
outputFileName = OUTPUT_DIR + "test_dict_str_tensor.pt"
script_module.save(outputFileName)
print("Saved to " + outputFileName)
