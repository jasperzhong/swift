import timeit

import numpy as np
import torch
import torchvision.models as models

def main():
    m = 1
    torch.set_num_threads(m)
    x = torch.ones((m, 3, 224, 224))
    # x = torch.ones((m, 3, 64, 64))

    # model = models.resnet18()
    # model = models.squeezenet1_0()
    # model = models.densenet161()
    model = models.googlenet()
    # model = models.shufflenet_v2_x1_0()
    # model = models.mobilenet_v2()
    # model = models.resnext50_32x4d()
    # model = models.wide_resnet50_2()
    # model = models.mnasnet1_0()

    # model = models.segmentation.fcn_resnet50()
    # model = models.segmentation.fcn_resnet101()

    for _ in range(10):
        model(x)

    torch.set_autotune("random on")
    for _ in range(25):
        model(torch.ones((1, 3, 64, 64)))

    results = {
        "on": [],
        "off": [],
    }

    for i in range(40):
        if i in (2, 10):
            print("-" * 80)
            [i.clear() for i in results.values()]
        for autotune in ("off", "on"):
            torch.set_autotune(autotune)
            # torch.set_autotune("logging on")

            for _ in range(10):
                st = timeit.default_timer()
                model(x)
                results[autotune].append(timeit.default_timer() - st)
            print(
                f"{np.mean(results[autotune]) * 1000:5.1f} ms "
                f"({np.median(results[autotune]) * 1000:5.1f}) "
                f"{'AUTOTUNE' if autotune == 'on' else ''}   ", end="")
        print()

    # torch.set_autotune("on")
    # torch.set_autotune("logging on")
    # for _ in range(5):
    #     model(x)
    #     # torch.set_autotune("flush")
    #     # print()
    # torch.set_autotune("summarize")


if __name__ == "__main__":
    main()
