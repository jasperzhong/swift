import argparse
import json
import multiprocessing
import multiprocessing.dummy
import os
import queue
import subprocess
import tempfile
import timeit

import numpy as np
import torch
import torchvision.models as models


_TEST_MODELS = {
    "FasterRCNN_ResNet50": models.detection.fasterrcnn_resnet50_fpn,
    "MaskRCNN_ResNet50": models.detection.maskrcnn_resnet50_fpn,
    "KeyPointRCNN_ResNet50": models.detection.keypointrcnn_resnet50_fpn,

    "FCN_ResNet50": models.segmentation.fcn_resnet50,
    "FCN_ResNet101": models.segmentation.fcn_resnet101,
    "DeepLabV3_ResNet50": models.segmentation.deeplabv3_resnet50,
    "DeepLabV3_ResNet101": models.segmentation.deeplabv3_resnet101,

    "VGG11": models.vgg11,
    "VGG19": models.vgg19,
    "VGG11_BN": models.vgg11_bn,
    "VGG19_BN": models.vgg19_bn,
    "ResNet18": models.resnet18,
    "ResNet34": models.resnet34,
    "ResNet50": models.resnet50,
    "ResNet101": models.resnet101,
    "ResNet152": models.resnet152,
    "SqueezeNet_1.0": models.squeezenet1_0,
    "SqueezeNet_1.1": models.squeezenet1_1,
    "DenseNet_121": models.densenet121,
    "DenseNet_161": models.densenet161,
    "DenseNet_169": models.densenet169,
    "DenseNet_201": models.densenet201,
    "InceptionV3": models.inception_v3,
    "ResNext50_32x4d": models.resnext50_32x4d,
    "ResNext101_32x8d": models.resnext101_32x8d,
    "Wide_ResNet_50_2": models.wide_resnet50_2,
    "Wide_ResNet_101_2": models.wide_resnet101_2,
    "MNASNet1_0": models.mnasnet1_0,
}

_X_SIZES = {
    "1,3,64,64": (1, 3, 64, 64),
    "1,3,224,224": (1, 3, 224, 224),
}

_INVALID = {("InceptionV3", "1,3,64,64")}


_RESULT_FILE = "/tmp/torchvision_results.txt"
PYTHONPATH = os.getenv("PYTHONPATH")
assert PYTHONPATH is not None  # Subprocess will fail
_REPEATS = 5
_MAIN, _SUBPROCESS = "main", "subprocess"
_WORKERS = int(multiprocessing.cpu_count() / 2)
_AVAILABLE_CPUS = queue.Queue()
for i in range(_WORKERS):
    _AVAILABLE_CPUS.put(i * 2)


_SUBPROCESS_CMD_TEMPLATE = (
    "taskset --cpu-list {cpu} "
    f"python {__file__} "
    f"--DETAIL_context {_SUBPROCESS} "
    "--DETAIL_model {model} "
    "--DETAIL_size {size} "
    "--DETAIL_result_file {result_file} "
)


def benchmark_model(name, size, result_file):
    torch.set_num_threads(1)
    model = _TEST_MODELS[name]()
    model.eval()
    x = torch.ones(_X_SIZES[size])

    print("Begin warmup (Phase 1)")
    for _ in range(5):
        st = timeit.default_timer()
        model(torch.ones((1, 3, 96, 96)))
        print(timeit.default_timer() - st)

    print("Begin warmup (Phase 2)")
    torch.set_autotune("random on")
    for _ in range(5):
        st = timeit.default_timer()
        model(torch.ones((1, 3, 96, 96)))
        print(timeit.default_timer() - st)

    print("Warmup complete")
    results = {
        "on": [],
        "off": [],
    }

    for i in range(50):
        for autotune in ("off", "on"):
            torch.set_autotune(autotune)
            if result_file is None:
                torch.set_autotune("logging on")
            st = timeit.default_timer()
            model(x)
            results[autotune].append(timeit.default_timer() - st)
            print(f"{autotune:<3}  {results[autotune][-1]}")

    if result_file is not None:
        with open(result_file, "wt") as f:
            json.dump(results, f)
    else:
        pass
        # torch.set_autotune("flush")
        torch.set_autotune("summarize")


def run_subprocess(args):
    try:
        model, size_str = args
        cpu = _AVAILABLE_CPUS.get(timeout=120)
        _, result_file = tempfile.mkstemp(suffix=".json")
        cmd = _SUBPROCESS_CMD_TEMPLATE.format(
            cpu=cpu, model=model, size=size_str, result_file=result_file)

        subprocess.run(
            cmd,
            env={
                "PATH": os.getenv("PATH"),
                "PYTHONPATH": PYTHONPATH,
            },
            stdout=subprocess.PIPE,
            shell=True
        )

        with open(result_file, "rt") as f:
            return (args, json.load(f))
    except KeyboardInterrupt:
        pass
    except queue.Empty:
        cpu = None
        print(f"Failed to schedule: {str(args)}")
    finally:
        if cpu is not None:
            _AVAILABLE_CPUS.put(cpu)
        if os.path.exists(result_file):
            os.remove(result_file)


def subprocess_main(args):
    benchmark_model(
        name=args.DETAIL_model,
        size=args.DETAIL_size,
        result_file=args.DETAIL_result_file
    )


def main():
    tasks = [
        (m, size)
        for m in _TEST_MODELS.keys()
        for size in ["1,3,64,64", "1,3,224,224"]
    ] * _REPEATS
    tasks = [t for t in tasks if t not in _INVALID]
    results = []
    with multiprocessing.dummy.Pool(_WORKERS) as pool, open(_RESULT_FILE, "wt") as f:
        for i, r in enumerate(pool.imap_unordered(run_subprocess, tasks, 1)):
            results.append(r)
            f.write(f"{json.dumps(r)}\n")
            print(f"{i + 1} / {len(tasks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DETAIL_context", type=str, choices=(_MAIN, _SUBPROCESS), default=_MAIN)
    parser.add_argument("--DETAIL_model", type=str, default=None)
    parser.add_argument("--DETAIL_size", type=str, default=None)
    parser.add_argument("--DETAIL_result_file", type=str, default=None)
    args = parser.parse_args()

    if args.DETAIL_context == _MAIN:
        main()
    else:
        subprocess_main(args)
