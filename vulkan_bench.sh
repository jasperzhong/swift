#!/bin/bash

./build/bin/speed_benchmark_torch --model=maskrcnn_lite_opt.pt --input_type=float --input_dims=1,3,224,224 --btinput=true --vulkan=true
