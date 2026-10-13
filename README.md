# CondLSTR: 2D and 3D Lane Detection

This repository contains the ICCV 2023 CondLSTR implementation for 2D lane
detection and the 3D journal extension. The 3D model uses lane representations
as the key/value memory for BEV construction and generates the dynamic detection
kernels used in both image and BEV views.

The journal implementation, exact configurations, OpenLane training/evaluation
entry points, complexity profiler, and controlled variants are under
[`experiments/ijcv_3d`](experiments/ijcv_3d). Dataset adapters and official
evaluator wrappers for OpenLane, ONCE-3DLanes, and ApolloSim are under
[`data/datasets/lane`](data/datasets/lane) and
[`tools/metrics/lane`](tools/metrics/lane).

## Framework
   ![The framwork of CondLSTR](docs/framework.jpg)

## Installation
   1. Create python environment:
      ```
      conda create -n hdmapnet python=3.10
      conda activate hdmapnet
      ```
   2. Install pytorch (>=2.0.0):
      ```
      pip install torch torchvision torchaudio torchdata
      ```
   3. Install mmcv (>=2.0.0):
      ```
      pip install -U openmim
      mim install mmcv
      ```
   4. Install requirements:
      ```
      pip install -r requirements.txt
      ```
   5. Install cuda11.7 (add to your ~/.bashrc and then `source ~/.bashrc`)
      ```
      export CUDA_HOME=/usr/local/cuda-11
      export PATH=$PATH:$CUDA_HOME/bin
      export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
      ```

## Prepare Dataset
   1. Put your dataset into `/path/to/datasets/`
   2. Rename the dataset directory with the parameter after `-d`, e.g., `/path/to/datasets/openlane`
   3. A more convenient way is `ln -s /path/to/datasets/original_name /path/to/datasets/openlane`
   4. The directory structure should look like this:
      ```
      /path/to/datasets/
      ├── openlane
      │   ├── images
      │   │   ├── training
      │   │   └── validation
      │   ├── lane3d_1000
      │   │   ├── training
      │   │   ├── validation
      │   │   └── test
      │   │       ├── curve_case
      │   │       ├── merge_split_case
      │   │       └── ...
      │   └── lane3d_3000.tar
      ├── curvelanes
      ├── culane
      └── tusimple
      ```

## Install Evaluation Tools
   1. Install opencv:
      ```
      git clone https://github.com/opencv/opencv.git
      cd ~/opencv
      mkdir build
      cd build
      cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
      make -j7
      sudo make install
      ```
   2. Install openlane evaluation tool:
      ```
      cd /path/to/project/tools/metrics/lane/openlane/lane2d
      # specify your opencv path in ./Makefile#L40,41 (OPENCV_INCLUDE and OPENCV_LIB_PATH)
      make
      ```


## Training & Evaluation

### Journal 3D model

The released configuration uses a `368 x 480` input, a `50 x 32` BEV feature,
a `400 x 256` output raster, 80 lane queries, and 50 training epochs. Prepare
the OpenLane cache once:

```bash
python data/datasets/lane/openlane/preprocess.py \
  --root /path/to/datasets/openlane --version 3d --num-workers 32
```

Train the complete model on four GPUs:

```bash
torchrun --standalone --nproc-per-node=4 -m experiments.ijcv_3d.train \
  --data-root /path/to/datasets/openlane \
  --config experiments/ijcv_3d/configs/complete.json \
  --output /path/to/runs/condlstr_3d
```

Evaluate a checkpoint with the official OpenLane evaluator:

```bash
python -m experiments.ijcv_3d.evaluate \
  --data-root /path/to/datasets/openlane \
  --config experiments/ijcv_3d/configs/complete.json \
  --checkpoint /path/to/runs/condlstr_3d/epoch_050.pt \
  --output /path/to/runs/condlstr_3d/evaluation
```

Measure parameters, attributed FLOPs, synchronized mixed-precision batch-1
latency, and peak allocated memory. The profiler uses FP16 on V100 and BF16 on
GPUs with native BF16 support, and records the selected precision in
`protocol.json`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.ijcv_3d.profile \
  --data-root /path/to/datasets/openlane \
  --config experiments/ijcv_3d/configs/complete.json \
  --checkpoint /path/to/runs/condlstr_3d/epoch_050.pt \
  --output /path/to/runs/condlstr_3d/profile
```

The controlled configurations are `3d_only.json`, `direct_3d.json`,
`generic_tokens.json`, and `dense_image.json`. For the Same-O direct control,
initialize and freeze the common representation path:

```bash
torchrun --standalone --nproc-per-node=4 -m experiments.ijcv_3d.train \
  --data-root /path/to/datasets/openlane \
  --config experiments/ijcv_3d/configs/direct_3d.json \
  --init-common-from /path/to/runs/condlstr_3d/epoch_050.pt \
  --freeze-common --output /path/to/runs/direct_3d
```

Run the configuration and CPU loss tests with:

```bash
python -m unittest discover -s experiments/ijcv_3d/tests -v
```

### Conference 2D model
   1. Training debug:
      ```
      python tools/train.py -a CondLSTR2DRes34 -d openlane -v 2d -c 21 -t lane_det_2d --data-dir /path/to/datasets/  --logs-dir /path/to/checkpoint -b 4 -j 4
      ```
   2. Evaluation debug:
      ```
      python tools/train.py -a CondLSTR2DRes34 -d openlane -v 2d -c 21 -t lane_det_2d --data-dir /path/to/datasets/  --logs-dir /path/to/checkpoint -b 4 -j 4 --eval
      ```
   3. Test/visualization debug:
      ```
      python tools/test.py -a CondLSTR2DRes34 -d openlane -v 2d -c 21 -t lane_det_2d --data-dir /path/to/datasets/  --logs-dir /path/to/checkpoint --test-dir /path/to/output -b 4 -j 4
      ```
   4. Multi-gpu training:
      ```
      python -m torch.distributed.launch --nproc_per_node=4 tools/train.py -a CondLSTR2DRes34 -d openlane -v 2d -c 21 -t lane_det_2d --data-dir /path/to/datasets/  --logs-dir /path/to/checkpoint --gpu-ids 0,1,2,3 -b 4 -j 4
      ```
   5. Multi-gpu evaluation:
      ```
      python -m torch.distributed.launch --nproc_per_node=4 tools/train.py -a CondLSTR2DRes34 -d openlane -v 2d -c 21 -t lane_det_2d --data-dir /path/to/datasets/  --logs-dir /path/to/checkpoint --gpu-ids 0,1,2,3 -b 4 -j 4 --eval
      ```
   6. Multi-gpu test/visualization:
      ```
      python -m torch.distributed.launch --nproc_per_node=4 tools/test.py -a CondLSTR2DRes34 -d openlane -v 2d -c 21 -t lane_det_2d --data-dir /path/to/datasets/  --logs-dir /path/to/checkpoint --test-dir /path/to/output --gpu-ids 0,1,2,3 -b 4 -j 4
      ```
   7. Resume from checkpoint:
      ```
      python -m torch.distributed.launch --nproc_per_node=4 tools/train.py -a CondLSTR2DRes34 -d openlane -v 2d -c 21 -t lane_det_2d --data-dir /path/to/datasets/  --logs-dir /path/to/checkpoint --gpu-ids 0,1,2,3 -b 4 -j 4 --resume
      ```
   8. Training with AMP (Automated Mixed Precision):
      ```
      python -m torch.distributed.launch --nproc_per_node=4 tools/train.py -a CondLSTR2DRes34 -d openlane -v 2d -c 21 -t lane_det_2d --data-dir /path/to/datasets/  --logs-dir /path/to/checkpoint --gpu-ids 0,1,2,3 -b 4 -j 4 -p amp
      ```
   9. Finetune with a pretrained model (Put its `checkpoint.pth.tar` under `/path/to/checkpoint`):
      ```
      python -m torch.distributed.launch --nproc_per_node=4 tools/train.py -a CondLSTR2DRes34 -d openlane -v 2d -c 21 -t lane_det_2d --data-dir /path/to/datasets/  --logs-dir /path/to/checkpoint --gpu-ids 0,1,2,3 -b 4 -j 4 --resume --load-model-only
      ```

## Test
   1. OpenLane
      ```
      python -m torch.distributed.launch --nproc_per_node=4 tools/test.py -a CondLSTR2DRes34 -d openlane -v 2d -c 21 -t lane_det_2d --data-dir /path/to/datasets/  --logs-dir /path/to/checkpoint --test-dir /path/to/checkpoint --gpu-ids 0,1,2,3 -b 4 -j 4
      python tools/metrics/lane/openlane/openlane_2d.py --pred-dir /path/to/output --anno-dir /path/to/datasets/openlane --translate
      ```
   2. CurveLanes


