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

## Code Reading
   Please refer to [Link to Code Reading](docs/reading.md).


## Lane Detection
   Please refer to [Link to Lane Detection](docs/lane.md).
