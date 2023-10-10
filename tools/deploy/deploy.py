from torch2onnx import torch2onnx
from onnx2tRt import onnx2trt


def main():
    onnx_file = './model_best.onnx'
    trt_file = './model_best.trt'
    onnx2trt(onnx_file, trt_file)



if __name__ == "__main__":
    main()