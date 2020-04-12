from __future__ import print_function
import sys
import os
import tensorrt as trt
import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default='./weights/yolov3_tiny_416.onnx', help='onnx file to convert')
    parser.add_argument('--output_engine', type=str, default='./weights/yolov3_tiny_416.engine', help="output path to output")

    args = parser.parse_args()

    return args


sys.path.insert(1, os.path.join(sys.path[0], ".."))


TRT_LOGGER = trt.Logger()


def get_engine(onnx_file_path, engine_file_path=""):
    # """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    # def build_engine():
    #     """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 28 # 256MB is for jetson nano
        builder.max_batch_size = 1
        builder.fp16_mode = True
        # Parse model file
        if not os.path.exists(onnx_file_path):
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
            exit(0)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())



def main():
    """Create a TensorRT engine for ONNX-based YOLOv3-416 and run inference."""
    args = parse_args()
    get_engine(args.onnx, args.output_engine)


if __name__ == '__main__':
    main()
