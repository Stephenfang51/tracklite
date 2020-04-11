<h1 align='center'>Tracklite</h1>

## Introduction

This repo using TensorRT to speed up yolov3 backbone and work with [deep_sort torch](https://github.com/ZQPei/deep_sort_pytorch).  mainly run on **Nvidia Jetson Nano** but x86 may also works. haven't tried yet. note that it is a inference pipeline not for training model.

currently only support yolov3 trt for now,  yolov3 tiny  will be released soon.

Thanks for [ZQPei](https://github.com/ZQPei)'s great work. and also thanks to [jkjung-avt](https://github.com/jkjung-avt) for his [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos), which give me a a lot to learn.

------

## Update

2020.4.11

> first upload the project

------

## Speed

Environment

- Jetson nano with TensorRT 5.1.6.1

Whole process time ( includes deepsort process )

| Backbone    | before TensorRT | TensorRT | FPS     |
| :---------- | --------------- | -------- | ------- |
| Yolov3_416  | 750ms           | 450ms    | 1.5 - 2 |
| yolov3-tiny | Soon            | Soon     | Soon    |

will add yolov3-tiny soon

------

## Install

follow my step to set up everything

1. clone this repo

```
git clone xxxx
```

2. Download YOLOv3 parameters

```
cd detector/YOLOv3/weight/
wget https://pjreddie.com/media/files/yolov3.weights
```

3. Download deepsort parameters ckpt.t7

```
cd deep_sort/deep/checkpoint
# download ckpt.t7 from
https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6 to this folder
```

4. Compile nms

```
cd detector/YOLOv3/nms
sh build.sh
```

------



## Convert yolov3 weights to onnx to tensorrt

1. firstly check the yolo weights under **weights directory** and just simply command like below to convert yolov3.weights file to onnx,  and onnx will be yielded at the same dir ( ./weights/yolov3_416.onnx )

   ```
   python3 yolov3_to_onnx.py
   ```

2. convert yolov3_416.onnx to tensorrt engine

   ```
   python3 onnx_to_tensorrt --onnx /path/to/your.onnx --output_engine /path/to/youroutput.engine
   ```

   **Note**: In `onnx_to_tensorrt.py` , you can set `max_workspace_size` = 1 << 30 in `get_engine` function and delete ` builder.fp16_mode = True` if you are using x86 arch for better performance (both mAP and frames per second)

------

## Demo

support video and webcam demo for now

1. Make sure everything is settled down
   - Yolov3_416 engine file
   - demo video you want to test on
2. Command line

- Webcam demo - onboard camera, csi camera

  ```
  python3 run_tracker.py
  ```

- Webcam demo - usb camera

  ```
  python3 run_tracker.py --usb
  ```

- Video demo

  ```
  python3 --file --filename your_test.mp4 --output_file ./output.mp4
  ```



![twice.gif](https://github.com/Stephenfang51/tracklite/blob/master/example/twice.gif)

------

## Issue 

I had a hard time on saving video, now the VideoWriter works for me, but it might not work for you, issue me if you have any problem.
