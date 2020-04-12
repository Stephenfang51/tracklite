import sys
import argparse

from utils.camera_setting import *
from utils.parser import get_config
import cv2
import time

from tracker.tracker_tiny import Tracker_tiny

WINDOW_NAME = 'TrtYolov3_tiny_deepsort'

def parse_args():
    """Parse camera and input setting arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time MOT with TensorRT optimized '
            'YOLOv3 model on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    #TODO change default tiny engine
    parser.add_argument('--engine_path', type=str, default='./weights/yolov3_tiny_416.engine', help='set your engine file path to load')
    parser.add_argument('--config_deepsort', type=str, default="./configs/deep_sort.yaml")
    parser.add_argument('--output_file', type=str, default='./test.mp4', help='path to save your video like  ./test.mp4')

    args = parser.parse_args()
    return args

def open_window(window_name, width, height, title):
    """Open the display window."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.setWindowTitle(window_name, title)


def loop_and_track(cam, tracker, arg):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      tracker: the TRT YOLOv3 object detector instance.
    """

    if arg.filename:
        while True:
            if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
                break

            img = cam.read()
            if img is not None: #this line is a must in case not reading img correctly
                start = time.time()
                img_final = tracker.run(img)
                cv2.imshow(WINDOW_NAME, img_final)
                cam.write(img_final)
                end = time.time()
                print("time: {:.03f}s, fps: {:.03f}".format(end - start, 1 / (end - start)))
            key = cv2.waitKey(1)
            if key == 27:  # ESC key: quit program
                break


    else:
        while True:
            # if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            #     break
            img = cam.read()
            if img is not None: #this line is a must in case not reading img correctly
                start = time.time()
                img_final = tracker.run(img)
                cv2.imshow(WINDOW_NAME, img_final)
                end = time.time()
                print("time: {:.03f}s, fps: {:.03f}".format(end - start, 1 / (end - start)))
            key = cv2.waitKey(1)
            if key == 27:  # ESC key: quit program
                break




def main():
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)

    cam = Camera(args)
    cam.open()
    if not cam.is_opened:
        sys.exit('Failed to open camera!')

    tracker = Tracker_tiny(cfg, args.engine_path) #TODO

    cam.start()
    open_window(WINDOW_NAME, args.image_width, args.image_height,
                'TrtYolov3_deepsort')
    loop_and_track(cam, tracker, args)

    cam.stop()

    cam.release()
    cv2.destroyAllWindows()

    if args.filename:
        print('result video saved at (%s)' %(args.output_file))
    else:
        print('close')


if __name__ == '__main__':
    main()