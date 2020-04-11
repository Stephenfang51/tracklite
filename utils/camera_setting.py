import logging
import threading
import subprocess

import numpy as np
import cv2

USB_GSTREAMER = True

def add_camera_args(parser):
    """Add parser augument for camera options."""
    parser.add_argument('--file', dest='use_file',
                        help='use a video file as input (remember to '
                        'also set --filename)',
                        action='store_true')
    parser.add_argument('--image', dest='use_image',
                        help='use an image file as input (remember to '
                        'also set --filename)',
                        action='store_true')
    parser.add_argument('--filename', dest='filename',
                        help='video file name, e.g. test.mp4',
                        default=None, type=str)
    parser.add_argument('--usb', dest='use_usb',
                        help='use USB webcam (remember to also set --vid)',
                        action='store_true')
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [0]',
                        default=0, type=int)
    parser.add_argument('--width', dest='image_width',
                        help='image width [640]',
                        default=640, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [480]',
                        default=480, type=int)

    return parser




def open_cam_usb(dev, width, height):
    """Open a USB webcam."""
    if USB_GSTREAMER:
        gst_str = ('v4l2src device=/dev/video{} ! '
                   'video/x-raw, width=(int){}, height=(int){} ! '
                   'videoconvert ! appsink').format(dev, width, height)
        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    else:
        return cv2.VideoCapture(dev)


def open_cam_onboard():
    """Open the Jetson onboard camera."""
    gst_str = ("nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)1280, height=(int)720, "
        "format=(string)NV12, framerate=(fraction)60/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink")
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def grab_img(cam):
    """This 'grab_img' function is designed to be run in the sub-thread.
    Once started, this thread continues to grab a new image and put it
    into the global 'img_handle', until 'thread_running' is set to False.
    """
    while cam.thread_running:
        _, cam.img_handle = cam.cap.read()
        if cam.img_handle is None:
            logging.warning('grab_img(): cap.read() returns None...')
            break
    cam.thread_running = False

class VideoWriter:
    def __init__(self, width, height, args, fps=24):
        # type: (str, int, int, int) -> None
        assert args.output_file.endswith('.mp4'), 'please specify the (.mp4) at the end '
        # self._name = name
        # self._height = height
        # self._width = width
        self.args = args
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.__writer = cv2.VideoWriter(args.output_file, fourcc, fps, (width, height))

    def write(self, frame):
        if frame.dtype != np.uint8:  # 检查frame的类型
            raise ValueError('frame.dtype should be np.uint8')
        self.__writer.write(frame)

    def release(self):
        self.__writer.release()


class Camera():
    """Camera class which supports reading images from theses video sources:

    1. Video file
    2. USB webcam
    3. Jetson onboard camera
    """

    def __init__(self, args):
        self.args = args
        self.is_opened = False
        self.use_thread = False
        self.thread_running = False
        self.img_handle = None
        self.img_width = 0
        self.img_height = 0
        self.cap = None
        self.thread = None
        #-----#
        self.vwriter = None

    def open(self):
        """Open camera based on command line arguments."""
        assert self.cap is None, 'Camera is already opened!'
        args = self.args
        if args.use_file: #video
            self.cap = cv2.VideoCapture(args.filename)
            # ignore image width/height settings here

            #TODO may occurs error since differnet opencv verison has different attribute name
            width, height = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.vwriter = VideoWriter(width=width, height=height, args=args, fps=24)
            self.use_thread = False
        # elif args.use_image:
        #     self.cap = 'OK'
        #     self.img_handle = cv2.imread(args.filename)
        #     # ignore image width/height settings here
        #     if self.img_handle is not None:
        #         self.is_opened = True
        #         self.img_height, self.img_width, _ = self.img_handle.shape
        #     self.use_thread = False

        elif args.use_usb:
            self.cap = open_cam_usb(
                args.video_dev,
                args.image_width,
                args.image_height
            )
            self.use_thread = True
        else:  # by default, use the jetson onboard camera
            self.cap = open_cam_onboard()
            print('using onboard cam now !')
            self.use_thread = True
        if self.cap != 'OK':
            if self.cap.isOpened():
                # Try to grab the 1st image and determine width and height
                _, img = self.cap.read()
                if img is not None:
                    self.img_height, self.img_width, _ = img.shape
                    self.is_opened = True

    #-------thread-----------------#
    def start(self):
        assert not self.thread_running
        if self.use_thread:
            self.thread_running = True
            self.thread = threading.Thread(target=grab_img, args=(self,))
            self.thread.start()

    def stop(self):
        self.thread_running = False
        if self.use_thread:
            self.thread.join()

    def read(self):
        if self.args.use_file:
            _, img = self.cap.read()
            # if img is None:
            #     #logging.warning('grab_img(): cap.read() returns None...')
            #     # looping around
            #     self.cap.release()
            #     self.cap = cv2.VideoCapture(self.args.filename)
            #     _, img = self.cap.read()
            return img
        elif self.args.use_image:
            return np.copy(self.img_handle)
        else:
            return self.img_handle

    def write(self, frame):
        if self.vwriter:
            self.vwriter.write(frame)

    def release(self):
        # if self.cap != 'OK':
        self.cap.release()
        if self.vwriter is not None:
            self.vwriter.release()









