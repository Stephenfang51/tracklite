import tensorrt as trt
from utils import common
from utils.data_processing import *
from utils.draw import draw_boxes
from deep_sort import build_tracker

TRT_LOGGER = trt.Logger()


def get_engine(engine_file_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

class Tracker_tiny():
    def __init__(self, cfg, engine_file_path):
        self.cfg = cfg
        # self.args = args

        self.deepsort = build_tracker(cfg, use_cuda=True)
        #---tensorrt----#
        self.engine = get_engine(engine_file_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        # ---tensorrt----#

        #---input info for yolov3-416------#
        self.input_resolution_yolov3_HW = (416, 416)

        self.preprocessor = PreprocessYOLO(self.input_resolution_yolov3_HW)

        # self.image_raw, self.image = self.preprocessor.process(ori_im)

        # self.shape_orig_WH = image_raw.size
        #TODO tiny
        self.output_shapes = [(1, 255, 13, 13), (1, 255, 26, 26)]
        self.postprocessor_args = {"yolo_masks": [ (3, 4, 5), (0, 1, 2)],
                              # A list of 3 three-dimensional tuples for the YOLO masks
                              "yolo_anchors": [(10, 14), (23, 27), (37, 58),
                                            (81, 82), (135, 169), (344, 319)],
                              "obj_threshold": 0.6,  # Threshold for object coverage, float value between 0 and 1
                              "nms_threshold": 0.3,
                              # Threshold for non-max suppression algorithm, float value between 0 and 1
                              "yolo_input_resolution": self.input_resolution_yolov3_HW}

        self.postprocessor = PostprocessYOLO(**self.postprocessor_args)


    def run(self, ori_im):

        image_raw, image = self.preprocessor.process(ori_im)
        shape_orig_WH = image_raw.size
        # print('type of image:',  type(image))

        self.inputs[0].host = image
        trt_outputs = common.do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)

        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, self.output_shapes)]
        bbox_xywh, cls_ids, cls_conf = self.postprocessor.process(trt_outputs, (shape_orig_WH))

        if bbox_xywh is not None:
            # select person class
            mask = cls_ids == 0

            bbox_xywh = bbox_xywh[mask]
            bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]
            # print('hahahat', bbox_xywh.dtype)

            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, ori_im)

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities)


        return ori_im