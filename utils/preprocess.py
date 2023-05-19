import os
import argparse
from pathlib import Path

from PIL import Image

from yolov5 import detect


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / '..'  # current directory

DATASETS = ROOT / 'datasets'
YOLO = ROOT / 'yolov5'
YOLO_OUTPUT = YOLO / 'runs/detect'

IMAGE_SQUARE_SIZE = 128

parser = argparse.ArgumentParser()
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='show results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--save-crop', default='true', action='store_true', help='save cropped prediction boxes')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--visualize', action='store_true', help='visualize features')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
opt = parser.parse_args()

opt.weights = YOLO / 'yolov5m.pt'
opt.data = YOLO / 'data/coco128.yaml'
# source = YOLO / 'data/images/10/Video/clap/hb-1-1-1-c01.dat'
# source = ROOT / '10/Video'
# project = YOLO / 'runs/detect'
opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand


# image_padding_width = 0
# image_padding_height = 0

"""""""""
only modify here
same as dataset_divider.py
"""""""""
# working = f'occlusion/board'
# working: str = f'occlusion/cabinet'
# working: str = f'occlusion/desk'
# working: str = f'light/10'
# working: str = f'light/100'
# working: str = f'light/dark'
working: str = f'occlusion/cabinet'


def preprocess(TEST=True):
    # source = ROOT / 'datasets/10/Video_test' if TEST else ROOT / 'datasets/10/Video'
    # project = YOLO / 'runs/detect/video_test' if TEST else YOLO / 'runs/detect/video'
    source = DATASETS / working / 'test' / 'Video' if TEST else DATASETS / working / 'train' / 'Video'
    project = YOLO_OUTPUT / working / 'test' if TEST else YOLO_OUTPUT / working / 'train'
    labels = os.listdir(source)

    # max_width, max_height = 0, 0
    datasets_path = DATASETS / working
    datasets_path = datasets_path / 'test' if TEST else datasets_path / 'train'
    datasets_path = datasets_path / 'crop'
    if not os.path.exists(datasets_path):
        os.mkdir(datasets_path)

    for label in labels:
        label_path = datasets_path / label
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        for person in os.listdir(source / label):
            opt.source = source / label / person
            opt.project = project / label / person

            detect.run(**vars(opt))
            # max_width, max_height = detect.run(**vars(opt))

            pre_path = label_path / person
            if not os.path.exists(pre_path):
                os.mkdir(pre_path)

            # if no person detected, create a symbolic link from crop to Video
            if not os.path.exists(opt.project / 'exp/crops'):
                for jpg in os.listdir(opt.source):
                    file_path = opt.source / jpg
                    save_path = pre_path / jpg
                    img = Image.open(file_path)
                    img.resize((IMAGE_SQUARE_SIZE, IMAGE_SQUARE_SIZE))
                    img.save(save_path)
                continue

            for name in os.listdir(opt.project / 'exp/crops'):
                file_path = opt.project / 'exp/crops' / name
                save_path = pre_path / name
                img = Image.open(file_path)
                img.resize((IMAGE_SQUARE_SIZE, IMAGE_SQUARE_SIZE))
                img.save(save_path)

    # global image_padding_height, image_padding_width
    # image_padding_width = max_width if max_width > image_padding_width else image_padding_width
    # image_padding_height = max_height if max_height > image_padding_height else image_padding_height

    # format images to the same size
    # with black occupied
    # print(image_padding_height, image_padding_width)


preprocess(True)
preprocess(False)
