# eval_trt(temp_demo)

import numpy as np
import torch
import torch.nn as nn
# tensorrt-lib
import util_trt
import glob,os,cv2

BATCH_SIZE = 16
BATCH = 100
CALIB_IMG_DIR = '/home/willer/yolov5-3.1/data/coco/images/train2017'
onnx_model_path = "/home/willer/nanodet_concat/tools/nanodet-simple.onnx"

height = 320
width = 320
img_mean = np.array([103.53, 116.28, 123.675])
img_std = np.array([57.375,  57.12,  58.395 ])

def preprocess_v1(image_raw):
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = width / w
    r_h = height / h
    if r_h > r_w:
        tw = width
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((height - th) / 2)
        ty2 = height - th - ty1
    else:
        tw = int(r_h * w)
        th = height
        tx1 = int((width - tw) / 2)
        tx2 = width - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (0,0,0)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (0, 0, 0)
    )
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image -= img_mean
    image /= img_std
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    return image

def preprocess(image_raw):
    h, w, c = image_raw.shape
    # Calculate widht and height and paddings
    if float(width) / float(image_raw.shape[1]) < float(height) / float(image_raw.shape[0]):
        ratio = float(width) / float(image_raw.shape[1])
    else:
        ratio = float(height) / float(image_raw.shape[0])
    # Resize the image with long side while maintaining ratio
    rz_image = cv2.resize(image_raw, (int(image_raw.shape[0]*ratio), int(image_raw.shape[1]*ratio)))
    # Pad the short side with (0,0,0)
    image = np.zeros((width,height,3),np.float32)
    image[0:int(image_raw.shape[1]*ratio),0:int(image_raw.shape[0]*ratio)] = rz_image
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image -= img_mean
    image /= img_std
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    return image

class DataLoader:
    def __init__(self):
        self.index = 0
        self.length = BATCH
        self.batch_size = BATCH_SIZE
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(CALIB_IMG_DIR, "*.jpg"))
        assert len(self.img_list) > self.batch_size * self.length, '{} must contains more than '.format(CALIB_IMG_DIR) + str(self.batch_size * self.length) + ' images to calib'
        print('found all {} images to calib.'.format(len(self.img_list)))
        self.calibration_data = np.zeros((self.batch_size,3,height,width), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = preprocess(img)
                self.calibration_data[i] = img

            self.index += 1

            # example only
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length

def main():
    # onnx2trt
    fp16_mode = True
    int8_mode = False 
    print('*** onnx to tensorrt begin ***')
    # calibration
    calibration_stream = DataLoader()
    engine_model_path = "models_save/nanodet_int8.trt"
    calibration_table = 'models_save/nanodet_calibration.cache'
    # fixed_engine,校准产生校准表
    engine_fixed = util_trt.get_engine(BATCH_SIZE, onnx_model_path, engine_model_path, fp16_mode=fp16_mode, 
        int8_mode=int8_mode, calibration_stream=calibration_stream, calibration_table_path=calibration_table, save_engine=True)
    assert engine_fixed, 'Broken engine_fixed'
    print('*** onnx to tensorrt completed ***\n')
    
if __name__ == '__main__':
    main()
    
