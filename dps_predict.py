import argparse
import glob
import os
import numpy as np
from PIL import Image
import cv2
from skimage import restoration
import imageio
import tensorflow as tf
from models import *
from utils import prctile_norm, rm_outliers

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="test")
parser.add_argument("--folder_test", type=str, default="small")
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--na", type=str, default=3)
parser.add_argument("--pixelsize", type=str, default=32.5)
parser.add_argument("--exictation_lambda", type=str, default=488)
parser.add_argument("--gpu_memory_fraction", type=float, default=0.5)
parser.add_argument("--model_weights", type=str, default="weights/weights.latest.h5")

args = parser.parse_args()
gpu_id = args.gpu_id
gpu_memory_fraction = args.gpu_memory_fraction
data_dir = args.data_dir
folder_test = args.folder_test
model_weights = args.model_weights
na=args.na
pixelsize=args.pixelsize
ex_lambda=args.exictation_lambda

output_name = 'output_DPS'
test_images_path = data_dir + '/' + folder_test
output_dir = data_dir + '/' + output_name

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] ="1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
   try:
#     # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
       logical_gpus = tf.config.experimental.list_logical_devices('GPU')
       print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
       print(e)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
#     # select image
img_path = glob.glob(test_images_path + '/*.tif')
img_path.sort()
im_count = 0
n_channel = 1
#     # create output path
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
#     # define model and predict
for curp in img_path:
    img = np.array(imageio.imread(curp).astype(np.float64))
    factor=pixelsize*na*ex_lambda/(488*32.5*3)
    img=cv2.resize(img,dsize=None,fx=factor,fy=factor)
    background = restoration.rolling_ball(img,radius=50)
    img=img-background
    img = img[np.newaxis, :, :, np.newaxis]
    img = prctile_norm(img)
    i  = resunet_dbpn.model_pre(model_weights, n_channel, img ,im_count)
    pr = rm_outliers(prctile_norm(np.squeeze(i)))
    outName = curp.replace(test_images_path, output_dir)
    if not outName[-4:] == '.tif':
        outName = outName + '.tif'
    im = Image.fromarray(np.uint16(pr * 65535))
    im_count = im_count + 1
    im.save(outName)






