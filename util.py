

import os
from cfg import config,IMG_EXTENSIONS

import cv2
from skimage.measure import compare_psnr, compare_ssim
import tensorflow as tf

# def calc_psnr(im1, im2):
#     # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
#     # im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
#     return compare_psnr(im1, im2,data_range=im2.max()-im2.min())
#   #   print("ssim_multi_dims:",compare_psnr(im1, im2))
#   #   print("ssim_sigle_dim:",compare_psnr(im1_y,im2_y))
#
# def calc_ssim(im1, im2):
#     # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
#     # im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
#     return compare_ssim(im1,im2,data_range=im2.max()-im2.min(),multichannel=True)
def calc_psnr(im1, im2):

    if im1.shape[-1] is 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]


        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    else:
        im1 = im1.reshape(( 256, 256))
        im2 = im2.reshape(( 256, 256))


    return compare_psnr(im1, im2)


def calc_ssim(im1, im2):
    if im1.shape[-1] is 3:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]

        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    else:
        im1 = im1.reshape(( 256, 256))
        im2 = im2.reshape(( 256, 256))
    return compare_ssim(im1,im2)

def _read_py_function(filename, label):
  image_decoded = cv2.imread(filename.decode())

  label = cv2.imread(label.decode())

  return image_decoded, label

def normalization(X):

  return X / 127.5 - 1
# Use standard TensorFlow operations to resize the images to a fixed shape.
def _resize_function(image_decoded, label):

  image_decoded.set_shape([None, None, None])

  image_resized = tf.image.resize_images(image_decoded, [config.image_size,config.image_size])
  image_resized = normalization(image_resized)
  label.set_shape([None,None,None])

  label_resized = tf.image.resize_images(label,[config.image_size,config.image_size])
  label_resized = normalization(label_resized)
  return image_resized, label_resized


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def prepare_test_data(train_path):

    image_data = []
    image_gt = []

    train_data = train_path + "data/"
    train_gt = train_path + "gt/"

    for root, _, fnames in sorted(os.walk(train_data)):
        for fname in fnames:
            if is_image_file(fname):
                path_data = os.path.join(train_data, fname)
                path_gt = os.path.join(train_gt, fname)

                image_data.append(path_data)
                image_gt.append(path_gt)

    return  image_data, image_gt


def input_fn(val_batch_size,synthetic = False):

    if synthetic == False:
        val_filenames, val_labels = prepare_test_data(config.val_dataset+"/real/")
    else:
        val_filenames, val_labels = prepare_test_data(config.val_dataset+"/synthetic/")

    data_nums = len(val_filenames)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
    val_dataset = val_dataset.map(
        lambda filename, label: tuple(tf.py_func(
            _read_py_function, [filename, label], [tf.uint8, tf.uint8])))
    val_dataset = val_dataset.map(_resize_function)
    val_dataset = val_dataset.shuffle(buffer_size=val_batch_size*10)

    val_dataset = val_dataset.batch(val_batch_size)

    # with tf.Session() as sess:
    #     val_iteration = val_dataset.make_initializable_iterator()
    #     val_input = val_iteration.get_next()
    #
    #     sess.run(val_iteration.initializer)
    #     val_input = sess.run(val_input)

    return val_dataset,data_nums
