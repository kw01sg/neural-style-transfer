import tensorflow as tf
import numpy as np
from PIL import Image


def load_image(image_path):
    max_dim = 512
    img = tf.io.read_file(str(image_path))
    img = tf.image.decode_image(img, channels=3)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = tf.expand_dims(img, axis=0)

    return img


def save_image(image_array, file_path, format=None):
    """Expect image_array to have shape of n_height * n_width * n_channel
    """

    # PIL fromarray takes in array of dtype 'uint8'
    unsigned_image_array = tf.squeeze(
        image_array, axis=0).numpy().astype('uint8')
    image = Image.fromarray(unsigned_image_array, 'RGB')
    image.save(file_path, format=format)
    print('Image saved at {file_path}.'.format(file_path=file_path))


def clip_image(image):
    return tf.clip_by_value(image, clip_value_min=0, clip_value_max=255)
