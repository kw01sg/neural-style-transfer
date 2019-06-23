import tensorflow as tf
import numpy as np
from PIL import Image
from time import time


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


def print_progress(current_step, total_steps, epoch_start_time):
    progress_bar_length = 50
    fill = '='
    padding = len(str(total_steps))

    percent = '{0:.2f}'.format(current_step / total_steps * 100)
    filled_length = progress_bar_length * current_step // total_steps
    bar = fill * filled_length + '>' \
        + '.' * (progress_bar_length - filled_length - 1)
    elapsed_time = time() - epoch_start_time

    # print(('\r{current_step:' + str(padding) + '}/{total_steps} [{bar}] {percent}% Complete')
    #       .format(current_step=current_step, total_steps=total_steps,
    #               bar=bar, percent=percent), end='')

    print(('\r{current_step:' + str(padding) + '}/{total_steps} [{bar}] - {elapsed_time:.0f}s')
          .format(current_step=current_step, total_steps=total_steps, bar=bar, percent=percent, elapsed_time=elapsed_time),
          end='')

    if current_step == total_steps:
        time_per_step = elapsed_time / current_step * 1e3
        print((', {time_per_step:.1f}ms/step').format(time_per_step=time_per_step))


def test_print_progress():
    from time import sleep
    items = list(range(0, 1000))
    l = len(items)

    start_time = time()
    for i, item in enumerate(items):
        sleep(0.01)
        print_progress(i+1, l, start_time)
