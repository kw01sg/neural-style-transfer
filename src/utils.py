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


def save_image(image_array, file_name):
    """Expect image_array to have shape of n_height * n_width * n_channel
    file_name should be a Path object referencing a file
    """

    # PIL fromarray takes in array of dtype 'uint8'
    unsigned_image_array = tf.squeeze(
        image_array, axis=0).numpy().astype('uint8')
    image = Image.fromarray(unsigned_image_array, 'RGB')

    save_path = file_name
    if file_name.suffix == '':
        save_path = file_name.with_suffix('.png')

    image.save(save_path)
    print('Image saved at {file_name}'.format(
        file_name=save_path.resolve()))


def clip_image(image):
    return tf.clip_by_value(image, clip_value_min=0, clip_value_max=255)


def normalize_weights(weights):
    weights_array = np.array(weights, dtype=np.float32)
    return (weights_array / weights_array.sum()).tolist()


def print_progress(current_step, total_steps, epoch_start_time):
    progress_bar_length = 50
    fill = '='
    padding = len(str(total_steps))

    percent = '{0:.2f}'.format(current_step / total_steps * 100)
    filled_length = progress_bar_length * current_step // total_steps
    bar = fill * filled_length + '>' * (current_step != total_steps) \
        + '.' * (progress_bar_length - filled_length - 1)
    elapsed_time = time() - epoch_start_time

    print(('\r{current_step:' + str(padding) + '}/{total_steps} [{bar}] - {elapsed_time:.0f}s')
          .format(current_step=current_step, total_steps=total_steps, bar=bar, percent=percent, elapsed_time=elapsed_time),
          end='')

    if current_step == total_steps:
        time_per_step = elapsed_time / current_step * 1e3
        print((' - {time_per_step:.0f}ms/step').format(time_per_step=time_per_step))


def get_white_noise_image(shape):
    """Helper function to create white noise image"""
    img = np.random.randint(low=0, high=256, size=shape).astype(np.float32)
    return np.expand_dims(img, axis=0)
