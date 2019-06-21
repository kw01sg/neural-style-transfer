import tensorflow as tf
from pathlib import Path
import time
import argparse

from src.utils import load_image, save_image
from src.model import VGG19Model

DATA_PATH = Path('./data/')
DEMO_DATA_PATH = DATA_PATH / 'demo'
RESULTS_DATA_PATH = DATA_PATH / 'results'

demo_content_path = DEMO_DATA_PATH / 'chicago.jpg' if DEMO_DATA_PATH.is_dir() \
    else tf.keras.utils.get_file('chicago.jpg', 'https://i.imgur.com/tGnrc1a.jpg')

demo_style_path = DEMO_DATA_PATH / 'candy.jpg' if DEMO_DATA_PATH.is_dir() \
    else tf.keras.utils.get_file('candy.jpg', 'https://i.imgur.com/dRhrEEu.jpg')

# Content layer where we will pull our feature maps
CONTENT_LAYERS = ['block5_conv2']

# Style layer we are interested in
STYLE_LAYERS = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

parser = argparse.ArgumentParser(description='Perform neural style transfer.')
parser.add_argument('--content-path', type=str, default=demo_content_path,
                    dest='content_path', help='path of content image')
parser.add_argument('--style-path', type=str, default=demo_style_path,
                    dest='style_path', help='path of style image')
parser.add_argument('--style-weight', type=float, default=1.0,
                    dest='style_weight', help='style weight')
parser.add_argument('--content-weight', type=float, default=0.4,
                    dest='content_weight', help='content weight')
parser.add_argument('--variation-weight', type=float, default=2e4,
                    dest='variation_weight', help='variation weight')
parser.add_argument('-lr', '--learning-rate', type=float, default=10.0,
                    dest='learning_rate', help='learning rate for Adam optimizer')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--steps', type=int, default=100,
                    dest='steps_per_epoch', help='number of steps per epoch')

args = parser.parse_args()

print('Running neural style tranfer with the following parameters:')
print()

for key, value in vars(args).items():
    print('\t{key}: {value}'.format(key=key, value=value))
print()

content_path = args.content_path
style_path = args.style_path

content_image, style_image = [load_image(
    path) for path in (content_path, style_path)]

style_content_model = VGG19Model(CONTENT_LAYERS, STYLE_LAYERS)

image = tf.Variable(content_image)
style_targets = style_content_model(style_image)['style_outputs']
content_targets = style_content_model(content_image)['content_outputs']

style_weight = args.style_weight
content_weight = args.content_weight
variation_weight = args.variation_weight

learning_rate = args.learning_rate
opt = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1)

style_content_model.compile(opt)

start = time.time()

epochs = args.epochs
steps_per_epoch = args.steps_per_epoch

for n in range(epochs*steps_per_epoch):
    style_content_model.fit(image,
                            content_targets=content_targets,
                            style_targets=style_targets,
                            content_layer_weights=[1],
                            style_layer_weights=[
                                1.0/len(STYLE_LAYERS)] * len(STYLE_LAYERS),
                            content_weight=content_weight,
                            style_weight=style_weight,
                            variation_weight=variation_weight)

end = time.time()
print("Total time: {:.1f}s".format(end-start))

if not RESULTS_DATA_PATH.is_dir():
    RESULTS_DATA_PATH.mkdir()

save_image(image, RESULTS_DATA_PATH / 'result.png')
