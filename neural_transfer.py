from src.model import VGG19Model
from src.utils import load_image, save_image, print_progress, get_white_noise_image
import tensorflow as tf
from pathlib import Path
import time
import argparse

# set logging level as tf.logging is removed in tf 2.0
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


DATA_PATH = Path('./data/')
DEMO_DATA_PATH = DATA_PATH / 'demo'
RESULTS_DATA_PATH = DATA_PATH / 'results'
DEFAULT_OUTPUT_FILE = 'result.png'

demo_content_path = DEMO_DATA_PATH / 'chicago.jpg' if DEMO_DATA_PATH.is_dir() \
    else tf.keras.utils.get_file('chicago.jpg', 'https://i.imgur.com/tGnrc1a.jpg')

demo_style_path = DEMO_DATA_PATH / 'candy.jpg' if DEMO_DATA_PATH.is_dir() \
    else tf.keras.utils.get_file('candy.jpg', 'https://i.imgur.com/dRhrEEu.jpg')

# Content layer where we will pull our feature maps
CONTENT_LAYERS = ['block4_conv2']

# Style layer we are interested in
STYLE_LAYERS = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

parser = argparse.ArgumentParser(description='Perform neural style transfer.')
parser.add_argument('-c', '--content-path', type=str, default=demo_content_path,
                    dest='content_path', help='path of content image')
parser.add_argument('-s', '--style-path', type=str, default=demo_style_path,
                    dest='style_path', help='path of style image')
parser.add_argument('-cw', '--content-weight', type=float, default=1e-3,
                    dest='content_weight', help='content weight')
parser.add_argument('-sw', '--style-weight', type=float, default=1.0,
                    dest='style_weight', help='style weight')
parser.add_argument('-vw', '--variation-weight', type=float, default=2e3,
                    dest='variation_weight', help='variation weight')
parser.add_argument('-slw', '--style-layer-weights', nargs=len(STYLE_LAYERS), type=float,
                    default=[1.0] * len(STYLE_LAYERS), dest='style_layer_weights',
                    help='weights for layers in style image')
parser.add_argument('-wn', '--white-noise-input', action='store_true', dest='white_noise_input',
                    help="""flag to use white noise image as initial image. 
                        If false, content image will be used as initial image.""")
parser.add_argument('-lr', '--learning-rate', type=float, default=10.0,
                    dest='learning_rate', help='learning rate for Adam optimizer')
parser.add_argument('-e', '--epochs', type=int,
                    default=10, help='number of epochs')
parser.add_argument('-steps', '--steps', type=int, default=100,
                    dest='steps_per_epoch', help='number of steps per epoch')
parser.add_argument('-o', '--output-file', type=str, default=DEFAULT_OUTPUT_FILE,
                    dest='output_file', help="""file name for generated image file. 
                        Path can include extension, for example \'example.png\'. 
                        If no extension is given, default extension is \'png\'.
                        If no file name is provided, generated image will be output 
                        as \'result.png\'. All output files are saved in 
                        \'data/results\' directory.""")

args = parser.parse_args()

print('Running neural style transfer with the following parameters:')
print()

for key, value in vars(args).items():
    print('\t{key}: {value}'.format(key=key, value=value))
print()

content_path = args.content_path
style_path = args.style_path

content_image, style_image = [load_image(
    path) for path in (content_path, style_path)]

style_content_model = VGG19Model(CONTENT_LAYERS, STYLE_LAYERS)

image = tf.Variable(get_white_noise_image(tf.shape(content_image)[1:])) \
    if args.white_noise_input else tf.Variable(content_image)
style_targets = style_content_model(style_image)['style_outputs']
content_targets = style_content_model(content_image)['content_outputs']

style_weight = args.style_weight
content_weight = args.content_weight
variation_weight = args.variation_weight

style_layer_weights = args.style_layer_weights

learning_rate = args.learning_rate
opt = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1)

style_content_model.compile(opt)

epochs = args.epochs
steps_per_epoch = args.steps_per_epoch

start_time = time.time()

for epoch in range(epochs):
    epoch_start_time = time.time()
    for step in range(steps_per_epoch):
        style_content_model.fit(image,
                                content_targets=content_targets,
                                style_targets=style_targets,
                                content_layer_weights=[1],
                                style_layer_weights=style_layer_weights,
                                content_weight=content_weight,
                                style_weight=style_weight,
                                variation_weight=variation_weight)

        # printing in the loop as fit statement prints logs that causes epoch 0 statement to be separated from progress bar
        if step == 0:
            print('Epoch {epoch}/{epochs}'.format(epoch=epoch+1, epochs=epochs))

        print_progress(current_step=step+1, total_steps=steps_per_epoch,
                       epoch_start_time=epoch_start_time)

end_time = time.time()
print("Total time: {:.1f}s".format(end_time-start_time))

if not RESULTS_DATA_PATH.is_dir():
    RESULTS_DATA_PATH.mkdir()

output_file = Path(RESULTS_DATA_PATH / args.output_file)
save_image(image, output_file)
