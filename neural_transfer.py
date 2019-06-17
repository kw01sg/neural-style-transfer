import tensorflow as tf
from pathlib import Path
import time

from src.utils import load_image, save_image
from src.model import VGG19Model

DATA_PATH = Path('./data/')
# Content layer where we will pull our feature maps
CONTENT_LAYERS = ['block5_conv2']

# Style layer we are interested in
STYLE_LAYERS = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']


content_path = tf.keras.utils.get_file(
    'turtle.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg')
style_path = tf.keras.utils.get_file(
    'kandinsky.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

content_image, style_image = [load_image(
    path) for path in (content_path, style_path)]

style_content_model = VGG19Model(CONTENT_LAYERS, STYLE_LAYERS)

image = tf.Variable(content_image)
style_targets = style_content_model(style_image)['style_outputs']
content_targets = style_content_model(content_image)['content_outputs']

style_weight = 1
content_weight = 0.4
variation_weight = 2e4

opt = tf.keras.optimizers.Adam(learning_rate=10, beta_1=0.99, epsilon=1e-1)
style_content_model.compile(opt)

start = time.time()

epochs = 10
steps_per_epoch = 100

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
print("Total time: {:.1f}".format(end-start))

save_image(image, DATA_PATH / 'test.png')
