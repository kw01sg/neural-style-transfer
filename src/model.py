import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

from src.loss import style_content_loss, calculate_variation_loss
from src.utils import clip_image


class VGG19Model(tf.keras.Model):

    def __init__(self, content_layers, style_layers):
        super(VGG19Model, self).__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)
        self.model = self.get_vgg_layers()

    def get_vgg_layers(self):
        vgg = VGG19(weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(
            layer).output for layer in self.content_layers + self.style_layers]

        model = tf.keras.Model(vgg.input, outputs)
        return model

    def call(self, inputs):
        inputs = preprocess_input(inputs)
        outputs = self.model(inputs)

        return {'content_outputs': outputs[:self.num_content_layers],
                'style_outputs': outputs[self.num_content_layers:]}

    def compile(self, optimizer):
        self.optimizer = optimizer

    @tf.function()
    def fit(self, image, content_targets, style_targets, content_layer_weights,
            style_layer_weights, content_weight, style_weight, variation_weight):
        with tf.GradientTape() as tape:
            output = self(image)
            loss = style_content_loss(
                generated_outputs=output,
                content_targets=content_targets,
                style_targets=style_targets,
                content_layer_weights=content_layer_weights,
                style_layer_weights=style_layer_weights,
                alpha=content_weight,
                beta=style_weight)

            variation_loss = calculate_variation_loss(image)
            loss = loss + variation_weight*variation_loss

        gradient = tape.gradient(loss, image)
        self.optimizer.apply_gradients([(gradient, image)])
        image.assign(clip_image(image))
