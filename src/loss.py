import tensorflow as tf
from src.utils import normalize_weights


def gram_matrix(input_tensor):
    """Expect input_tensor to have shape of n_batch * n_activation_height * n_activation_width * n_channel"""
    return tf.einsum('abcd,abce->ade', input_tensor, input_tensor)


def style_content_loss(generated_outputs,
                       content_targets,
                       style_targets,
                       content_layer_weights,
                       style_layer_weights,
                       alpha,
                       beta):
    """
    Calculates the weighted style and content loss between generated image and
    content and style image

    Args:
      generated_outputs (Dict): Dictionary containing `content_outputs` and `style_outputs` outputs for generated image
      content_targets (List(Tensor)): output of content layers for target content image
      style_targets (List(Tensor)): output of style layers for target style image
      content_layer_weights (List[float]): List of weights of each content output towards content loss
      style_layer_weights (List[float]): List of weights of each style output towards style loss
      alpha (float): Weight of content loss towards total loss
      beta (float): Weight of style loss towards total loss

    Returns:
      loss: weighted style and content loss
    """

    # Calculate content loss
    content_loss = calculate_content_loss(
        content_targets, generated_outputs['content_outputs'], content_layer_weights)

    # calculate style loss
    style_loss = calculate_style_loss(
        style_targets, generated_outputs['style_outputs'], style_layer_weights)

    # calculate total weighted loss
    return alpha*content_loss + beta*style_loss


def calculate_content_loss(original_content, generated_content, content_layer_weights):
    content_loss = 0.5 * tf.reduce_sum([weight * ((original - generated) ** 2) for original,
                                        generated, weight in zip(original_content, generated_content, content_layer_weights)])
    return content_loss


def calculate_style_loss(original_style, generated_style, style_layer_weights):
    normalized_weights = normalize_weights(style_layer_weights)
    gram_original = [gram_matrix(layer) for layer in original_style]
    gram_generated = [gram_matrix(layer) for layer in generated_style]

    style_loss = 0
    for i in range(len(original_style)):
        layer = original_style[i]
        # Layers have shape of n_batch * n_activation_height * n_activation_width * n_channel
        num_channel = layer.shape[-1]
        activation_size = layer.shape[1] * layer.shape[2]
        style_loss = style_loss + (normalized_weights[i] * tf.reduce_sum(
            (gram_generated[i] - gram_original[i]) ** 2) / (4 * num_channel**2 * activation_size**2))

    return style_loss


def calculate_variation_loss(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    return tf.reduce_mean((x_var**2)) + tf.reduce_mean((y_var**2))
