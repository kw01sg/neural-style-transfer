# Neural Style Transfer
Implementation of Neural Style Transfer from the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (_Gatys et al._) in Tensorflow 2.0.

Total variation loss has also been included as a component in the loss function. This was not covered in the paper by Gatys et al. but was inspired by Tensorflow's [implementation of neural style transfer](https://www.tensorflow.org/beta/tutorials/generative/style_transfer#total_variation_loss).

# Demo
A demo is available on [Google Colab](https://colab.research.google.com/github/kw01sg/neural-style-transfer/blob/google-colab/google-colab-demo.ipynb) in the form of a Colab notebook.

Code for the Colab notebook can be found in `google-colab-demo.ipynb`, which also includes a link to the notebook itself.

__Make sure to install Tensorflow 2.0 by running__
```
!pip install tensorflow-gpu==2.0.0-beta1
```
__and that GPU is enabled for the notebook.__

# Usage
## Requirements
* `Pillow==6.0.0`
* `tensorflow-gpu==2.0.0-beta1`

Required packages can be installed by running `pip install` on `requirements.txt`
```
pip install -r requirements.txt
```

To use the CPU version, replace `tensorflow-gpu==2.0.0-beta1` with `tensorflow==2.0.0-beta1` in `requirements.txt`. However, running on CPU is __very slow and is generally advised against__.

## Running
```
python neural_transfer.py --content-path <path of content image> --style-path <path of style image>
```

### Options
* `-h`, `--help` : Display help message
* `--content-path` : Path of content image. _Default_: `data/demo/chicago.jpg`
* `--style-path` : Path of style image. _Default_: `data/demo/candy.jpg`
* `--content-weight` : Content weight. _Default_: `0.4`
* `--style-weight` : Style weight. _Default_: `1.0`
* `--variation-weight` : Variation weight. _Default_: `2e4`
* `-lr`, `--learning-rate` : Learning rate for Adam optimizer. _Default_: `10.0`
* `--epochs` : Number of epochs. _Default_: `10`
* `--steps` : Number of steps per epoch. _Default_: `100`
