# Neural Style Transfer
Implementation of Neural Style Transfer from the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (_Gatys et al._) in Tensorflow 2.0.

Total variation loss has also been included as a component in the loss function. This was not covered in the paper by Gatys et al. but was inspired by Tensorflow's [implementation of neural style transfer](https://www.tensorflow.org/beta/tutorials/generative/style_transfer#total_variation_loss).

# Examples
These examples are generated using default [options](#options).

Images used can be found in the `data/demo` directory.

## Example 1
<p align='center'>
    <img src='./data/demo/candy.jpg' height='200px' style='padding-right: 4px'/>
    <img src='./data/demo/chicago.jpg' height='200px'/>
    <img src='./data/demo/chicago_candy.png' width='508px'>
<p>

## Example 2
<p align='center'>
    <img src='./data/demo/udnie.jpg' height='200px' style='padding-right: 4px'/>
    <img src='./data/demo/bean.jpg' height='200px'/>
    <img src='./data/demo/bean_udnie.png' width='564px'>
<p>

# Demo
A demo is available on [Google Colab](https://colab.research.google.com/github/kw01sg/neural-style-transfer/blob/master/google-colab-demo.ipynb) in the form of a Colab notebook.

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
* `-c`, `--content-path` : Path of content image. _Default_: `data/demo/chicago.jpg`
* `-s`, `--style-path` : Path of style image. _Default_: `data/demo/candy.jpg`
* `-sw`, `--content-weight` : Content weight. _Default_: `0.4`
* `-cw`, `--style-weight` : Style weight. _Default_: `1.0`
* `-vw`, `--variation-weight` : Variation weight. _Default_: `2e4`
* `-lr`, `--learning-rate` : Learning rate for Adam optimizer. _Default_: `10.0`
* `-e`, `--epochs` : Number of epochs. _Default_: `10`
* `-steps`, `--steps` : Number of steps per epoch. _Default_: `100`
* `-o`, `--output-file` : File name for generated image file. Path can include extension, for example `example.png`. If no extension is given, default extension is `png`. If no file name is provided, generated image will be output as `result.png`. All output files are saved in `data/results` directory. _Default_: `result.png`
