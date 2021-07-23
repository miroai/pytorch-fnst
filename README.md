# fast-neural-style :city_sunrise: :rocket:
This repository contains a pytorch implementation of an algorithm for artistic style transfer. The algorithm and all the code cames from [PyTorch/examples](https://github.com/pytorch/examples/tree/master/fast_neural_style). Slight modification was made in the training script to output more meaningful model names and a streamlit app was added for stylizing images.

For an detailed explanation of how the model works, [this repo](https://github.com/gordicaleksa/pytorch-neural-style-transfer) is particularly helpful.

## Requirements
The program is written in Python, and uses [pytorch](http://pytorch.org/). Install `torch` and `torchvision` following [the lastest official instructions](https://pytorch.org/get-started/locally/). [Streamlit](https://streamlit.io/) is used for the demo app. 

All the requirements and can be installed by `pip install -r requirements.txt`

A GPU is not necessary, but can provide a significant speed up especially for training a new model. Regular sized images can be styled on a laptop or desktop (CPU only) using saved models. 

## Usage
### Stylize image

use the streamlit app by running `streamlit run app_streamlit_evaluate.py` or 

python neural_style/neural_style.py eval --content-image </path/to/content/image> --model </path/to/saved/model> --output-image </path/to/output/image> --cuda 0
```
* `--content-image`: path to content image you want to stylize.
* `--model`: saved model to be used for stylizing the image (eg: `mosaic.pth`)
* `--output-image`: path for saving the output image.
* `--content-scale`: factor for scaling down the content image if memory is an issue (eg: value of 2 will halve the height and width of content-image)
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.

### Train model
```bash
python neural_style/neural_style.py train --dataset </path/to/train-dataset> --style-image </path/to/style/image> --save-model-dir </path/to/save-model/folder> --epochs 2 --cuda 1
```

There are several command line arguments, the important ones are listed below
* `--dataset`: path to training dataset, the path should point to a folder containing another folder with all the training images. An example of this is the mini-imagenet dataset [4GB] which you can download [here](https://www.kaggle.com/ifigotin/imagenetmini-1000).
* `--style-image`: path to style-image.
* `--save-model-dir`: path to folder where trained model will be saved.
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.
* `--batch-size`: defaults to 4 but this depends on how much CUDA memory you have available and the size of the style image. Larger style image requires more CUDA memory. In general, if your style image is no longer than 1000 px on the long edge, you should be able to train comfortablly with batch size 4 on 16GB of CUDA memory
* `--style-weight`: when keeping the `content-weight` constant (1e5), a higher style weight will minimize the style's feature map's gram loss more, therefore, making the input image more and more like the style image. It's best adjusted by power of 10 (I personally start with 1e10, 5e10, 1e11)

## Models

Models for the examples shown below can be downloaded from [here](https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=0) or by running the script ``download_saved_models.py``.
