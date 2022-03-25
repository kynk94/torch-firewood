# Firewood

Implementation of custom torch layers and models.  
All layers are compatible with official pytorch `nn.Module` and support `ddp` strategy.

## [Installation](https://github.com/kynk94/torch-firewood/blob/main/INSTALL.md)

## Models & Trainers

(nn.Module model & pytorch-lightning trainer)

### GAN

* [x] GAN
  [:evergreen_tree:Model](https://github.com/kynk94/torch-firewood/blob/main/firewood/models/gan/GAN.py)
  [:fire:Trainer](https://github.com/kynk94/torch-firewood/blob/main/firewood/trainer/gan/GAN.py)
* [x] DCGAN
  [:evergreen_tree:Model](https://github.com/kynk94/torch-firewood/blob/main/firewood/models/gan/DCGAN.py)
  [:fire:Trainer](https://github.com/kynk94/torch-firewood/blob/main/firewood/trainer/gan/DCGAN.py)
* [x] LSGAN
  [:evergreen_tree:Model](https://github.com/kynk94/torch-firewood/blob/main/firewood/models/gan/LSGAN.py)
  [:fire:Trainer](https://github.com/kynk94/torch-firewood/blob/main/firewood/trainer/gan/LSGAN.py)
* [x] Pix2Pix
  [:evergreen_tree:Model](https://github.com/kynk94/torch-firewood/blob/main/firewood/models/gan/pix2pix.py)
  [:fire:Trainer](https://github.com/kynk94/torch-firewood/blob/main/firewood/trainer/gan/pix2pix.py)
* [x] Pix2PixHD
  [:evergreen_tree:Model](https://github.com/kynk94/torch-firewood/blob/main/firewood/models/gan/pix2pixHD.py)
  [:fire:Trainer](https://github.com/kynk94/torch-firewood/blob/main/firewood/trainer/gan/pix2pixHD.py)
* [ ] ProGAN
* [ ] StyleGAN

### Semantic Segmentation

* [ ] BiSeNetV1
* [ ] BiSeNetV2

<details>
  <summary>
    <b>Layers</b>
  </summary>

### Separable Convolution

* [x] Depthwise - Pointwise Convolution
  * weight shape: `Conv(in, out, K, K)` &#8594; `Conv(in, 1, K, K) X Conv(1, out, 1, 1)`
* [x] Spatialwise Convolution
  * weight shape: `Conv(in, out, K, K)` &#8594; `Conv(in, smaller, K, 1) X Conv(smaller, out, 1, K)`

### Denormalizations

* [x] AdaIN
* [ ] SPADE

### Introduced from ProGAN

* [x] Learning rate Equalizer hooks

### Introduced from StyleGAN

* [x] Weight Gradient Fixable Convolution
  * All options are compatible with tensorflow convolution. (e.g. "same" padding)  
  * Can implement exactly same with tensorflow.
* [x] Fused Activation (biased activation, cuda extension)
* [x] Up Fir Down filter 1D, 2D, 3D (only 2D support cuda extension)
* [x] Weight Demodulation hooks
  * Support Conv and Linear

</details>

## [PyTorchLightning](https://github.com/PyTorchLightning/pytorch-lightning)

### Callbacks

* [x] Latent Interpolator
* [x] Latent Sampler
* [x] Condition Interpolator (Multi-Condition)
* [x] Image to Image Translation Sampler
* [x] Save Last K ModelCheckpoint

### Metrics

* [x] FID
  * selectable resizing method
    * default: antialiased torchvision
    * original: tf1
  * On CPU computable to avoid GPU VRAM overflow error

## Dataset

* [x] NoClassImageFolder for single class model
* [x] PairedImageFolder for I2I model
* [x] ConditionImageFolder for multi-condition(multi-class) model
