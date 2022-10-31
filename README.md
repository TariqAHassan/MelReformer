# MelReformer

Resizing inversion of variable length melspectrograms using vanilla Encoder/Decoder Transformer.

Please note that this repository contains ongoing work, and is not a final result (yet).

## Overview

The immediate goal of this repository is to explore using a vanilla Encoder/Decoder Transformer architecture
to reconstruct resized melspectrograms. The broader goal is to realize the following music generation
pipeline:

![](/assets/Music%20Pipeline.png)

The motivation for this pipeline is threefold. First, it allows us to exploit existing
and well-understood image synthesis models. Second, the strong conditioning of the expansion
model should, in principle, enable very good reconstructions, even without the use of exotic attention
mechanisms. Third, an autoregressive approach allows for streaming audio.

Not shown: in this work, recovering audio from the melspectrogram is performed with a neural vocoder.
Specifically, I use [HiFiHybrid](https://github.com/TariqAHassan/HiFiHybrid), which I implemented based on the
latest research a short while ago. (It works quite well with melspectrograms of music.)

#### Idea Model

Here, a model would be used to invent high-level "ideas" for pieces of music.
Critically, this representation can be of a small fixed size, making it compatible
with existing, well-understood image synthesis models, such as StyleGAN2 or UNet-based image diffusion models.
(I have run a few experiments of this kind using StyleGAN2, and it seems to be able to perform this task very well.)

Some modifications would be needed to these existing models, but those changes should be quite small.
For example, if StyleGAN2 was tasked with modeling "images" of the form `[batch, 1, mel_channels, time]`,
and `time` was fixed to 256, then we would also need to obtain the scalar which, when multiplied by 256, 
would give us back the origional number of time steps (which could be used to position a stop "token" in the expansion model, say).
One simple way to do this would be to simultaneously train a layer on top of StyleGAN's style code, which emits 
such a scalar. This scalar would then be fed into the discriminator along with the fake melspectrogram itself. 
(The case with real data is easy because we would know the real melspectrogram's size before and after it was resized.)

#### Expansion Model

In order to invert the resizing used to train the idea model, an encoder/decoder transformer
can be trained. The basic structure here is to condition the transformer on the musical
idea developed by the _Idea Model_, and use that conditioning to guide the (autoregressive) decoder.

There is a wrinkle here, however. Because the attention of the transformer is quadratic,
autoregressively decoding each column of the melspectrogram is problematic. Concretely, in the diagram above,
the expanded melspectrogram is 2048 columns long (representing ~23 sec of audio at 22050 samples/second),
which is well beyond what can be done with a vanilla attention meachism on modest hardware. 
So, drawing inspiration from ViT, I propose "folding"/stacking $n$ adjaent columns on top of one another,
feeding them into the transformer, and then "unfolding"/unstacking the output. 

This folding/stacking trick is quite appealing because, if made to work properly, it provides
a way to model extremely long pieces of music sourced from raw audio. The arithmetic here
is quite simple. If $n=32$, and 23 seconds of audio results in 2048 melspectrogram columns, then
the transformer only needs to model 64 time steps. If we increase the length of the audio by `8x`
we would be able to model 184 seconds (~3 minutes) of audio, while only needing to represent 512 timesteps
in the decoder. Of course, it remains to be seen if this can _actually_ be done, and if so under what
values of $n$.

## Experiments

### Data

I have pooled the [Maestro V3](https://magenta.tensorflow.org/datasets/maestro) and 
[MusicNet](https://zenodo.org/record/5120004#.Y18uZi0r0YI) datasets.

### Training

```shell
python train_sequence.py /path/to/maestro,path/to/musicnet --vocoder_path=path/to/hifivocoder.ckpt
```

Note that the dataloader here relies on caching. That is, the data loader will request
a random chunk of audio from a random file in the dataset. If the request file has not been 
encountered before, it first be decoded and persisted to the cache as a pickled pytorch tensor.
Once this has been done for all files in the dataset, data loading is *very* fast, 
however it comes at the cost of some very slow initial epochs.

### Results (Zooming in can help 😊)

Below you can find some examples of what the expansion model produces after 63 epochs
using $n=32$.

#### Reconstruction at inference time with teacher forcing enabled:

![](/assets/epoch_recon_63.png)

Interpretation: each section (denoted by a white boarder) contains two images: the origional (left)
and the reconstruction (right).

#### Results without teacher forcing (goal):

![](/assets/epoch_condn_63.png)

Note: here, only the reconstruction (i.e., part on the right) is shown. If the left part
was shown, it would match what is found in the figure above (which relies on teacher forcing).

Even without teacher forcing, the model does seem to do a reasonably good job of reconstructing the melspectrogram
(although there are a few clear failures, such as the example in the top right). However, while the overall
structure is often and generally good, the reconstruction is blurry. 

## Future Directions

The blurriness stems directly from the fact that the model is trained to minimize the L1 reconstruction error. 
There are several approaches one could explore to eliminate this and restore the high-frequency information, namely:

  * Adding a discriminator, which has shown success in other domains like vocoders, neural image compression, etc.
  * Training this model as a diffusion model, although this does complicate the problem of variable length inputs.
  * Switching to a pre-trained neural audio compression model and training the model with cross entropy, which
    can be done because the compressed codes are discrete.
       * This may be possibly by attempting to repurpose the Spatial/Depth Transformer paradigm proposed in
         [Autoregressive Image Generation using Residual Quantization](https://arxiv.org/abs/2203.01941). (Instead of 
         a "Depth Transformer" this transformer would be responsible for decoding $n$ adjacent timesteps.)

While each of these approaches is appealing, I am most partial to the last one.
Why? A vocoder can be seen as an autoencoder, with a 2D latent space and fixed encoder (the melspectrogram transform). 
Under this framing, a neural audio compression model like [EnCodec](https://github.com/facebookresearch/encodec) 
can be viewed as a drop-in replacement, with potentially much better reconstruction capabilities because, in part, 
it is not constrained to having a fixed encoder, much less one which does not account for the Heisenberg-Gabor Uncertainty Principle
(see [wikipedia/Short-time Fourier transform](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)).
