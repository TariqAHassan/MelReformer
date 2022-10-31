# MelReformer

Resizing inversion of variable length melspectrograms using vanilla Encoder/Decoder Transformer

## Overview

The immediate goal of this repository is to explore using a vanilla Encoder/Decoder Transformer architecture
to reconstruct resized melspectrograms. The broader goal is to realize the following music generation
pipeline:

[](/assets/Music%20Pipeline.png)

### Components:
  
#### Idea model. 

Here, a model would be used to invent high-level "ideas" for pieces of music.
Critically, this representation can be of a small fixed size, making it compatible
with existing image synthesis models (e.g., StyleGAN or UNet-based image diffusion models).

#### Expansion model

In order to invert the resizing use to train the idea model, an encoder/decoder transformer
can be trained. The basic structure here is to condition the transformer on the musical
idea developed by the idea model, and use that conditioning to guide the (autoregressive) decoder.
This approach has a few advantages. First, the melspectrogram contains rich conditioning information, 
which is far more precise than, say, text. Second, an autoregressive approach allows for streaming
audio, meaning the resulting music could start playing almost instantly, rather than having 
to wait for the entire piece of audio to be rendered.

There is a wrinkle here, however. Because the attention of the transformer is quadratic,
autoregressively decoding each column of the melspectrogram is problematic. Concretely, in the example above,
the expanded melspectrogram is 2048 columns long (representing ~23 sec of audio at 22050 samples/second),
which is well beyond what the vanilla attention meachism can handle.

So, drawing inspiration from ViT, we propose "folding"/stacking $n$ adjaent columns on top of one another,
feeding them into the transformer, and then "unfolding"/unstacking the output. In the result shown below,
$n=32$ was used. 

### Results (so far)

Reconstruction with teach enabled:

[](/assets/epoch_recon_63.png)

Note: the images on the left are the originals, and the image on the right are the corresponding
reconstructions.

Results without teacher forcing (goal):

[](/assets/epoch_condn_63.png)

Note: here the original is the same as it is above. 

As you can see, in both cases, the reconstructed melspectrogram is blurry.
This stems directly from the fact that the model is trained to minimize the L1
reconstruction error. There are several approaches one could explore to eliminate this,
and restore the high-frequency information, namely:

  * Adding a discriminator, which has shown success in other domains like vocoders, neural image compression, etc.
  * Training this model as a diffusion model, although this does complicate the problem of variable length inputs.

