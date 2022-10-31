# MelReformer

Resizing inversion of variable length melspectrograms using vanilla Encoder/Decoder Transformer.

## Overview

The immediate goal of this repository is to explore using a vanilla Encoder/Decoder Transformer architecture
to reconstruct resized melspectrograms. The broader goal is to realize the following music generation
pipeline:

![](/assets/Music%20Pipeline.png)

The motivation for this pipeline is threefold. First, it allows us to exploit existing
and well-understood image synthesis models.<sup>*</sup> Second, the strong conditioning of the expansion
model should, in principle, enable very good reconstructions, without the use of exotic attention
mechanisms. Third, an aggressive approach allows for streaming audio.

\* I have performed some initial experiments using StyleGAN2 for this purpose, and it seems to perform quite well, 
and I assume the same would be true of more recent UNet-based diffusion models.

Also note that, in this work, recovering audio from the melspectrogram here is performed with a neural vocoder.
Specifically, I use [HiFiHybrid](https://github.com/TariqAHassan/HiFiHybrid), which I implemented based on the
latest research a short while ago, and it performs quite well with music.

#### Idea model. 

Here, a model would be used to invent high-level "ideas" for pieces of music.
Critically, this representation can be of a small fixed size, making it compatible
with existing, well-understood image synthesis models (e.g., StyleGAN or UNet-based image diffusion models).

Some modifications would be needed to these existing models, but those changes should be quite small.
For example, if StyleGAN2 was tasked with modeling "images" of the form `[batch, 1, mel_channels, time]`,
and `time` was fixed to 256, then we would also need to obtain the scalar which would give us back
the origional number of time steps (which could be used position a stop "token" in the expansion model, say).
One simple way to do this would be to simultaneously train a layer on top of StyleGAN's style code, which emits 
such a scalar. This scalar would then be fed into the discriminator along with the melspectrogram itself. 
(The case with real data is easy because we would know the real melspectrogram's size before and after it was resized.)

#### Expansion model

In order to invert the resizing use to train the idea model, an encoder/decoder transformer
can be trained. The basic structure here is to condition the transformer on the musical
idea developed by the _Idea Model_, and use that conditioning to guide the (autoregressive) decoder.
This approach has a few advantages. First, the melspectrogram contains rich conditioning information, 
which is far more precise than, say, text. Second, as stated above, an autoregressive approach allows for streaming
audio, meaning the resulting music could start playing almost instantly, rather than having 
to wait for the entire piece of audio to be rendered.

There is a wrinkle here, however. Because the attention of the transformer is quadratic,
autoregressively decoding each column of the melspectrogram is problematic. Concretely, in the example above,
the expanded melspectrogram is 2048 columns long (representing ~23 sec of audio at 22050 samples/second),
which is well beyond what the vanilla attention meachism can handle. So, drawing inspiration from ViT, I 
propose "folding"/stacking $n$ adjaent columns on top of one another, feeding them into the transformer, and then 
"unfolding"/unstacking the output. In the result shown below, $n=32$ was used. 

### Results so far (Zooming in can help 😊)

Reconstruction at inference time with teacher forcing enabled:

![](/assets/epoch_recon_63.png)

Note: the images on the left are the originals, and the image on the right are the corresponding
reconstructions.

Results without teacher forcing (goal):

![](/assets/epoch_condn_63.png)

Note: here, only the reconstruction is shown (i.e., part on the right). If the original was
shown too, it would be the same as the corresponding "block" in the teacher forcing example above.

As you can see, in both cases, the model does seem to capture the desired structure in the reconstructed melspectrogram,
but the result is blurry. This stems directly from the fact that the model is trained to minimize the L1
reconstruction error. There are several approaches one could explore to eliminate this and restore the high-frequency 
information, namely:

  * Adding a discriminator, which has shown success in other domains like vocoders, neural image compression, etc.
  * Training this model as a diffusion model, although this does complicate the problem of variable length inputs.
  * Preprocessing the input a pre-trained RQ-VAE and training the model with cross entropy. This may be possibly by 
    attempting to repurpose the Spatial/Depth Transformer paradigm proposed in 
    [Autoregressive Image Generation using Residual Quantization](https://arxiv.org/abs/2203.01941). (Instead of a 
   "Depth Transformer" this transformer would be responsible for decoding $n$ adjacent melspectogram columns.)
