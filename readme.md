# Audio Style Transfer

### Introduction

Style transfer is a concept which is successfully applied to image domain with the example of creating a Van Gogh painting from any given input image. [1] Aim of this project is to adapt the "style transfer" concept to audio domain. Specifically, we aim to transfer the style of an audio (preferably a song) which is labeled as the "style", to another audio which is labeled as the "content", and synthesize a new audio with the general characteristics of the "style" by also remaining loyal to the "content". Through this goal, we can take a step forward for understanding the features of raw music audio signals such as the style, melody, rhythm, and tempo.

Some of the proposed solutions to this problem in the literature include using multiple time-frequency representations [2], short time Fourier transform and Griffin-Lim algorithm [3], and shallow convolutional networks [4]. We aim to implement some of these methods, use the results we will obtain as baselines and try to improve the baseline results by using different features, methods, and models. We want to contribute to this relatively new field of research and come up with interesting results which may bring more attention to the subject.

### Progress

We implement and try two baseline implementations, one from the paper of Mital and the other
from the blog post of Ulyanov.

### Papers

[Neural Style Transfer for Audio Spectograms](https://arxiv.org/abs/1801.01589)

- NIPS 2017 Workshop paper

[Audio style transfer](https://hal.archives-ouvertes.fr/hal-01626389/document)

-  IEEE International
Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2018

[Time Domain Neural Audio Style Transfer](https://arxiv.org/abs/1711.11160) (Baseline implementation: Mital)

- NIPS 2017 Workshop paper

- Code here: https://github.com/pkmital/time-domain-neural-audio-style-transfer

### Blogs

[Audio texture synthesis and style transfer](https://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer/) (Baseline implementation: Ulyanov)

[Neural Style Transfer on Audio Signals](https://alishdipani.github.io/signal_processing/2018/08/29/Neural-Style-Transfer-Audio/)

 - Code here: https://github.com/alishdipani/Neural-Style-Transfer-Audio

### References

[1] A Neural Algorithm of Artistic Style, https://arxiv.org/abs/1508.06576

[2] “Style” Transfer for Musical Audio Using Multiple Time-Frequency Representations, https://openreview.net/forum?id=BybQ7zWCb

[3] Audio texture synthesis and style transfer, https://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer/

[4] Time Domain Neural Audio Style Transfer, https://arxiv.org/abs/1711.11160