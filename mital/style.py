import numpy as np
import sys
import librosa
from audio_style_transfer.models import timedomain

style = sys.argv[1]
audio, sr = librosa.core.load(style, sr=22050)

content = sys.argv[2]
audio, sr = librosa.core.load(content, sr=22050)

timedomain.run(content,
               style,
               'cnt_' + sys.argv[1][(sys.argv[1].find('/')+1):sys.argv[1].find('.')] + '_style_' + sys.argv[2][(sys.argv[2].find('/')+1):sys.argv[2].find('.')] + '_mital.wav',
               n_fft=4096,
               n_filters=4096,
               hop_length=512,
               alpha=0.05,
               k_w=4)
