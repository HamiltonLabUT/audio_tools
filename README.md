# audio_tools
Tools for creating envelopes, spectrograms, pitch contours from audio.

Before you start, you may need to pip install:
* soundsig
* librosa

## Envelope:

```python
import librosa
from audio_tools import get_envelope

# Get the audio waveform and sampling frequency, preserving the original sampling
# rate of the file
audio_waveform, aud_fs = librosa.load('myaudio.wav', sr=None)

print("Getting envelope")
fs = 100 # Desired sampling rate for the envelope
env = get_envelope(audio_waveform, aud_fs, fs, cof=25, bef_aft=[0, 0], pad_next_pow2=False)
```

## Spectrogram:

```python
import librosa
from audio_tools import get_mel_spectrogram

# Get the audio waveform and sampling frequency, preserving the original sampling
# rate of the file
audio_waveform, aud_fs = librosa.load('myaudio.wav', sr=None)

print("Getting mel spectrogram")
fs = 100 # Desired sampling rate for the spectrogram
nfilts = 80 # How many mel bands to return
mel_spectrogram, freqs = get_mel_spectrogram(audio_waveform, aud_fs, steptime=1/fs, nfilts=nfilts) 
```

## Phoneme feature matrix:
To convert a binary phoneme matrix to a feature matrix, we use the code below:

```python
from phn_tools import convert_phn


phns= ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 
		   'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 
		   'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pcl', 'q', 'r', 's', 'sh', 
		   't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
binary_phn_mat = ... # this will be a binary matrix of time x phonemes. Phonemes are assumed to be in the order above
binary_feat_mat, fkeys = convert_phn(binary_phn_mat, 'features')
```
