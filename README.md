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
