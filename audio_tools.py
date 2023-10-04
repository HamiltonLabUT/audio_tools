# Audio processing tools
#
# Liberty Hamilton 2020
# 
# Some code modified from original MATLAB rastamat package.
# 

import numpy as np
from scipy.signal import spectrogram, resample, hilbert, butter, filtfilt, boxcar, convolve
try:
    from scipy.signal.windows import hann as hanning
except:
    from scipy.signal import hanning

from scipy.io import wavfile
from fbtools import fft2melmx
from matplotlib import pyplot as plt
from soundsig import sound


def get_envelope(audio, audio_fs, new_fs, cof=25, bef_aft=[0, 0], pad_next_pow2=False):
    ''' Get the envelope of a sound file
    Inputs:
        w [float] : audio signal vector
        fs [int] : sampling rate of audio signal
        new_fs [int] : desired sampling rate of the envelope (same as your EEG, for example)
    Outputs:
        envelope [array-like] : returns the envelope of the sound as an array
    '''
    
    if pad_next_pow2:
        print("Padding the signal to the nearest power of two...this should speed things up")
        orig_len = len(audio)
        sound_pad = np.hstack((audio, np.zeros((2**int(np.ceil(np.log2(len(audio))))-len(audio),))))
        audio = sound_pad

    print("calculating hilbert transform")
    env_hilb = np.abs(hilbert(audio))

    nyq = audio_fs/2. #Nyquist frequency
    b, a = butter(3, cof/nyq, 'low'); #this designs a 3-pole low-pass filter
    
    print("Low-pass filtering hilbert transform to get audio envelope")
    envelope_long = np.atleast_2d(filtfilt(b, a, env_hilb, axis=0)) #filtfilt makes it non-causal (fwd/backward)

    envelope = resample(envelope_long.T, int(np.floor(envelope_long.shape[1]/(audio_fs/new_fs))))
    if pad_next_pow2:
        print("Removing padding")
        final_len = int((orig_len/audio_fs)*new_fs)
        envelope = envelope[:final_len,:]
        print(envelope.shape)

    if bef_aft[0] < 0:
        print("Adding %.2f seconds of silence before"%bef_aft[0])
        envelope = np.vstack(( np.zeros((int(np.abs(bef_aft[0])*new_fs), 1)), envelope ))
    if bef_aft[1] > 0:
        print("Adding %.2f seconds of silence after"%bef_aft[1])
        envelope = np.vstack(( envelope, np.zeros((int(bef_aft[1]*new_fs), 1)) ))

    envelope[envelope<0] = 0
    envelope = envelope/envelope.max()
    
    return envelope

def get_cse_onset(audio, audio_fs, specgram=None, wins = [0.04], nfilts=80, pos_deriv=True, spec_noise_thresh=1.04):
    """
    Get the onset based on cochlear scaled entropy

    Inputs:
        audio [np.array] : your audio
        audio_fs [float] : audio sampling rate
        wins [list] : list of windows to use in the boxcar convolution
        pos_deriv [bool] : whether to detect onsets only (True) or onsets and offsets (False)

    Outputs:
        cse [np.array] : rectified cochlear scaled entropy over window [wins]
        auddiff [np.array] : instantaneous derivative of spectrogram
    """
    new_fs = 100 # Sampling frequency of spectrogram
    if specgram is None:
        print("Calculating spectrogram")
        specgram = get_mel_spectrogram(audio, audio_fs, nfilts=nfilts)
    
    specgram_thresh = specgram.copy()
    specgram_thresh[specgram_thresh<spec_noise_thresh] = 0
    
    nfilts, ntimes = specgram_thresh.shape

    if pos_deriv is False:
        auddiff= np.sum(np.diff(np.hstack((np.atleast_2d(specgram_thresh[:,0]).T, specgram_thresh)))**2, axis=0)
    else:
        all_diff = np.diff(np.hstack((np.atleast_2d(specgram_thresh[:,0]).T, specgram_thresh)))
        all_diff[all_diff<0] = 0
        auddiff = np.sum(all_diff**2, axis=0)
    cse = np.zeros((len(wins), ntimes))

    # Get the windows over which we are summing as bins, not times
    win_segments = [int(w*new_fs) for w in wins]

    for wi, w in enumerate(win_segments):
        box = np.hstack((np.atleast_2d(boxcar(w)), -np.ones((1, int(0.15*new_fs))))).ravel()
        cse[wi,:] = convolve(auddiff, box, 'full')[:ntimes]

    cse[cse<0] = 0
    cse = cse/cse.max()

    return cse, auddiff

def get_peak_rate(envelope):
    env_diff = np.diff(np.concatenate((0, envelope), axis=None))
    env_diff[env_diff<0] = 0

    return env_diff

def get_mel_spectrogram(w, fs, wintime=0.025, steptime=0.010, nfilts=80, minfreq=0, maxfreq=None):
    ''' Make mel-band spectrogram
    Inputs:
        w [float] : audio signal vector
        fs [int] : sampling rate of audio signal
        wintime [float] : window size
        steptime [float] : step size (time resolution)
        nfilts [int] : number of mel-band filters
        minfreq [int] : Minimum frequency to analyze (in Hz)
        maxfreq [int] : Maximum frequency to analyze (in Hz). If none, defaults to fs/2
    
    Outputs:
        mel_spectrogram [array]: mel-band spectrogram
        freqs [array] : array of floats, bin edges of spectrogram

    '''
    if maxfreq is None:
        maxfreq = int(fs/2)

    pspec, e = powspec(w, sr=fs, wintime=wintime, steptime=steptime, dither=1)
    aspectrum, wts, freqs = audspec(pspec, sr=fs, nfilts=nfilts, fbtype='mel', minfreq=minfreq, maxfreq=maxfreq, sumpower=True, bwidth=1.0)
    mel_spectrogram = aspectrum**0.001

    return mel_spectrogram, freqs

def powspec(x, sr=8000, wintime=0.025, steptime=0.010, dither=1):
    '''
    # compute the powerspectrum and frame energy of the input signal.
    # basically outputs a power spectrogram
    #
    # each column represents a power spectrum for a given frame
    # each row represents a frequency
    #
    # default values:
    # sr = 8000Hz
    # wintime = 25ms (200 samps)
    # steptime = 10ms (80 samps)
    # which means use 256 point fft
    # hamming window
    #
    # $Header: /Users/dpwe/matlab/rastamat/RCS/powspec.m,v 1.3 2012/09/03 14:02:01 dpwe Exp dpwe $

    # for sr = 8000
    #NFFT = 256;
    #NOVERLAP = 120;
    #SAMPRATE = 8000;
    #WINDOW = hamming(200);
    '''

    winpts = int(np.round(wintime*sr))
    steppts = int(np.round(steptime*sr))

    NFFT = 2**(np.ceil(np.log(winpts)/np.log(2)))
    WINDOW = hanning(winpts).T

    # hanning gives much less noisy sidelobes
    NOVERLAP = winpts - steppts
    SAMPRATE = sr

    # Values coming out of rasta treat samples as integers,
    # not range -1..1, hence scale up here to match (approx)
    f,t,Sxx = spectrogram(x*32768, nfft=NFFT, fs=SAMPRATE, nperseg=len(WINDOW), window= WINDOW, noverlap=NOVERLAP)
    y = np.abs(Sxx)**2

    # imagine we had random dither that had a variance of 1 sample
    # step and a white spectrum.  That's like (in expectation, anyway)
    # adding a constant value to every bin (to avoid digital zero)
    if dither:
        y = y + winpts

    # ignoring the hamming window, total power would be = #pts
    # I think this doesn't quite make sense, but it's what rasta/powspec.c does

    # 2012-09-03 Calculate log energy - after windowing, by parseval
    e = np.log(np.sum(y))

    return y, e 

def audspec(pspectrum, sr=16000, nfilts=80, fbtype='mel', minfreq=0, maxfreq=8000, sumpower=True, bwidth=1.0):
    '''
    perform critical band analysis (see PLP)
    takes power spectrogram as input
    '''

    [nfreqs,nframes] = pspectrum.shape

    nfft = int((nfreqs-1)*2)
    freqs = []

    if fbtype == 'mel':
        wts, freqs = fft2melmx(nfft=nfft, sr=sr, nfilts=nfilts, bwidth=bwidth, minfreq=minfreq, maxfreq=maxfreq);
    elif fbtype == 'htkmel':
        wts = fft2melmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq, 1, 1);
    elif fbtype == 'fcmel':
        wts = fft2melmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq, 1, 0);
    elif fbtype == 'bark':
        wts = fft2barkmx(nfft, sr, nfilts, bwidth, minfreq, maxfreq);
    else:
        error(['fbtype ' + fbtype + ' not recognized']);

    wts = wts[:, 0:nfreqs]
    #figure(1)
    #plt.imshow(wts)

    # Integrate FFT bins into Mel bins, in abs or abs^2 domains:
    if sumpower:
        aspectrum = np.dot(wts, pspectrum)
    else:
        aspectrum = np.dot(wts, np.sqrt(pspectrum))**2.

    return aspectrum, wts, freqs

def get_mps(audio, audio_fs, window=0.5):
    '''
    Calculate the modulation power spectrum based on Theunissen lab code from soundsig package
    Inputs:
        audio [array]: sound pressure waveform
        audio_fs [int]: sampling rate of the sound
        window [float] : Time window for the modulation power spectrum
    
    Outputs:
        mps [array] : modulation power spectrum matrix, dimensions are spectral modulation x temporal modulation
        wt [array] : array of temporal modulation values (Hz) to go with temporal dimension axis of mps
        wf [array] : array of spectral modulation values (cyc/kHz) to go with spectral dimension axis of mps

    Example:
    
    import librosa
    s, sample_rate =librosa.load('/Users/liberty/Box/stimulidb/distractors/use_these_ITU_R/stim167_gargling__ITU_R_BS1770-3_12ms_200ms_-29LUFS.mp3')
    mps, wt, wf = get_mps(s, sample_rate)

    '''
    soundobj = sound.BioSound(audio, audio_fs)
    soundobj.mpsCalc(window=window)

    # Return the modulation power spectrum, which is in units of spectral modulation x temporal modulation
    # The actual temporal and spectral modulation values are given by wt and wf

    return soundobj.mps, soundobj.wt, soundobj.wf


if __name__=='main':
    stim_wav = '/Users/liberty/Documents/Austin/code/semantic_EOG/stimuli/blah.wav'
    print("Reading in %s"%stim_wav)
    [stim_fs, w] = wavfile.read(stim_wav)
    
    stim_fs = 44100 # Should be this for everyone, some of them were 48000 but played at 44kHz
    eeg_fs = 128

    print(stim_fs)
    envelope = get_envelope(w[:,0], stim_fs, eeg_fs, pad_next_pow2=True)
