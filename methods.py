import librosa
from pysndfx import AudioEffectsChain
import numpy as np
import math
import python_speech_features
import scipy as sp
from scipy import signal



'''
FILE READER:
    receives filename,
    returns audio time series (y) and sampling rate of y (sr)
'''
def read_file(file_path):
    #sample_file = file_name
    #sample_directory = 'assets/'
    #sample_path = sample_directory + sample_file

    # generating audio time series and a sampling rate (int)
    y, sr = librosa.load(file_path)

    return y, sr



'''
NOISE REDUCTION USING POWER:
    receives an audio matrix,
    returns the matrix after gain reduction on noise
'''
def reduce_noise_power(y, sr):

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = round(np.median(cent))*1.5
    threshold_l = round(np.median(cent))*0.1

    less_noise = AudioEffectsChain().lowshelf(gain=-30.0, 
                                              frequency=threshold_l, 
                                              slope=0.8).highshelf(gain=-12.0,
                                                                   frequency=threshold_h, 
                                                                   slope=0.5).limiter(gain=6.0)
    y_clean = less_noise(y)

    return y_clean


'''
NOISE REDUCTION USING CENTROID ANALYSIS:
    receives an audio matrix,
    returns the matrix after gain reduction on noise
'''

def reduce_noise_centroid_s(y, sr):

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = np.max(cent)
    threshold_l = np.min(cent)

    less_noise = AudioEffectsChain().lowshelf(gain=-12.0, 
                                              frequency=threshold_l, 
                                              slope=0.5).highshelf(gain=-12.0, 
                                                                   frequency=threshold_h, 
                                                                   slope=0.5).limiter(gain=6.0)
    y_cleaned = less_noise(y)

    return y_cleaned


def reduce_noise_centroid_mb(y, sr):

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = np.max(cent)
    threshold_l = np.min(cent)

    less_noise = AudioEffectsChain().lowshelf(gain=-30.0, 
                                              frequency=threshold_l, 
                                              slope=0.5).highshelf(gain=-30.0, 
                                                                   frequency=threshold_h, 
                                                                   slope=0.5).limiter(gain=10.0)
    y_cleaned = less_noise(y)

    cent_cleaned = librosa.feature.spectral_centroid(y=y_cleaned, sr=sr)
    columns, rows = cent_cleaned.shape
    boost_h = math.floor(rows/3*2)
    boost_l = math.floor(rows/6)
    boost = math.floor(rows/3)

    boost_bass = AudioEffectsChain().lowshelf(gain=16.0, 
                                              frequency=boost_h, 
                                              slope=0.5)
    y_clean_boosted = boost_bass(y_cleaned)

    return y_clean_boosted


'''
NOISE REDUCTION USING MEDIAN:
    receives an audio matrix,
    returns the matrix after gain reduction on noise
'''

def reduce_noise_median(y, sr):
    y = sp.signal.medfilt(y,3)
    return (y)


'''
SILENCE TRIMMER:
    receives an audio matrix,
    returns an audio matrix with less silence and the amout of time that was trimmed
'''
def trim_silence(y):
    y_trimmed, index = librosa.effects.trim(y, top_db=20, frame_length=2, hop_length=500)
    trimmed_length = librosa.get_duration(y) - librosa.get_duration(y_trimmed)

    return y_trimmed, trimmed_length


'''
AUDIO ENHANCER:
    receives an audio matrix,
    returns the same matrix after audio manipulation
'''
def enhance(y):
    apply_audio_effects = AudioEffectsChain().lowshelf(gain=10.0, 
                                                       frequency=260, 
                                                       slope=0.1).reverb(reverberance=25, 
                                                                         hf_damping=5, 
                                                                         room_scale=5, 
                                                                         stereo_depth=50, 
                                                                         pre_delay=20, 
                                                                         wet_gain=0, 
                                                                         wet_only=False)#.normalize()
    y_enhanced = apply_audio_effects(y)

    return y_enhanced


'''
OUTPUT GENERATOR:
    receives a destination path, file name, audio matrix, and sample rate,
    generates a wav file based on input
'''
def output_file(destination ,filename, y, sr, ext=""):
    destination = destination + filename[:-4] + ext + '.wav'
    librosa.output.write_wav(destination, y, sr)
















