from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm


def make_prediction(args):

    model = load_model(args.model_fn,
        custom_objects={'Melspectrogram':Melspectrogram,
                        'Normalization2D':Normalization2D})

    wav_paths = glob('{}/**'.format('wavfiles'), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    classes = sorted(os.listdir('wavfiles'))
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    results = []

    for z, wav_fn in tqdm(enumerate(wav_paths), total=len(wav_paths)):
        try:
            rate, wav = downsample_mono(src_fn, args.sr)
        except:
            print('Error with downsampling')
            continue
        mask, env = envelope(wav, rate, threshold=args.threshold)
        clean_wav = wav[mask]
        step = int(args.sr*args.dt)
        batch = []

        for i in range(0, clean_wav.shape[0], step):
            sample = clean_wav[i:i+step]
            sample = sample.reshape(1,-1)
            if sample.shape[0] < step:
                tmp = np.zeros(shape=(1,step), dtype=np.int16)
                tmp[:,:sample.shape[1]] = sample.flatten()
                sample = tmp
            batch.append(sample)
        X_batch = np.array(batch)
        y_pred = model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)
        y_pred = np.argmax(y_mean)
        results.append(y_mean)
    
    #np.save(os.path.join('logs', args.pred_fn), np.array(results))
    return np.array(results)
    
