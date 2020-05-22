import os
import pandas as pd
import soundfile
from tqdm import tqdm

import clean
import train 
import predict
import argparse



class Preprocessor():
    """Initial data preprocessing"""
    def __init__(self):
        pass
    
    def prepare_data(self):
        try:
            os.mkdir('wavfiles/')
        except FileExistsError:
            print('wavfiles dir exists')       
        df_info = pd.read_csv('UrbanSound8K.csv')       
        classes = [
            'air_conditioner',
            'car_horn',
            'children_playing',
            'dog_bark',
            'drilling',
            'engine_idling',
            'gun_shot',
            'jackhammer',
            'siren',
            'street_music']
        for classname in classes:
            try:
                os.mkdir('wavfiles/'+str(classname)+'/')
            except FileExistsError:
                print(f'{classname} dir exists')       
        for filename, fold, classname in tqdm(zip(df_info['slice_file_name'], 
                                                  df_info['fold'], 
                                                  df_info['class'])):
            data, sr = soundfile.read('rawdata/fold'+str(fold)+'/'+str(filename))
            soundfile.write('wavfiles/'+str(classname)+'/'+str(filename),
                            data, sr, subtype='PCM_16')
        print('[CORE]:', 'Data is ready')


class Cleaner():
    """Cleaning data for further training"""
    def __init__(self):
        pass
    
    def clean(self):
        parser = argparse.ArgumentParser(description='Cleaning audio data')
        parser.add_argument('--src_root', type=str, default='wavfiles',
                            help='directory of audio files in total duration')
        parser.add_argument('--dst_root', type=str, default='clean',
                            help='directory to put audio files split by delta_time')
        parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                            help='time in seconds to sample audio')
        parser.add_argument('--sr', type=int, default=16000,
                            help='rate to downsample audio')
    
        parser.add_argument('--fn', type=str, default='100032-3-0-0',
                            help='file to plot over time to check magnitude')
        parser.add_argument('--threshold', type=str, default=20,
                            help='threshold magnitude for np.int16 dtype')
        args, _ = parser.parse_known_args()
        ''' Uncomment to check the result of threshold '''
        '''
        try:
            clean.test_threshold(args)
        except:
            print('[CLEANER]:', 'Input signal too small')
        '''
        clean.split_wavs(args)
        print('[CORE]:', 'Data is cleaned')


class Trainer():
    """Training ML models"""
    def __init__(self):
        pass
    
    def train(self, model='lstm'):
        parser = argparse.ArgumentParser(description='Audio Classification Training')
        parser.add_argument('--model_type', type=str, default=model,
                            help='model to run. i.e. conv1d, conv2d, lstm')
        parser.add_argument('--src_root', type=str, default='clean',
                            help='directory of audio files in total duration')
        parser.add_argument('--batch_size', type=int, default=16,
                            help='batch size')
        parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                            help='time in seconds to sample audio')
        parser.add_argument('--sample_rate', '-sr', type=int, default=16000,
                            help='sample rate of clean audio')
        args, _ = parser.parse_known_args()
        train.train(args)
        print('[CORE]:', model, 'model is trained')



class Predictor():
    """Making predictions on audio files"""
    def __init__(self):
        pass
    
    def predict(self):
        parser = argparse.ArgumentParser(description='Audio Classification Training')
        parser.add_argument('--model_fn', type=str, default='models/lstm.h5',
                            help='model file to make predictions')
        parser.add_argument('--pred_fn', type=str, default='y_pred',
                            help='fn to write predictions in logs dir')
        parser.add_argument('--dt', type=float, default=1.0,
                            help='time in seconds to sample audio')
        parser.add_argument('--sr', type=int, default=16000,
                            help='sample rate of clean audio')
        parser.add_argument('--threshold', type=str, default=20,
                            help='threshold magnitude for np.int16 dtype')
        args, _ = parser.parse_known_args()
        predict.make_prediction(args)








if __name__ == '__main__':
    print('Program started')
    #prepro = Preprocessor()
    #prepro.prepare_data()
    #cleaner = Cleaner()
    #cleaner.clean()
    
    trainer = Trainer()
    trainer.train('conv2d')
    
    


    
    
    