import numpy as np
import methods
import predict
import os
import soundfile
import tkinter as tk
from tkinter import ttk, filedialog
from scipy.optimize import minimize

class Denoiser():
    def __init__(self):
        pass
    
    def denoise(self, file):
        
        y, sr = methods.read_file(file)
        y, time_trimmed = methods.trim_silence(y)
        
        def opt(x):
            soundfile.write('temp/temp.wav', y, sr, subtype='PCM_16')
            
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
            weights0 = predict.make_prediction(args)
            
            y1 = np.copy(y)
            
            for _ in range(x[0]):
                y1 = methods.reduce_noise_power(y1, sr)
            for _ in range(x[1]):
                y1 = methods.reduce_noise_centroid_s(y1, sr)
            for _ in range(x[2]):
                y1 = methods.reduce_noise_centroid_mb(y1, sr)
            for _ in range(x[3]):
                y1 = methods.reduce_noise_median(y1, sr)
            
            soundfile.write('temp/temp.wav', y1, sr, subtype='PCM_16')
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
            
            weights1 = predict.make_prediction(args)
            return max(weights0) - max(weights1)
        
        res = minimize(opt, np.array([0, 0, 0, 0]))
        x = res.x
        
        for _ in range(x[0]):
            y = methods.reduce_noise_power(y, sr)
        for _ in range(x[1]):
            y = methods.reduce_noise_centroid_s(y, sr)
        for _ in range(x[2]):
            y = methods.reduce_noise_centroid_mb(y, sr)
        for _ in range(x[3]):
            y = methods.reduce_noise_median(y, sr)
            
        soundfile.write('temp/temp.wav', y, sr, subtype='PCM_16')
        print('File written')
        
        
        



class Root(tk.Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Audio Denoiser")
        self.minsize(300, 200)
        self.labelFrame = ttk.LabelFrame(self, text="Open a file")
        self.labelFrame.grid(column=0, row=1, padx=20, pady=20)
        self.buttonBrowse()
        self.buttonStart()
        
    def buttonBrowse(self):
        self.buttonBrowse = ttk.Button(
            self.labelFrame,
            text="Browse",
            command=self.fileDialog)
        self.buttonBrowse.grid(column=1, row=1)
        
    def buttonStart(self):
        self.buttonBrowse = ttk.Button(
            self.labelFrame,
            text="Start",
            command=self.enhance)
        self.buttonBrowse.grid(column=2, row=1)
        
    def fileDialog(self):
        self.filename = filedialog.askopenfilename(
            initialdir="/",
            title="Select A File",
            filetype=(("wav", "*.wav"), ("All Files", "*.*")))
        self.label = ttk.Label(self.labelFrame, text="")
        self.label.grid(column=1, row=2)
        self.label.configure(text=self.filename)
         
    def enhance(self):
        self.progressBar = ttk.Progressbar()
        self.progressBar.grid(column=1, row=3)
        self.progressBar.start()
        
        denoiser = Denoiser()
        denoiser.denoise(self.filename)
        
        self.progressBar.stop()


        
        
if __name__ == '__main__':
    #root = Root() 
    #root.mainloop()
    
    filename = 'clean/siren/40722-8-0-0_0.wav'
    print(os.listdir('assets/'))
    
    denoiser = Denoiser()
    denoiser.denoise(filename)
    
    
    
    
    