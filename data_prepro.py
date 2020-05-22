try:
    import os
    import pandas as pd
    import shutil
    import soundfile
    from tqdm import tqdm
except ImportError as error:
    if(error.__class__.__name__ == 'ModuleNotFoundError'):
         print(error.__class__.__name__ + ' please install '+ error.name)
    else:
        print(error.__class__.__name__ + ": " + error.message + " : please, install it")
    exit(1)

try:
    os.mkdir('wavfiles/')
except FileExistsError:
    print('wavfiles dir exists')

try:
    df_info = pd.read_csv('UrbanSound8K.csv')
except FileNotFoundError:
    print('Please, install UrbanSound8K.csv from https://urbansounddataset.weebly.com/urbansound8k.html')
    exit(1)

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
        print('{classname} dir exists')
        
for filename, fold, classname in tqdm(zip(df_info['slice_file_name'], 
                                          df_info['fold'], 
                                          df_info['class'])):
    #shutil.copy('rawdata/fold'+str(fold)+'/'+str(filename), 
    #            'wavfiles/'+str(classname)+'/')
    data, sr = soundfile.read('rawdata/fold'+str(fold)+'/'+str(filename))
    soundfile.write('wavfiles/'+str(classname)+'/'+str(filename),
                    data, sr, subtype='PCM_16')

print('Data is ready')
    
