import librosa
import os
import numpy as np

test_audio_dir = 'Respiratory/testAudio'
files = os.listdir(test_audio_dir)
for f in files:
    if f.endswith('.wav'):
        path = os.path.join(test_audio_dir, f)
        try:
            x, sr = librosa.load(path)
            spectrum = librosa.feature.mfcc(y=x, sr=sr)
            size = spectrum.ravel().size
            print(f"{f}: {size}")
        except Exception as e:
            print(f"Error loading {f}: {e}")
