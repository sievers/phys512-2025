import numpy as np
from scipy.io import wavfile
#import sounddevice if it exists on the system
try:
    import sounddevice as sd
    have_sd=True
except:
    have_sd=False

play_sounds=True #set to False if you want your computer to remain quiet

#read in the sample sound
samples=np.load('organ_sample.npy')
fs=44100 #set the frequency rate (this value is standard for audio)

#if requested, play the organ sample
if play_sounds:
    if have_sd:
        print('playing organ sample')
        sd.play(samples/samples.max(),fs)

#write out the 
wavname='organ.wav'
wavfile.write(wavname,fs,np.asarray(samples,dtype='short'))
