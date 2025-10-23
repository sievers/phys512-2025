import numpy as np
from matplotlib import pyplot as plt
import h5py
import glob
plt.ion()

#we'll smooth out our noise model
def smooth_ps(ps,npix):
    x=np.fft.fftfreq(len(ps))*len(ps)
    win=np.exp(-0.5*(x/npix)**2)
    win=win/win.sum()
    winft=np.fft.rfft(win)
    ps_ft=np.fft.rfft(ps)
    return np.fft.irfft(winft*ps_ft,len(ps))

    
def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    tp=template[0]
    tx=template[1]
    return tp,tx
def read_file(filename):
    dataFile=h5py.File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    qmask=dqInfo['DQmask'][...]

    meta=dataFile['meta']
    #gpsStart=meta['GPSstart'].value
    gpsStart=meta['GPSstart'][()]
    #print meta.keys()
    #utc=meta['UTCstart'].value
    utc=meta['UTCstart'][()]
    #duration=meta['Duration'].value
    duration=meta['Duration'][()]
    #strain=dataFile['strain']['Strain'].value
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)

    dataFile.close()
    return strain,dt,utc



fname='H-H1_LOSC_4_V2-1126259446-32.hdf5'
print('reading file ',fname)
hstrain,dt,utc=read_file(fname)

fname='L-L1_LOSC_4_V2-1126259446-32.hdf5'
print('reading file ',fname)
lstrain,dt,utc=read_file(fname)
#dt,utc are the same for both files, so doesn't matter we overwrote


#th,tl=read_template('GW150914_4_template.hdf5')
template_name='GW150914_4_template.hdf5'
tp,tx=read_template(template_name)

hstrain_ft=np.fft.rfft(hstrain)
plt.clf()
plt.loglog(np.abs(hstrain_ft))
plt.show()

hstrain=hstrain-np.median(hstrain)
x=np.linspace(-np.pi,np.pi,len(hstrain))
win=0.5+0.5*np.cos(x)
plt.clf();plt.plot(hstrain);plt.plot(hstrain*win);plt.show()
hft_win=np.fft.rfft(hstrain*win)
plt.clf()
plt.loglog(np.abs(hstrain_ft))
plt.loglog(np.abs(hft_win))
plt.show()

ps=smooth_ps(np.abs(hft_win)**2,10)
plt.loglog(np.sqrt(ps))

tft=np.fft.rfft(tp)
mf=np.fft.irfft(hft_win*np.conj(tft)/ps)
plt.clf()
plt.plot(np.fft.fftshift(mf))
plt.show()

template_filt=np.fft.irfft(tft/ps)
denom=np.sum(template_filt*tp)

noise1=np.median(np.abs(mf[:10000]))/0.67 #factor to convert from median to RMS
noise2=np.median(np.abs(mf[-10000:]))/0.67
print('noises are ',noise1,noise2)
print('nsigma is ',np.max(np.abs(mf))/noise2)

assert(1==0)
plt.clf()
plt.plot(np.fft.fftshift(mf**2*denom))
