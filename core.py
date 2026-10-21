"""Research-only signal quality and sensor fusion prototype. Not a diagnostic device."""
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

REQUIRED = ('time_s', 'imu_x', 'imu_y', 'imu_z', 'ppg')

def demo_data(seconds=60, fs=50, seed=17, noisy=False):
    rng=np.random.default_rng(seed)
    t=np.arange(0,seconds,1/fs)
    gait=np.sin(2*np.pi*1.65*t)
    tremor=.13*np.sin(2*np.pi*5.1*t)
    a=np.column_stack([.7*gait+tremor, .5*np.cos(2*np.pi*1.65*t), .35*gait])
    a+=rng.normal(0,.07 if not noisy else .5,a.shape)
    ppg=np.sin(2*np.pi*1.15*t)+.2*np.sin(2*np.pi*2.3*t)+rng.normal(0,.06 if not noisy else .6,len(t))
    if noisy:
        a[len(t)//3:len(t)//2]+=rng.normal(0,2,(len(t)//2-len(t)//3,3))
        ppg[len(t)//3:len(t)//2]+=rng.normal(0,2,len(t)//2-len(t)//3)
    return pd.DataFrame(dict(time_s=t,imu_x=a[:,0],imu_y=a[:,1],imu_z=a[:,2],ppg=ppg))

def validate(df):
    missing=set(REQUIRED)-set(df.columns)
    if missing: raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    df=df.loc[:,REQUIRED].copy()
    for col in REQUIRED: df[col]=pd.to_numeric(df[col],errors='coerce')
    if len(df)<200: raise ValueError('At least 200 samples required')
    if df.time_s.isna().any() or not np.isfinite(df.time_s).all(): raise ValueError('Invalid time values')
    dt=np.diff(df.time_s.to_numpy())
    if np.any(dt<=0): raise ValueError('time_s must be strictly increasing')
    fs=1/np.median(dt)
    if not 10<=fs<=500: raise ValueError('Sampling rate must be 10–500 Hz')
    if np.max(np.abs(dt-np.median(dt)))>np.median(dt)*.1: raise ValueError('Irregular sampling: resample before analysis')
    return df,fs

def bandpass(x,fs,low,high):
    sos=butter(3,[low,high],btype='bandpass',fs=fs,output='sos')
    return sosfiltfilt(sos,x)

def spectral_power(x,fs,lo,hi):
    f,p=welch(x,fs=fs,nperseg=min(len(x),int(fs*8)))
    m=(f>=lo)&(f<=hi)
    return float(np.trapezoid(p[m],f[m])) if m.sum()>1 else 0.0

def analyze(df,window_s=8):
    df,fs=validate(df)
    n=max(200,int(window_s*fs)); rows=[]
    for start in range(0,len(df)-n+1,n):
        block=df.iloc[start:start+n]
        arr=block[['imu_x','imu_y','imu_z','ppg']].to_numpy(float)
        missing=1-np.isfinite(arr).mean()
        if missing>.05:
            rows.append(dict(time_s=float(block.time_s.iloc[0]),imu_quality=0.,ppg_quality=0.,fusion_score=np.nan,imu_rms=np.nan,tremor_band_ratio=np.nan,ppg_peak_hz=np.nan,status='insufficient_data'))
            continue
        # Interpolate isolated gaps; quality penalizes missingness.
        arr=pd.DataFrame(arr).interpolate(limit_direction='both').to_numpy()
        imu=np.linalg.norm(arr[:,:3],axis=1)
        imu=imu-np.median(imu)
        ppg=arr[:,3]-np.median(arr[:,3])
        imu_f=bandpass(imu,fs,.5,min(12,fs*.45))
        ppg_f=bandpass(ppg,fs,.5,min(5,fs*.45))
        imu_noise=np.std(imu-imu_f); ppg_noise=np.std(ppg-ppg_f)
        imu_q=float(np.clip(np.std(imu_f)/(np.std(imu_f)+imu_noise+1e-9),0,1))*(1-missing)
        ppg_q=float(np.clip(np.std(ppg_f)/(np.std(ppg_f)+ppg_noise+1e-9),0,1))*(1-missing)
        ratio=spectral_power(imu_f,fs,4,8)/(spectral_power(imu_f,fs,.5,12)+1e-12)
        freqs,px=welch(ppg_f,fs=fs,nperseg=min(len(ppg_f),int(fs*8)))
        band=(freqs>=.7)&(freqs<=3)
        peak=float(freqs[band][np.argmax(px[band])]) if band.any() else np.nan
        # Fusion score is a QUALITY score, not disease probability.
        fusion=(imu_q**2+ppg_q**2)/(imu_q+ppg_q+1e-12)
        rows.append(dict(time_s=float(block.time_s.iloc[0]),imu_quality=imu_q,ppg_quality=ppg_q,fusion_score=float(fusion),imu_rms=float(np.sqrt(np.mean(imu_f**2))),tremor_band_ratio=float(ratio),ppg_peak_hz=peak,status='research_features_only'))
    if not rows: raise ValueError('Recording too short for selected window')
    return pd.DataFrame(rows),fs
