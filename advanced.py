"""Research-only methods: artifact-aware window comparisons, signal reliability, bootstrap CIs."""
import numpy as np
import pandas as pd
from core import analyze, validate

def experiment(data, window_s=8, threshold=0.45):
    windows, fs = analyze(data, window_s)
    out=windows.copy()
    out['reliable_imu']=out['imu_quality']>=threshold
    out['reliable_ppg']=out['ppg_quality']>=threshold
    out['usable_modalities']=out[['reliable_imu','reliable_ppg']].sum(axis=1)
    out['equal_weight_quality']=out[['imu_quality','ppg_quality']].mean(axis=1)
    out['adaptive_quality']=out['fusion_score']
    out['quality_delta']=out['adaptive_quality']-out['equal_weight_quality']
    out['review_required']=(out['usable_modalities']==0)|(out['status']!='research_features_only')
    return out,fs

def summarize(out):
    valid=out.loc[out['status']=='research_features_only']
    def safe_mean(col):
        v=pd.to_numeric(valid[col],errors='coerce').dropna()
        return float(v.mean()) if len(v) else None
    return {'windows':len(out),'valid_windows':len(valid),'rejected_windows':int(out.review_required.sum()),
            'imu_quality':safe_mean('imu_quality'),'ppg_quality':safe_mean('ppg_quality'),
            'adaptive_quality':safe_mean('adaptive_quality'),'equal_quality':safe_mean('equal_weight_quality'),
            'mean_delta':safe_mean('quality_delta'),'mean_motion_ratio':safe_mean('tremor_band_ratio')}

def bootstrap_delta(out, iterations=400, seed=42):
    vals=out.loc[out.status=='research_features_only','quality_delta'].dropna().to_numpy(float)
    if len(vals)<3:return None
    rng=np.random.default_rng(seed)
    means=np.mean(rng.choice(vals,size=(iterations,len(vals)),replace=True),axis=1)
    return tuple(map(float,np.percentile(means,[2.5,97.5])))

def validate_manual_row(record):
    required=['study_id','sensor','sample_rate_hz','measurement','unit','notes']
    if not str(record.get('study_id','')).strip():raise ValueError('Study ID is required')
    if not str(record.get('sensor','')).strip():raise ValueError('Sensor type is required')
    rate=float(record.get('sample_rate_hz',0))
    if not 0<rate<=100000:raise ValueError('Sampling rate must be positive and reasonable')
    value=float(record.get('measurement'))
    if not np.isfinite(value):raise ValueError('Measurement must be finite')
    return {**{k:str(record.get(k,'')) for k in ['study_id','sensor','unit','notes']},'sample_rate_hz':rate,'measurement':value}
