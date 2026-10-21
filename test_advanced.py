from core import demo_data
from advanced import experiment,summarize,bootstrap_delta,validate_manual_row

def test_experiment():
    out,fs=experiment(demo_data(seconds=32),8)
    assert len(out)==4 and abs(fs-50)<.01
    assert {'adaptive_quality','equal_weight_quality','review_required'}<=set(out.columns)

def test_summary():
    out,_=experiment(demo_data(seconds=32,noisy=True))
    assert summarize(out)['windows']==4
    assert bootstrap_delta(out) is not None

def test_manual():
    row=validate_manual_row(dict(study_id='A',sensor='EIS',sample_rate_hz=50,measurement=1.3,unit='Ohm',notes='test'))
    assert row['measurement']==1.3
