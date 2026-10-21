import io, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
from core import demo_data, validate
from advanced import experiment, summarize, bootstrap_delta, validate_manual_row
from biomarker import CATALOG, evaluate_assay

st.set_page_config(page_title='Point-of-Care Research Platform',page_icon='🧬',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:Inter,Arial,sans-serif}
.stApp{background:radial-gradient(ellipse at 12% 3%,rgba(63,96,206,.22),transparent 32%),radial-gradient(ellipse at 87% 18%,rgba(13,173,165,.15),transparent 28%),linear-gradient(155deg,#081327 0%,#101d35 53%,#0a2031 100%);color:#eef5ff}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#101e34,#0b2331);border-right:1px solid rgba(157,202,255,.15)}
[data-testid="stSidebar"] *{color:#e9f4ff}
.block-container{padding-top:1.8rem;max-width:1500px}
h1,h2,h3{color:#f4f8ff!important;letter-spacing:-.03em}
p,label,.stMarkdown{color:#d9e7f7}
.hero{background:linear-gradient(110deg,rgba(65,93,203,.35),rgba(13,155,153,.22));border:1px solid rgba(171,216,255,.25);border-radius:24px;padding:28px 34px;margin-bottom:18px;box-shadow:0 20px 55px rgba(0,0,0,.16)}
.hero h1{font-size:2.35rem;margin:0 0 8px}
.hero p{font-size:1.02rem;color:#d7ecff;margin:0}
.pill{display:inline-block;border:1px solid #62d4c7;color:#b9fff1;padding:5px 12px;border-radius:30px;font-size:12px;margin-bottom:14px;letter-spacing:.08em}
[data-testid="stMetric"]{background:rgba(22,48,79,.73);padding:16px;border:1px solid rgba(135,190,233,.17);border-radius:16px}
[data-testid="stMetricLabel"]{color:#b7d4ef}
[data-testid="stMetricValue"]{color:#f5fbff}
div[data-testid="stTabs"] button{font-weight:700}
.stButton button[kind="primary"],.stDownloadButton button{background:linear-gradient(100deg,#386ce0,#0a9f99);border:0;color:white;border-radius:12px}
[data-testid="stFileUploader"], [data-testid="stDataFrame"]{border-radius:15px}
small,.muted{color:#abc4d9}
</style>''',unsafe_allow_html=True)
st.markdown('''<div class="hero"><span class="pill">BIOMEDICAL ENGINEERING · RESEARCH PROTOTYPE</span><h1>Point-of-Care Research Platform</h1><p>Quality-aware biomedical signal processing · Adaptive multimodal fusion · Reproducible experimental evaluation</p></div>''',unsafe_allow_html=True)
st.warning('Research-only software. Synthetic data are not patient measurements. No clinical diagnosis, Parkinson prediction, or validated biomarker determination.')

if 'manual_rows' not in st.session_state:st.session_state.manual_rows=[]
with st.sidebar:
    st.header('⚙️ Experiment configuration')
    mode=st.radio('Input source',['Synthetic signals','Upload time-series CSV'],index=0)
    win=st.select_slider('Window length (seconds)',options=[4,6,8,10,12],value=8)
    quality_threshold=st.slider('Reliability threshold',0.1,0.9,0.45,0.05)
    st.divider()
    st.subheader('Synthetic signal parameters')
    duration=st.slider('Recording duration (s)',20,180,80,10)
    sampling=st.select_slider('Sampling rate (Hz)',options=[25,50,100,200],value=50)
    seed=st.number_input('Random seed',min_value=0,max_value=99999,value=17)
    noisy=st.checkbox('Introduce motion artifacts',value=True)
    st.caption('CSV columns: time_s, imu_x, imu_y, imu_z, ppg. Never upload identifiable patient data to a public deployment.')

if mode=='Synthetic signals':
    # Original generator samples at 50 Hz. Resampling the time axis uses interpolation for demonstration only.
    source=demo_data(seconds=duration,fs=sampling,seed=int(seed),noisy=noisy)
    data=source
else:
    upload=st.file_uploader('Upload research CSV (max 15 MB)',type=['csv'])
    if upload is None:
        st.info('Upload a CSV file or switch to Synthetic signals to explore the dashboard.')
        st.stop()
    if upload.size>15_000_000:st.error('Maximum file size is 15 MB');st.stop()
    try:data=pd.read_csv(io.BytesIO(upload.getvalue()))
    except Exception as exc:st.error(f'CSV reading failed: {exc}');st.stop()
try:
    out,fs=experiment(data,win,quality_threshold)
    summary=summarize(out)
except (ValueError,TypeError,KeyError) as exc:
    st.error(f'Input validation failed: {exc}');st.stop()

def fmt(v,unit=''):
    return 'N/A' if v is None or not np.isfinite(v) else f'{v:.3f}{unit}'

t1,t2,t3,t4,t5,t6=st.tabs(['📊 Overview','〰️ Signal laboratory','🧠 Fusion & validation','🧬 Biomarker concept','📝 Manual data entry','📄 Research report'])
with t1:
    st.subheader('Experimental overview')
    a,b,c,d=st.columns(4)
    a.metric('Sampling frequency',f'{fs:.1f} Hz')
    b.metric('Total windows',summary['windows'])
    c.metric('Adaptive quality',fmt(summary['adaptive_quality']))
    d.metric('Review required',summary['rejected_windows'])
    st.markdown('#### Multimodal quality timeline')
    st.line_chart(out.set_index('time_s')[['imu_quality','ppg_quality','adaptive_quality']],height=320)
    st.caption('The displayed fusion score represents signal quality, not disease probability.')
    st.markdown('#### Reproducibility metadata')
    st.json({'mode':mode,'window_s':win,'sampling_rate_hz':round(fs,3),'quality_threshold':quality_threshold,'synthetic_seed':int(seed) if mode=='Synthetic signals' else None})
with t2:
    st.subheader('Biomedical signal laboratory')
    selected=st.multiselect('Select signal channels',['imu_x','imu_y','imu_z','ppg'],default=['imu_x','ppg'])
    if selected:st.line_chart(data.set_index('time_s')[selected],height=370)
    x,y=st.columns(2)
    with x:
        st.markdown('#### Motion-related features')
        st.line_chart(out.set_index('time_s')[['imu_rms','tremor_band_ratio']],height=250)
    with y:
        st.markdown('#### PPG dominant frequency (exploratory)')
        st.line_chart(out.set_index('time_s')[['ppg_peak_hz']],height=250)
    st.info('The 4–8 Hz motion-band ratio is a descriptive spectral feature. It does not establish tremor etiology or a diagnosis.')
with t3:
    st.subheader('Adaptive vs. equal-weight signal quality')
    st.line_chart(out.set_index('time_s')[['adaptive_quality','equal_weight_quality']],height=310)
    a,b,c=st.columns(3)
    a.metric('Equal-weight quality',fmt(summary['equal_quality']))
    b.metric('Adaptive quality',fmt(summary['adaptive_quality']))
    c.metric('Difference',fmt(summary['mean_delta']))
    ci=bootstrap_delta(out)
    if ci:st.caption(f'Exploratory window-resampling 95% bootstrap interval for mean quality difference: [{ci[0]:.3f}, {ci[1]:.3f}]. Overlapping windows/subject dependence are not modeled; this is not a clinical confidence interval.')
    st.markdown('#### Window-level audit')
    st.dataframe(out,use_container_width=True,height=350)
    st.download_button('Download window features CSV',out.to_csv(index=False).encode('utf-8-sig'),'poc_features.csv','text/csv')
    st.info('An increased quality score is not evidence of increased diagnostic accuracy. Comparative studies require independent ground-truth labels and patient-level evaluation.')
with t4:
    st.subheader('Biomarker-guided Point-of-Care research concept')
    st.caption('Generate a testable biosensor research proposal, not a diagnosis or disease-risk prediction. Reference catalog is illustrative and must be verified against current literature.')
    focus=st.selectbox('Research application',list(CATALOG))
    options=CATALOG[focus]
    candidate=st.selectbox('Candidate biomarker',[v['name'] for v in options])
    selected=next(v for v in options if v['name']==candidate)
    a,b=st.columns(2)
    a.info(f"**Specimen:** {selected['matrix']}\n\n**Sensor concept:** {selected['method']}")
    b.warning(f"**Evidence boundary:** {selected['specificity']}\n\n**Main challenge:** {selected['challenge']}")
    st.markdown('#### Analytical feasibility screening')
    st.write('Enter measurements from an experimental assay. Use a consistent concentration unit for LoD and target concentration; blank standard deviation and slope must be expressed in compatible signal units.')
    with st.form('assay_form'):
        a,b,c=st.columns(3)
        lod=a.number_input('Declared assay LoD (concentration unit)',min_value=0.,value=0.5,format='%.5f')
        lowest=b.number_input('Lowest target concentration (same unit)',min_value=0.,value=1.,format='%.5f')
        unit=c.text_input('Concentration unit',value='ng/mL')
        d,e,f=st.columns(3)
        sd=d.number_input('Blank signal standard deviation',min_value=0.,value=0.1,format='%.5f')
        slope=e.number_input('Calibration slope (signal / concentration)',min_value=0.000001,value=1.,format='%.5f')
        reps=f.number_input('Independent assay replicates',min_value=1,max_value=100000,value=5)
        g,h,i,j=st.columns(4)
        cv=g.number_input('Replicate CV (%)',min_value=0.,max_value=100.,value=9.)
        recovery=h.number_input('Spike recovery (%)',min_value=0.,max_value=200.,value=95.)
        interference=i.number_input('Interference effect (%)',min_value=0.,max_value=100.,value=6.)
        minutes=j.number_input('Assay turnaround (minutes)',min_value=0.,value=20.)
        evaluate=st.form_submit_button('Evaluate research feasibility',type='primary')
    if evaluate:
        try:
            result=evaluate_assay(candidate,lod,lowest,sd,slope,int(reps),cv,recovery,interference,minutes)
            result['unit']=unit
            result['research_application']=focus
            st.session_state['assay_result']=result
        except ValueError as exc:st.error(str(exc))
    if 'assay_result' in st.session_state:
        r=st.session_state['assay_result']
        st.metric('Engineering screening checks passed',f"{r['checks_met']} / {r['checks_total']}")
        st.metric('Approximate analytical LoD',f"{r['estimated_lod_approx']:.4g} {r.get('unit','')}")
        st.dataframe(pd.DataFrame([{'Engineering check':k,'Result':'Pass' if v else 'Needs work'} for k,v in r['screening_checks'].items()]),hide_index=True,use_container_width=True)
        st.warning('These are exploratory engineering checks, not clinical sensitivity/specificity, disease detection, or validation. A high score does not mean the biomarker can diagnose a disease.')
        st.markdown('#### Proposed experimental research pathway')
        for item in r['next_steps']:st.write('• '+item)
        st.caption(r['notes'])
        st.download_button('Download assay feasibility record (JSON)',json.dumps(r,ensure_ascii=False,indent=2).encode('utf-8'),'assay_feasibility.json','application/json')

with t5:
    st.subheader('Manual experimental measurements')
    st.write('Enter measurements from a laboratory experiment when automatic extraction is unavailable. These values are stored only in the current session.')
    with st.form('measurement_form',clear_on_submit=True):
        c1,c2,c3=st.columns(3)
        study_id=c1.text_input('Experiment / study ID',placeholder='EXP-001')
        sensor=c2.selectbox('Sensor type',['EIS','ECG','PPG','IMU','Optical','Electrochemical','Other'])
        rate=c3.number_input('Sampling frequency (Hz)',min_value=0.1,max_value=100000.,value=50.,step=1.)
        c4,c5=st.columns(2)
        measurement=c4.number_input('Measured value',value=0.,format='%.5f')
        unit=c5.text_input('Unit',placeholder='Ohm, mV, a.u., ng/mL ...')
        notes=st.text_area('Experimental context / notes',placeholder='Measurement conditions, sample preparation, sensor setup ...')
        submitted=st.form_submit_button('Add measurement',type='primary')
    if submitted:
        try:
            row=validate_manual_row(dict(study_id=study_id,sensor=sensor,sample_rate_hz=rate,measurement=measurement,unit=unit,notes=notes))
            st.session_state.manual_rows.append(row)
            st.success('Measurement added to the current session.')
        except ValueError as exc:st.error(str(exc))
    if st.session_state.manual_rows:
        st.markdown('#### Editable measurement table')
        edited=st.data_editor(pd.DataFrame(st.session_state.manual_rows),num_rows='dynamic',use_container_width=True,key='manual_editor')
        if st.button('Apply edited table'):
            try:st.session_state.manual_rows=[validate_manual_row(r) for r in edited.to_dict('records')];st.success('Edits applied.')
            except (ValueError,TypeError) as exc:st.error(f'Invalid row: {exc}')
        st.download_button('Export manual measurements CSV',pd.DataFrame(st.session_state.manual_rows).to_csv(index=False).encode('utf-8-sig'),'manual_measurements.csv','text/csv')
    st.markdown('#### Import manual table')
    table_up=st.file_uploader('Optional additional table CSV',type=['csv'],key='manual_table_upload')
    if table_up:
        try:
            extra=pd.read_csv(io.BytesIO(table_up.getvalue()))
            st.dataframe(extra.head(100),use_container_width=True)
            st.caption('Preview only; this table is not automatically interpreted as sensor time-series data.')
        except Exception as exc:st.error(f'Table import failed: {exc}')
with t6:
    st.subheader('Research summary and limitations')
    report={'title':'Point-of-Care Research Platform — Exploratory Signal Quality Report','generated_utc':datetime.now(timezone.utc).isoformat(),
        'software_status':'research prototype; not for clinical decision-making','input':{'source':mode,'fs_hz':fs,'window_s':win,'quality_threshold':quality_threshold},
        'results':summary,'bootstrap_mean_delta_95pct':ci if 'ci' in locals() else bootstrap_delta(out),
        'manual_measurements':st.session_state.manual_rows,'biomarker_assay_screening':st.session_state.get('assay_result'),
        'method':'Bandpass filtering, Welch spectral estimates, windowed modality quality and quality-weighted fusion.',
        'limitations':['Synthetic signals do not establish clinical performance','Quality scores are not calibrated disease probabilities','Motion spectral ratios are not Parkinson diagnostic biomarkers','No external clinical validation or ethics-reviewed human study','Manual values are user-entered and not independently verified']}
    st.markdown('#### Study outputs')
    st.json(report)
    st.download_button('Download reproducible report (JSON)',json.dumps(report,indent=2,ensure_ascii=False,default=float).encode('utf-8'),'poc_research_report.json','application/json')
    st.download_button('Download original input data (CSV)',data.to_csv(index=False).encode('utf-8-sig'),'poc_input_data.csv','text/csv')
    st.markdown('#### Proposed scientific evaluation')
    st.markdown('Compare fixed and adaptive fusion under controlled artifact levels, missing channels, and multiple sampling rates. For clinical research, collect consented datasets with reference labels, subject-level train/test separation, external validation, and prespecified statistical metrics.')
