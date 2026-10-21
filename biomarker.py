"""Transparent feasibility triage for research assay concepts; never a diagnostic model."""
import math

CATALOG = {
 'Parkinson / neuromotor research': [
  {'name':'Oligomeric alpha-synuclein','matrix':'Saliva','method':'Immunoassay / aptasensor','specificity':'Investigational; not validated as a stand-alone diagnostic test','challenge':'Assay harmonization, aggregation state, preanalytics'},
  {'name':'Phosphorylated alpha-synuclein (pS129)','matrix':'Blood / plasma','method':'Immunoassay','specificity':'Investigational; blood-cell contamination can confound results','challenge':'Hemolysis and analytical specificity'},
  {'name':'Neurofilament light chain (NfL)','matrix':'Blood / plasma','method':'Sensitive immunoassay','specificity':'Neuronal injury marker, not Parkinson-specific','challenge':'Age effects and clinical context'},
 ],
 'Inflammation research': [
  {'name':'C-reactive protein (CRP)','matrix':'Blood','method':'Immunoassay','specificity':'Non-specific inflammation marker','challenge':'Interpretation depends on clinical context'},
 ],
 'Metabolic monitoring research': [
  {'name':'Glucose','matrix':'Blood / interstitial fluid','method':'Electrochemical enzymatic sensor','specificity':'Established analyte; clinical use requires validated devices','challenge':'Calibration, interferents and sample matrix'},
 ],
}

def evaluate_assay(target, lod, lowest_expected, blank_sd, slope, replicates, cv_pct, recovery_pct, interference_pct, sample_minutes, *, validated=False):
    """Unit-consistent feasibility checks, no clinical thresholds or diagnosis."""
    vals=[lod,lowest_expected,blank_sd,slope,cv_pct,recovery_pct,interference_pct,sample_minutes]
    if any(not math.isfinite(float(x)) for x in vals): raise ValueError('All numeric entries must be finite')
    if min(lod,lowest_expected,blank_sd,slope,sample_minutes)<0 or slope==0 or not 0<=cv_pct<=100 or not 0<=recovery_pct<=200 or not 0<=interference_pct<=100:
        raise ValueError('Check nonnegative inputs, positive slope, and percentage ranges')
    if not 1<=int(replicates)<=100000: raise ValueError('Replicate count out of range')
    # 3.3 sigma / slope is an exploratory approximation only, not an ICH validation.
    estimated_lod=3.3*blank_sd/slope
    gates={
      'Declared LoD below lowest target concentration':lod<=lowest_expected if lowest_expected>0 else False,
      'Estimated blank-noise LoD below lowest target concentration':estimated_lod<=lowest_expected if lowest_expected>0 else False,
      'Repeatability screening (CV <= 15%)':cv_pct<=15 and replicates>=3,
      'Spike recovery screening (80–120%)':80<=recovery_pct<=120,
      'Interference screening (<= 10%)':interference_pct<=10,
      'Measurement time screening (<= 30 min)':sample_minutes<=30,
    }
    met=sum(gates.values()); total=len(gates)
    return {'target':target,'screening_checks':gates,'checks_met':met,'checks_total':total,
            'estimated_lod_approx':estimated_lod,'declared_lod':lod,
            'status':'Research feasibility screening only — not clinically validated',
            'validated_claim_allowed':False,
            'next_steps':[
                'Verify calibration and limits with matrix-matched samples and independent replicates',
                'Assess cross-reactivity, storage, sampling and lot-to-lot variation',
                'Prespecify study protocol, independent controls, reference method and external validation',
                'Do not infer disease presence or absence from these measurements',
            ],'notes':'All concentrations must use the same unit. 3.3σ/slope is an approximate analytical estimate, not a verified LoD. Screening cutoffs are configurable engineering examples, not clinical acceptance standards.'}
