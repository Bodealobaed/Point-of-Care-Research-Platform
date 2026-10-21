import pytest
from biomarker import CATALOG, evaluate_assay

def test_catalog():
    assert 'Parkinson / neuromotor research' in CATALOG

def test_assay_screening():
    result=evaluate_assay('alpha-syn',0.5,1,0.1,1,5,9,95,6,20)
    assert result['checks_met']==6
    assert result['validated_claim_allowed'] is False

def test_bad_input():
    with pytest.raises(ValueError):
        evaluate_assay('x',1,1,1,0,3,10,100,0,10)
