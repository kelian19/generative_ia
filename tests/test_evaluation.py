import numpy as np
import pytest
from src.evaluation import compute_metrics
from src.actuarial import compute_prime_pure,print_actuarial_report

def test_metric_shapes_and_exact_payment():
 a=np.array([[1.],[2.],[3.],[4.]])
 result=compute_metrics(a,a.copy());assert result['wasserstein']==0 and result['ks_stat']==0
 probability,cost=compute_prime_pure(a,2,10000)
 assert probability==.5 and cost==5000

def test_invalid_samples_not_silently_filtered():
 for x in [[],[1,np.nan],[np.inf]]:
  with pytest.raises(ValueError):compute_metrics([1,2],x)
  with pytest.raises(ValueError):compute_prime_pure(x,2,100)

def test_payment_is_not_claimed_as_an_annual_premium(capsys):
 print_actuarial_report(np.arange(4),1,100)
 text=capsys.readouterr().out
 assert 'par observation' in text and 'Aucune prime annuelle' in text

def test_proxy_threshold_must_not_be_inherited_silently():
 with pytest.raises(ValueError,match='Seuil'):compute_prime_pure([1,2],None,100)
