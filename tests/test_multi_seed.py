"""Résumé multi-graines : moyennes, rangs par graine et stabilité validation/test, sur des rapports synthétiques."""
import pytest
from scripts.multi_seed import summarize,stability

def fake(values):
    """values : {modèle: (wasserstein validation, wasserstein test)}."""
    m={k:{sp:{'wasserstein':w,'q99_gen':10.,'trigger_probability':0.05} for sp,w in zip(('validation','test'),v)} for k,v in values.items()}
    return {'models':m}

def test_summary_ranks_and_counts():
    reports={1:fake({'a':(0.1,0.3),'b':(0.2,0.1)}),2:fake({'a':(0.3,0.2),'b':(0.2,0.4)})}
    d,agg=summarize(reports)
    v=agg[agg.split=='validation'].set_index('model')
    assert v.loc['a','w_mean']==pytest.approx(0.2) and v.loc['a','rank_best']==1 and v.loc['b','rank_best']==1
    assert v.loc['a','rank_mean']==pytest.approx(1.5)

def test_stability_detects_reversal():
    st=stability(summarize({1:fake({'a':(0.1,0.3),'b':(0.2,0.1)})})[0])
    assert st[0]['best_validation']=='a' and st[0]['best_test']=='b' and st[0]['spearman_val_test']==pytest.approx(-1)
