"""Real RF loading plus synthetic acquisition/evaluation regressions, not accuracy evidence."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from quality_only import QualityOnly
from evaluate_quality_csv import audit_window, run
from collect_quality_csv import parse_packet


def wave(n=1000):
    return pd.DataFrame(dict(sample_index=np.arange(n), timestamp_s=np.arange(n)/100,
                             ppg_red=np.sin(2*np.pi*1.2*np.arange(n)/100)))


def test_frozen_rf_runs_without_torch_or_af():
    code = """
import sys,numpy as np
sys.path.insert(0,'scripts')
from quality_only import QualityOnly
e=QualityOnly('models/quality_af_v2_final')
assert e.threshold == 0.2739494799615767
r=e.predict(np.sin(np.arange(1000)*.1),100)
assert r['status'] in ('good','bad') and 0<=r['quality_score']<=1
assert 'torch' not in sys.modules and 'finalize_two_sources' not in sys.modules
assert e.predict(np.zeros(1000),100)['status']=='invalid_input'
"""
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)


@pytest.mark.parametrize('kind,reason', [
    ('gap','sample_gap_duplicate_or_reset'), ('duplicate','sample_gap_duplicate_or_reset'),
    ('reset','sample_gap_duplicate_or_reset'), ('clock','nonmonotonic_timestamp'),
    ('jitter','irregular_sampling'), ('drift','window_duration_mismatch'),
    ('nan','nonfinite_ppg'), ('short','incomplete_window'),
    ('index_nan','invalid_sample_index'), ('time_nan','invalid_timestamp')])
def test_acquisition_errors_are_not_classified(kind, reason):
    d=wave()
    if kind=='gap': d.loc[500:,'sample_index']+=1
    if kind=='duplicate': d.loc[500,'sample_index']=499
    if kind=='reset': d.loc[500:,'sample_index']-=500
    if kind=='clock': d.loc[500,'timestamp_s']=d.loc[499,'timestamp_s']
    if kind=='jitter': d.loc[500:,'timestamp_s']+=.005
    if kind=='drift': d.timestamp_s*=1.01
    if kind=='nan': d.loc[500,'ppg_red']=np.nan
    if kind=='short': d=d.iloc[:-1]
    if kind=='index_nan': d.loc[500,'sample_index']=np.nan
    if kind=='time_nan': d.loc[500,'timestamp_s']=np.nan
    assert audit_window(d,100,'ppg_red')==reason


def test_boundary_gap_and_valid_clock():
    d=wave(2000)
    assert audit_window(d.iloc[:1000],100,'ppg_red')==''
    d.loc[1000:,'sample_index']+=1
    assert audit_window(d.iloc[1000:],100,'ppg_red',d.iloc[999:1000])=='sample_gap_duplicate_or_reset'


def test_packet_preserves_firmware_sequence():
    r=parse_packet('42,1234567,10,20,30,1,2,3,4,5,6')
    assert r['sample_index']==42 and r['timestamp_s']==1.234567 and r['gz']==6
    for line in ['12345','1,2,3,NaN,5','1.5,100,1,2,3']:
        with pytest.raises(ValueError): parse_packet(line)


def make_input(tmp_path):
    wave(2050).to_csv(tmp_path/'a.csv',index=False)
    pd.DataFrame([dict(filename='a.csv',subject_id='S01',device_id='D01',session_id='A01',
                       activity='rest',channel='ppg_red',fs=100)]).to_csv(tmp_path/'sessions.csv',index=False)


def test_evaluation_separates_invalid_labels_and_reports_groups(tmp_path):
    make_input(tmp_path)
    pd.DataFrame([dict(filename='a.csv',window_index=i,quality=q) for i,q in enumerate([1,0,0])]).to_csv(tmp_path/'labels.csv',index=False)
    r=run(tmp_path/'sessions.csv',tmp_path,ROOT/'models/quality_af_v2_final',tmp_path/'out',tmp_path/'labels.csv')
    assert r['total_windows']==3 and r['valid_labeled_windows']==2 and r['invalid_labeled_windows']==1
    assert r['metrics_valid_labeled_only']['n']==2
    assert r['bad_input_rejection_rate']==.5
    assert r['by_group']['subject_id']['S01']['total_windows']==3
    assert r['evaluation_complete']


def test_no_labels_means_no_accuracy(tmp_path):
    make_input(tmp_path)
    r=run(tmp_path/'sessions.csv',tmp_path,ROOT/'models/quality_af_v2_final',tmp_path/'out')
    assert r['metrics_valid_labeled_only'] is None


def test_duplicate_and_unmatched_labels(tmp_path):
    make_input(tmp_path)
    labels=pd.DataFrame([dict(filename='a.csv',window_index=99,quality=0)])
    labels.to_csv(tmp_path/'labels.csv',index=False)
    r=run(tmp_path/'sessions.csv',tmp_path,ROOT/'models/quality_af_v2_final',tmp_path/'out',tmp_path/'labels.csv')
    assert not r['evaluation_complete'] and r['unmatched_labels']==[['a.csv',99]]
    pd.concat([labels,labels]).to_csv(tmp_path/'labels.csv',index=False)
    with pytest.raises(ValueError,match='Duplicate'):run(tmp_path/'sessions.csv',tmp_path,ROOT/'models/quality_af_v2_final',tmp_path/'out2',tmp_path/'labels.csv')


def test_model_hash_checked_before_load(tmp_path):
    config=json.loads((ROOT/'models/quality_af_v2_final/manifest.json').read_text())
    (tmp_path/'manifest.json').write_text(json.dumps(config))
    (tmp_path/'quality_rf.joblib').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):QualityOnly(tmp_path)


def test_rf_training_shared_split_and_reload(tmp_path):
    from train_quality_models import fit_rf_comparison
    from but_quality_rf_runtime import FEATURES
    from raw_quality_features import EXTRA
    rng=np.random.default_rng(42)
    d=pd.DataFrame(rng.normal(size=(60,len(FEATURES+EXTRA))),columns=FEATURES+EXTRA)
    d['record_id']=[str(i) for i in range(60)]
    d['subject_id']=[str(i//4) for i in range(60)]
    d['quality']=np.tile([0,1],30)
    split={'train':np.arange(40),'validation':np.arange(40,48),'test':np.arange(48,60)}
    reports=fit_rf_comparison(d,split,tmp_path,{'test_fixture':True})
    for name in ['rf_baseline','rf_extended']:
        e=QualityOnly(tmp_path/name)
        assert reports[name]['test']['n']==12
        assert e.predict(wave().ppg_red,100)['status'] in ['good','bad']
    a=pd.read_csv(tmp_path/'rf_baseline/test_predictions.csv')
    b=pd.read_csv(tmp_path/'rf_extended/test_predictions.csv')
    assert a.record_id.equals(b.record_id)


def test_optional_cnn_training_and_reload(tmp_path):
    pytest.importorskip('torch')
    from train_quality_models import fit_cnn
    from quality_cnn_only import QualityCNNOnly
    import torch
    torch.set_num_threads(1)
    rng=np.random.default_rng(1)
    raw=rng.normal(size=(24,300))
    frame=pd.DataFrame(dict(record_id=np.arange(24),subject_id=np.repeat(np.arange(6),4),quality=np.tile([0,1],12)))
    split={'train':np.arange(16),'validation':np.arange(16,20),'test':np.arange(20,24)}
    report=fit_cnn(raw,frame,split,tmp_path,{'test_fixture':True},1,'cpu')
    assert report['test']['n']==4
    e=QualityCNNOnly(tmp_path/'cnn')
    assert e.predict(raw[0],30)['status'] in ['good','bad']
