"""Package the frozen candidate and replay a small fixed contract check on Colab GPU."""
import argparse,json,platform,shutil,subprocess,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch,sklearn,scipy,joblib
from scipy.io import loadmat
from quality_af_v2 import QualityAF,digest
from train_af import load_records


def replay_samples(predictions):
    """Keep the original within-record window ordinal before sampling rows."""
    predictions=predictions.copy()
    predictions['replay_window_index']=predictions.groupby(['source','record'],sort=False).cumcount()
    return predictions.groupby(['source','y','accepted'],sort=True,group_keys=False).head(2)


def run(base):
    import google.colab
    assert torch.cuda.is_available(),'통합 추론 확인에는 Colab GPU가 필요합니다. CPU로 대체하지 않습니다.'
    base=Path(base)
    experiment=base/'audits/rf_raw_artifact_v1/0ea3e8d56aa7'
    tested=experiment/'frozen_candidate_development_test_v1'
    readj=lambda p:json.loads(p.read_text(encoding='utf-8'))
    protocol=readj(tested/'frozen_protocol.json');complete=readj(tested/'complete.json')
    assert complete['protocol']==protocol
    assert digest(tested/'comparison.csv')==complete['comparison_sha256']
    assert digest(tested/'predictions.csv')==complete['predictions_sha256']
    af=base/'models/main_two_sources_v1/candidate/model.pt'
    rf=experiment/'experimental_rf.joblib'
    assert digest(af)==protocol['af_sha256'] and digest(rf)==protocol['candidate_rf_sha256']
    scripts=Path(__file__).resolve().parent
    code={p.name:digest(p) for p in sorted(scripts.glob('*.py'))}
    import hashlib
    identity=hashlib.sha256(json.dumps(dict(protocol=protocol,code=code),sort_keys=True).encode()).hexdigest()
    out=base/'releases'/('quality-af-v2-'+identity[:12])
    out.mkdir(parents=True,exist_ok=True);(out/'scripts').mkdir(exist_ok=True)
    def copy(source,dest):
        if dest.exists():assert digest(dest)==digest(source),'Existing bundle changed'
        else:shutil.copy2(source,dest)
    copy(af,out/'af_model.pt');copy(rf,out/'quality_rf.joblib')
    for name in code:copy(scripts/name,out/'scripts'/name)
    for source,name in [(tested/'comparison.csv','development_test.csv'),
                        (experiment/'matched_quality_comparison/af_validation.csv','development_validation.csv'),
                        (tested/'frozen_protocol.json','evaluation_protocol.json')]:copy(source,out/name)
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=scripts.parent,text=True).strip()
    except (OSError,subprocess.CalledProcessError):commit=None
    cfg=dict(format='capdyssey-quality-af-v2',release_status='research_prototype',
        quality_threshold=protocol['rf_threshold'],af_threshold=protocol['af_threshold'],
        input=dict(seconds=10,channel='one raw PPG channel',sampling_hz='actual rate supplied by caller',
                   validated_source_rates=[100,125],sensor_transfer_validated=False),
        output='AF/non_AF only after quality pass; remeasure has no AF label or AF score',
        limitations=['Reused development test; not independent or clinical validation.',
          'No wrist/longterm training in AF model. No demonstrated generalization to the team sensor.',
          'Quality labels are BUT HR quality, not independent AF diagnostic quality.',
          'MIMIC accepted specificity 79.84%; both-source 90% sensitivity/80% specificity target not met.',
          'Six-rhythm false alarms fall only 1164 to 1163 relative to no quality filter.',
          'Scores are model outputs, not verified clinical probabilities.'],
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            sklearn=sklearn.__version__,torch=torch.__version__,joblib=joblib.__version__),
        git_commit=commit,sha256={str(p.relative_to(out)).replace('\\','/'):digest(p)
            for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['manifest.json','reload_check.json']})
    manifest=out/'manifest.json'
    if manifest.exists():assert readj(manifest)==cfg,'Existing release manifest differs'
    else:manifest.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    # Import only the copied code for the release replay; fail if copied feature implementations differ.
    for name in ['quality_af_v2.py','raw_quality_features.py','but_quality_rf_runtime.py']:
        assert digest(out/'scripts'/name)==digest(scripts/name)
    engine=QualityAF(out)
    predictions=pd.read_csv(tested/'predictions.csv')
    assert (predictions['split']=='test').all()
    predictions['accepted']=predictions.q_candidate>=cfg['quality_threshold']
    # Fixed first 2 per source/label/gate: a replay consistency check, not another performance estimate.
    sample=replay_samples(predictions)
    bundle=base/'processed/provisional_centered_v2'
    rows={r['record']:r for r in readj(bundle/'manifest.json')['shards'] if r['source'] in ['mimic_perform_af','six_rhythms']}
    parents={r['path']:r for r in readj(base/'processed/integrated_10s_resume_v1/manifest.json')['shards']}
    mimic={p:v for p,_,v in load_records(base/'raw/mimic_af')}
    checks=[]
    for r in sample.itertuples(index=False):
        row=rows[r.record]
        if r.source=='mimic_perform_af':
            fs=125;start=int(round(r.start_seconds*fs));raw=mimic[r.record][start:start+1250]
        else:
            fs=100
            # Existing CSV order matches shard order. Six-rhythm windows have no continuous timestamps.
            source_csv=base/'models/quality_af_fixed_v1'/(row['path']+'.csv')
            assert digest(source_csv)==source_csv.with_suffix('.sha256').read_text().strip()
            existing=pd.read_csv(source_csv)
            # Explicit ordinal survives itertuples(index=False) and stratified row sampling.
            loc=int(r.replay_window_index)
            assert len(existing)==row['n']
            assert 0<=loc<len(existing) and int(existing.iloc[loc].y)==int(r.y)
            np.testing.assert_allclose(existing.iloc[loc].p_af,r.p_af,atol=1e-7,rtol=1e-7)
            indices=np.asarray(parents[row['path']]['provenance']['original_segment_indices'])[row['provenance']['original_window_indices']]
            raw=loadmat(base/'raw/six_rhythms'/(r.record+'.mat'),variable_names=['ppgseg'])['ppgseg'][indices[loc]]
        actual=engine.predict(raw,fs)
        np.testing.assert_allclose(actual['quality_score'],r.q_candidate,atol=1e-7,rtol=1e-7)
        assert (actual['status']=='classified')==r.accepted
        if r.accepted:
            np.testing.assert_allclose(actual['af_score'],r.p_af,atol=1e-5,rtol=1e-5)
            assert actual['label']==('AF' if r.p_af>=cfg['af_threshold'] else 'non_AF')
        else:assert actual['af_score'] is None and actual['label'] is None
        checks.append(dict(source=r.source,record=r.record,status=actual['status']))
    assert engine.predict(np.zeros(1250),125)['status']=='remeasure'
    (out/'reload_check.json').write_text(json.dumps(dict(checked=len(checks),passed=True,
        scope='Saved score replay only; not new accuracy validation',checks=checks),indent=2),encoding='utf-8')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,str(p.relative_to(out)))
    print('QUALITY_AF_V2_READY',out)
    print('저장 모델 재현 확인:',len(checks),'구간 / 임계값:',cfg['quality_threshold'],cfg['af_threshold'])
    print('다운로드 ZIP:',archive)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();run(a.base)
