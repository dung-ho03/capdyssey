"""Export and verify the frozen AF-only research release in Colab GPU."""
import argparse,json,shutil,platform
from pathlib import Path
import numpy as np
import scipy,torch
from prepare_af_integrated import digest
from train_af import write_json

def run(base):
    import google.colab
    if not torch.cuda.is_available():raise RuntimeError('Colab GPU required')
    root=base/'models/main_two_sources_v1';candidate=root/'candidate'
    selected=json.loads((candidate/'frozen_selection.json').read_text())
    report=json.loads((candidate/'test_report.json').read_text())
    assert report['selection']==selected and digest(candidate/'model.pt')==selected['checkpoint_sha256']
    out=base/'releases/af-two-source-v1';out.parent.mkdir(exist_ok=True)
    if out.exists():raise RuntimeError('Release already exists; do not overwrite a fixed version')
    stage=base/'releases/af-two-source-v1.staging';stage.mkdir(exist_ok=True)
    runtime=stage/'runtime';runtime.mkdir(exist_ok=True)
    files=['af_model_v1.py','finalize_two_sources.py','af_pulse_experiment.py','train_af_rhythm.py','train_af.py','prepare_af_centered.py','prepare_af_integrated.py']
    scriptdir=Path(__file__).parent
    for name in files:shutil.copyfile(scriptdir/name,runtime/name)
    for name in ['model.pt','frozen_selection.json','test_report.json','reload_check.json']:shutil.copyfile(candidate/name,stage/name)
    shutil.copyfile(scriptdir.parent/'docs/AF_RELEASE_V1.md',stage/'README.md')
    requirements=f'numpy=={np.__version__}\nscipy=={scipy.__version__}\ntorch=={torch.__version__}\n'
    import h5py
    requirements+=f'h5py=={h5py.__version__}\n'
    (stage/'requirements-lock.txt').write_text(requirements)
    config={'version':'af-two-source-v1','threshold':selected['threshold'],'sources':selected['sources'],
            'status':'research_prototype','clinical_validated':False,'wrist_validated':False,'quality_model_included':False,
            'research_goal_met':report['research_goal_met'],'python':platform.python_version(),
            'sha256':{str(p.relative_to(stage)).replace('\\','/'):digest(p) for p in sorted(stage.rglob('*')) if p.is_file() and p.name not in ['release.json','verification.json']}}
    write_json(stage/'release.json',config)
    from af_model_v1 import AFModel
    model=AFModel(stage)
    bundle=base/'processed/provisional_centered_v2';manifest=json.loads((bundle/'manifest.json').read_text())
    with np.load(root/f'seed{selected["seed"]}'/'validation_predictions.npz') as d:vp=d['p'];vy=d['y'];people=d['subject']
    rows=[r for r in manifest['shards'] if r['source'] in selected['sources'] and r['subject'] in people]
    checks={}
    for source in selected['sources']:
        row=next(r for r in rows if r['source']==source and sum(people==r['subject'])==r['n'])
        assert digest(bundle/row['path'])==row['sha256']
        with np.load(bundle/row['path']) as d:x=d['x'][:8];y=d['y'][:8]
        mask=people==row['subject'];assert np.array_equal(vy[mask][:8],y)
        # 50Hz prepared waveform used as API input: checks raw wrapper equivalence.
        scores=np.array([model.predict(w,50)['af_score'] for w in x])
        err=float(np.max(np.abs(scores-vp[mask][:8])));assert err<1e-5
        checks[source]={'n':len(x),'max_score_error':err}
    rejected=0
    for raw,fs in [(np.zeros(500),50),(np.full(500,np.nan),50),(np.ones(499),50),(np.ones((1,500)),50),(np.ones(500),float('nan'))]:
        try:model.predict(raw,fs)
        except ValueError:rejected+=1
    assert rejected==5
    verification={'gpu':torch.cuda.get_device_name(0),'api_equivalence':checks,'bad_input_cases_rejected':rejected,
                  'limits':'Prepared 50Hz validation inputs; not new sensor validation. No additional training or test-driven tuning.'}
    write_json(stage/'verification.json',verification)
    config['sha256']['verification.json']=digest(stage/'verification.json');write_json(stage/'release.json',config)
    stage.rename(out)
    archive=shutil.make_archive(str(out),'zip',root_dir=out.parent,base_dir=out.name)
    print('AF_V1_RELEASE_READY',str(out),flush=True);print(json.dumps(verification),flush=True)
    print('ARCHIVE',archive,'SHA256',digest(Path(archive)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);run(p.parse_args().base)
