"""Reproduce team notebook models in Colab when trained artifacts are absent."""
import argparse,json,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from download_data import entries,download,digest,ROOT
from prepare_af_integrated import atomic_json


def run(base):
    import google.colab
    assert torch.cuda.is_available(),'Colab GPU required'
    root=base/'models/team_quality_reproduction_v1';root.mkdir(parents=True,exist_ok=True)
    raw=base/'raw/but_quality';raw.mkdir(parents=True,exist_ok=True)
    spec=json.loads((ROOT/'data/catalog.json').read_text())['but']
    # Only quality labels and red/RGB PPG files are needed; ECG/ACC are not downloaded.
    files=[v for v in entries(spec,raw) if v[1].name=='quality-hr-ann.csv' or v[1].name.endswith(('_PPG.hea','_PPG.dat'))]
    assert len(files)==7777, f'Unexpected BUT PPG file count: {len(files)}'
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,_ in enumerate(pool.map(lambda v:download(*v),files),1):
            if i%250==0 or i==len(files):print('BUT_VERIFIED',i,len(files),flush=True)
    rf_notebook=ROOT/'PPG_quality_RF_improved.ipynb'
    cnn_notebook=next(ROOT.glob('02*11.ipynb'))
    identity={'rf_notebook_sha256':digest(rf_notebook),'cnn_notebook_sha256':digest(cnn_notebook),
        'script_sha256':digest(Path(__file__)),'gpu':torch.cuda.get_device_name(0),
        'note':'Team notebook reproduction, new run; not recovered original teammate weights.'}
    if (root/'source.json').exists():
        saved=json.loads((root/'source.json').read_text())
        for key in ['rf_notebook_sha256','cnn_notebook_sha256','script_sha256']:assert saved[key]==identity[key],'Changed reproduction code; use a new version directory'
    else:atomic_json(root/'source.json',identity)
    from but_quality_rf_runtime import build_but_features,train_holdout
    if not (root/'rf/complete.json').exists():
        frame,audit=build_but_features(raw,root/'rf_cache')
        bundle,scores,pred=train_holdout(frame,root/'rf')
        atomic_json(root/'rf/complete.json',{'sha256':digest(root/'rf/butppg_quality_rf.joblib'),'audit':audit})
        print('RF_TRAINING_COMPLETE',json.dumps(scores['model']),flush=True)
    if not (root/'cnn/complete.json').exists():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        def display(value):
            if hasattr(value,'shape'):print('TABLE',value.shape,flush=True)
        scope={'__name__':'team_cnn_reproduction','REPO_DIR':str(ROOT),'display':display}
        cells=json.loads(cnn_notebook.read_text(encoding='utf-8'))['cells']
        # Reviewed team cells: config, reader, preprocessing, training, CV, evaluation, final export.
        exec(compile(''.join(cells[6]['source']),str(cnn_notebook)+':config','exec'),scope)
        scope.update(BUT_DIR=str(raw),CACHE_PATH=str(root/'but_ppg_300_v2.npz'),
            RESULT_DIR=str(root/'cnn'),FIG_DIR=str(root/'cnn/figures'))
        Path(scope['FIG_DIR']).mkdir(parents=True,exist_ok=True)
        for index in [8,10,12,14,16,18,19,21,23]:
            print('CNN_TEAM_CELL',index,flush=True)
            exec(compile(''.join(cells[index]['source']),str(cnn_notebook)+f':cell{index}','exec'),scope)
            plt.close('all')
        atomic_json(root/'cnn/complete.json',{'sha256':digest(root/'cnn/best_model.pt'),'config_sha256':digest(root/'cnn/config.json')})
    assert digest(root/'rf/butppg_quality_rf.joblib')==json.loads((root/'rf/complete.json').read_text())['sha256']
    done=json.loads((root/'cnn/complete.json').read_text())
    assert digest(root/'cnn/best_model.pt')==done['sha256'] and digest(root/'cnn/config.json')==done['config_sha256']
    print('TEAM_QUALITY_READY',root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);run(p.parse_args().base)
