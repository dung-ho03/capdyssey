"""Reproduce stored PPG inputs from raw files and diagnose resampling edges in Colab."""
import argparse
import json
from pathlib import Path


def run(base):
    import google.colab
    import h5py
    import numpy as np
    from scipy.signal import resample_poly
    from prepare_af_integrated import chars,start_time
    from train_af import preprocess_window,write_json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=base/'models/provisional_exact01_v1/error_audit_v1'
    audit=json.loads((folder/'report.json').read_text());checks=[]
    for case in audit['random_example_cases']:
        person,recording=case['record'].split('_');recording=int(recording)
        with h5py.File(base/'raw/longterm_af_v3'/f'{person}_PPG.mat') as f:
            begin=start_time(chars(f[f['recording_startday'][()].ravel()[recording]]),chars(f[f['recording_starttime'][()].ravel()[recording]]))
            lo=int(round((case['start_seconds']-begin)*100));hi=lo+1000
            ds=f[f['PPG_GREEN'][()].ravel()[recording]]
            raw=(ds[0,lo:hi] if ds.shape[0]==1 else ds[lo:hi,0]).astype(float)
        with np.load(base/'processed/provisional_exact01_v1'/case['shard']) as d:stored=d['x'][case['index']]
        original=preprocess_window(raw,100,seconds=10)
        centered=resample_poly(raw-raw.mean(),1,2);centered=(centered-centered.mean())/centered.std()
        edge=np.r_[np.arange(10),np.arange(490,500)]
        checks.append({'subject':person,'outcome':case['outcome'],'record':case['record'],
            'raw_dc_to_std':float(abs(raw.mean())/raw.std()),
            'stored_reproduction_max_error':float(np.max(abs(stored-original))),
            'original_peak_index':int(np.argmax(abs(original))),
            'original_edge_energy_fraction':float(np.sum(original[edge]**2)/np.sum(original**2)),
            'centered_edge_energy_fraction':float(np.sum(centered[edge]**2)/np.sum(centered**2)),
            'original_peak_z':float(max(abs(original))),'centered_peak_z':float(max(abs(centered)))})
    write_json(folder/'raw_boundary_check.json',checks)
    fig,axes=plt.subplots(2,1,figsize=(11,6))
    for ax,values,title in zip(axes,[original,centered],['Current preprocessing','Diagnostic: remove DC before resampling; no model change']):
        ax.plot(np.arange(500)/50,values);ax.set_title(title);ax.set_xlabel('Seconds');ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(folder/'boundary_comparison.png',dpi=130);plt.close(fig)
    print('RAW_BOUNDARY_CHECK',json.dumps(checks));print('BOUNDARY_AUDIT_COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    run(p.parse_args().base)
