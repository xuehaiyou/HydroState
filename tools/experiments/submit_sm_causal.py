#!/usr/bin/env python
"""Snapshot source and submit a fail-closed gate -> smoke -> A screen -> eval DAG."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--gate-job',required=True)
    args=p.parse_args()
    root=args.root.resolve()
    jobs_path=root/'jobs.json'
    if jobs_path.exists():
        raise FileExistsError(f'{jobs_path}: refusing duplicate submission')
    gate=json.loads((root/'gate/passed.json').read_text())
    component=Path('hydrostate/experiments/sm_causal.py')
    if hashlib.sha256(component.read_bytes()).hexdigest()!=gate['component_sha256']:
        raise RuntimeError('Gate is stale')
    # Freeze all importable project code and experiment launchers before queuing.
    code=root/'code'
    code.mkdir(exist_ok=False)
    for name in ('hydrostate','tools','configs','scripts','tests'):
        shutil.copytree(name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    hashes={str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in code.rglob('*') if p.is_file()}
    (root/'code_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    config_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (root/'screen').glob('*.py')}
    (root/'config_sha256.json').write_text(json.dumps(config_hashes,indent=2)+'\n')
    jobs={'gate':args.gate_job,'code':str(code)}
    def save():
        jobs_path.write_text(json.dumps(jobs,indent=2)+'\n')
    save()
    def submit(name,options,script,*extra):
        result=subprocess.run(['sbatch','--parsable','--kill-on-invalid-dep=yes',*options,
            str(code/'scripts/slurm'/script),str(root),*extra],check=True,capture_output=True,text=True)
        job=result.stdout.strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError(result.stdout)
        jobs[name]=job
        save()
        print(name,job,flush=True)
        return job
    submit('temporal_audit',[], '18_sm_temporal_audit.sh')
    smoke=submit('smoke',[f'--dependency=afterok:{args.gate_job}','--time=01:00:00','--job-name=hs_sm_smoke'],
                 '15_sm_causal_train.sh','smoke')
    train=submit('train_screen',[f'--dependency=afterok:{smoke}','--array=0-3%2'],
                 '15_sm_causal_train.sh','train')
    evaluation=submit('evaluate_screen',[f'--dependency=aftercorr:{train}','--array=0-3%2'],
                      '16_sm_causal_evaluate.sh')
    submit('report',[f'--dependency=afterany:{evaluation}'],'17_sm_causal_report.sh')
    print(json.dumps(jobs,indent=2))


if __name__=='__main__':
    main()
