"""NumPy neural classification with separate training, calibration and policy splits."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

DATA_SHA256 = '09f66e6debdee2cd2b5ae59e0d6abbb73fc2b0e0185d2e1957e9ebb51e23aa22'


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def nll(logits, labels, temperature=1.0):
    z = logits / temperature
    m = z.max(axis=1)
    return float(np.mean(m + np.log(np.exp(z-m[:,None]).sum(axis=1)) - z[np.arange(len(labels)),labels]))


def load_data(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError('dataset checksum does not match the documented source')
    with gzip.open(path, 'rt') as f:
        data = np.loadtxt(f, delimiter=',')
    if data.shape != (1797, 65) or not np.isfinite(data).all():
        raise ValueError('unexpected dataset shape or values')
    return data[:,:64]/16.0, data[:,64].astype(int)


def split_indices(y, seed):
    rng = np.random.default_rng(seed)
    splits = {name:[] for name in ('train','calibration','policy','test')}
    for label in np.unique(y):
        ids = rng.permutation(np.flatnonzero(y == label))
        a,b,c = int(len(ids)*.55), int(len(ids)*.70), int(len(ids)*.85)
        for name, part in zip(splits,np.split(ids,[a,b,c]),strict=True):
            splits[name].extend(part.tolist())
    return {name:np.array(sorted(ids),dtype=int) for name,ids in splits.items()}


def initialize(features=64, hidden=64, classes=10, seed=42):
    rng = np.random.default_rng(seed)
    if hidden == 0:
        return {'w':rng.normal(0,.01,(features,classes)), 'b':np.zeros(classes)}
    return {'w1':rng.normal(0,np.sqrt(2/features),(features,hidden)), 'b1':np.zeros(hidden),
            'w2':rng.normal(0,np.sqrt(2/hidden),(hidden,classes)), 'b2':np.zeros(classes)}


def logits(params,x):
    if 'w' in params:
        return x@params['w']+params['b']
    return np.maximum(0,x@params['w1']+params['b1'])@params['w2']+params['b2']


def loss_grad(params,x,y,decay=1e-4):
    z = logits(params,x)
    loss = nll(z,y) + decay/2 * sum(float(np.sum(v*v)) for k,v in params.items() if k.startswith('w'))
    dz = softmax(z)
    dz[np.arange(len(y)),y] -= 1
    dz /= len(y)
    if 'w' in params:
        return loss, {'w':x.T@dz+decay*params['w'],'b':dz.sum(axis=0)}
    pre = x@params['w1']+params['b1']
    h = np.maximum(0,pre)
    dh = (dz@params['w2'].T)*(pre>0)
    return loss, {'w1':x.T@dh+decay*params['w1'],'b1':dh.sum(axis=0),
                  'w2':h.T@dz+decay*params['w2'],'b2':dz.sum(axis=0)}


def train(x,y,hidden=64,seed=42,epochs=120):
    params = initialize(x.shape[1],hidden,10,seed)
    m = {k:np.zeros_like(v) for k,v in params.items()}
    v = {k:np.zeros_like(a) for k,a in params.items()}
    rng = np.random.default_rng(seed)
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0,len(y),64):
            ids = order[start:start+64]
            _,grads = loss_grad(params,x[ids],y[ids])
            step += 1
            for k,g in grads.items():
                m[k] = .9*m[k]+.1*g
                v[k] = .999*v[k]+.001*g*g
                params[k] -= .003*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
    return params


def calibrate(z,y):
    grid = np.unique(np.r_[np.geomspace(.25,4,161),1.0])
    return float(min(grid,key=lambda t:nll(z,y,t)))


def choose_threshold(prob,y,target=.98,min_coverage=.5):
    confidence,correct = prob.max(axis=1), prob.argmax(axis=1)==y
    candidates = []
    for threshold in np.unique(np.r_[0,confidence]):
        keep = confidence >= threshold
        if keep.mean() >= min_coverage and correct[keep].mean() >= target:
            candidates.append((int(keep.sum()),float(threshold)))
    # Explicitly abstain on everything when the policy constraint cannot be met.
    return min(candidates,key=lambda c:(-c[0],c[1]))[1] if candidates else 1.000001


def metrics(prob,y,threshold=0):
    pred = prob.argmax(axis=1)
    confidence = prob.max(axis=1)
    correct = pred == y
    keep = confidence >= threshold
    cm = np.zeros((10,10),dtype=int)
    np.add.at(cm,(y,pred),1)
    f1 = [2*cm[k,k]/max(1,cm[k,:].sum()+cm[:,k].sum()) for k in range(10)]
    bins = []
    for lo in np.arange(10)/10:
        hi = lo+.1
        mask = (confidence >= lo)&((confidence<hi) if lo<.9 else (confidence<=1))
        bins.append({'lower':float(lo),'count':int(mask.sum()),
                     'confidence':float(confidence[mask].mean()) if mask.any() else None,
                     'accuracy':float(correct[mask].mean()) if mask.any() else None})
    ece = sum(b['count']/len(y)*abs(b['confidence']-b['accuracy']) for b in bins if b['count'])
    return {'n':len(y),'accuracy':float(correct.mean()),'macro_f1':float(np.mean(f1)),
            'nll':float(-np.log(np.clip(prob[np.arange(len(y)),y],1e-15,1)).mean()),
            'brier_multiclass':float(np.sum((prob-np.eye(10)[y])**2,axis=1).mean()),
            'ece_10_bins':float(ece),'coverage':float(keep.mean()),'accepted':int(keep.sum()),
            'selective_accuracy':float(correct[keep].mean()) if keep.any() else None,
            'confusion_matrix':cm.tolist(),'reliability_bins':bins,
            'per_class_recall':[float(cm[k,k]/cm[k,:].sum()) if cm[k,:].sum() else None for k in range(10)]}


def bootstrap_accuracy(prob,y,seed):
    correct = (prob.argmax(axis=1)==y).astype(float)
    rng = np.random.default_rng(seed)
    values = [float(rng.choice(correct,len(correct),replace=True).mean()) for _ in range(2000)]
    return np.quantile(values,[.025,.975]).tolist()


def make_artifact(params,temperature,threshold):
    return {'version':1,'input_shape':[64],'input_range':[0,16],'scale':16,
            'temperature':temperature,'threshold':threshold,
            'weights':{k:v.tolist() for k,v in params.items()}}


def predict(artifact,pixels):
    if artifact.get('version') != 1 or artifact.get('scale') != 16:
        raise ValueError('unsupported artifact')
    x = np.asarray(pixels,dtype=float)
    if x.ndim != 2 or x.shape[1] != 64 or len(x)==0 or not np.isfinite(x).all() or x.min()<0 or x.max()>16:
        raise ValueError('pixels must be a nonempty N-by-64 array of finite values in [0,16]')
    temp,threshold = artifact['temperature'],artifact['threshold']
    if not math.isfinite(temp) or temp<=0 or not math.isfinite(threshold) or not 0<=threshold<=1.000001:
        raise ValueError('invalid calibration/policy parameters')
    params = {k:np.asarray(v,dtype=float) for k,v in artifact['weights'].items()}
    expected = {'w1':(64,64),'b1':(64,),'w2':(64,10),'b2':(10,)}
    if set(params)!=set(expected) or any(params[k].shape!=shape or not np.isfinite(params[k]).all() for k,shape in expected.items()):
        raise ValueError('invalid model weight schema')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        z = logits(params,x/16)/temp
    if not np.isfinite(z).all():
        raise ValueError('model produced nonfinite logits')
    prob = softmax(z)
    return [{'prediction':int(p.argmax()),'confidence':float(p.max()),
             'decision':'accept' if p.max()>=threshold else 'review'} for p in prob]


def experiment(data_path,output,seeds=(42,43,44)):
    x,y = load_data(data_path)
    output.mkdir(parents=True,exist_ok=True)
    results = []
    for seed in seeds:
        split = split_indices(y,seed)
        a,c,v,t = (split[n] for n in ('train','calibration','policy','test'))
        baseline = train(x[a],y[a],hidden=0,seed=seed)
        model = train(x[a],y[a],seed=seed)
        temperature = calibrate(logits(model,x[c]),y[c])
        threshold = choose_threshold(softmax(logits(model,x[v])/temperature),y[v])
        raw = softmax(logits(model,x[t]))
        prob = softmax(logits(model,x[t])/temperature)
        rng = np.random.default_rng(seed+1000)
        shifted = np.clip(x[t]+rng.normal(0,.2,x[t].shape),0,1)
        majority = int(np.bincount(y[a]).argmax())
        result = {'seed':seed,'split_counts':{k:len(ids) for k,ids in split.items()},
                  'temperature':temperature,'threshold':threshold,
                  'majority_accuracy':float(np.mean(y[t]==majority)),
                  'linear_baseline':metrics(softmax(logits(baseline,x[t])),y[t]),
                  'neural_uncalibrated':metrics(raw,y[t]),
                  'neural_calibrated':metrics(prob,y[t],threshold),
                  'test_accuracy_bootstrap_95_percentile':bootstrap_accuracy(prob,y[t],seed),
                  'policy_split':metrics(softmax(logits(model,x[v])/temperature),y[v],threshold),
                  'noise_stress_sigma_0_2':metrics(softmax(logits(model,shifted)/temperature),y[t],threshold),
                  'risk_coverage_curve':[{'threshold':float(q),'coverage':metrics(prob,y[t],q)['coverage'],
                                          'selective_accuracy':metrics(prob,y[t],q)['selective_accuracy']} for q in (0,.5,.7,.8,.9,.95,.99)],
                  'errors':[{'row':int(t[i]),'true':int(y[t[i]]),'predicted':int(prob[i].argmax()),
                             'confidence':float(prob[i].max())} for i in range(len(t)) if prob[i].argmax()!=y[t[i]]]}
        results.append(result)
        if seed == seeds[0]:
            artifact = make_artifact(model,temperature,threshold)
            (output/'model.json').write_text(json.dumps(artifact,allow_nan=False))
            (output/'splits.json').write_text(json.dumps({k:v.tolist() for k,v in split.items()},indent=2))
            (output/'example.json').write_text(json.dumps((x[t[:3]]*16).tolist()))
    report = {'dataset_sha256':DATA_SHA256,'numpy':np.__version__,'python':platform.python_version(),
              'protocol':'55/15/15/15 stratified random row split; fixed 120 epochs; calibration and policy disjoint; no test-driven model selection',
              'runs':results}
    (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    summary = '\n'.join(f"Seed {r['seed']}: accuracy={r['neural_calibrated']['accuracy']:.4f}, coverage={r['neural_calibrated']['coverage']:.4f}, accepted accuracy={r['neural_calibrated']['selective_accuracy']}" for r in results)
    print(summary)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    run = sub.add_parser('experiment')
    run.add_argument('--data',type=Path,default=Path('data/digits.csv.gz'))
    run.add_argument('--output',type=Path,default=Path('results/local'))
    infer = sub.add_parser('predict')
    infer.add_argument('--model',type=Path,required=True)
    infer.add_argument('--input',type=Path,required=True)
    args = parser.parse_args(argv)
    try:
        if args.command=='experiment': experiment(args.data,args.output)
        else: print(json.dumps(predict(json.loads(args.model.read_text()),json.loads(args.input.read_text())),indent=2,allow_nan=False))
    except (OSError,ValueError,KeyError,TypeError) as error:
        parser.exit(2,f'selective-ml: {error}\n')


if __name__ == '__main__': main()
