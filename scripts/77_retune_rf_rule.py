import sys, itertools, json, importlib.util, yaml
sys.path.insert(0,'.')
from collections import Counter
from oesc_rag.data.build_benchmark import load_benchmark
from oesc_rag.evaluation.serialization import load_results
from oesc_rag.controller.rule_classifier import classify_state
spec=importlib.util.spec_from_file_location('t71','scripts/71_train_retrieve_first_policies.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
G=dict(coverage=[.1,.2,.3,.4,.5,.6],support=[.1,.2,.3,.4,.5,.6],conflict=[.05,.1,.15,.2,.25,.3,.4,.5],
       critical_need=[0,.3,.6],unanswerable_coverage=[.1,.2,.3],unanswerable_support=[.3,.4,.5])
out={}
for bb in ['deepseek41flash','qwen38flash']:
    cfg=yaml.safe_load(open(f'configs/valid_{bb}_rf_rule.yaml'))
    dis=frozenset(cfg['controller']['disabled_states'])
    gold={s.id:s for s in load_benchmark('data/benchmark/valid_dev.json')}
    rows=[]
    for r in load_results(f'data/results/rfdev_{bb}/oesc_results.json'):
        s=gold.get(r.sample_id)
        if s is None or len(r.rounds)<2: continue
        rows.append((r.rounds[1].features,bool((r.metadata or {}).get('time_sensitive')),m.observed_state(s,r.rounds[1])))
    labs=sorted(set(x[2] for x in rows))
    def score(th):
        t={**cfg['thresholds'],**th}
        hit=Counter(); tot=Counter(x[2] for x in rows)
        for f,ts,y in rows:
            if classify_state(f,t,time_sensitive=ts,round_idx=1,disabled_states=dis)==y: hit[y]+=1
        return sum(hit[k]/tot[k] for k in labs)/len(labs), sum(hit.values())/len(rows)
    base=score({})
    best=None
    keys=list(G)
    for vals in itertools.product(*G.values()):
        th=dict(zip(keys,vals)); sc=score(th)
        if best is None or sc[0]>best[0][0]+1e-12: best=(sc,th)
    print(bb,'base bal/acc',[round(v,3) for v in base],'best',[round(v,3) for v in best[0]],best[1])
    out[bb]={'objective':'balanced accuracy over post-retrieval observed states on valid_dev round-1 features','n':len(rows),'base':base,'best':best[0],'thresholds':best[1],'grid':G}
json.dump(out,open('data/results/rf_rule_retune.json','w'),indent=1)
