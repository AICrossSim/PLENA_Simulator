"""Bounded online prediction state. Estimates never grant resource readiness."""
from __future__ import annotations
import math,random

class Predictor:
    def __init__(self,name='ours',seed=20261007):
        if name not in ('random','static','btb','ema','ours','oracle'):raise ValueError(name)
        self.name=name;self.rng=random.Random(seed);self.means=[None,None];self.counts=[0,0]
        self.table={};self.correction={};self.profile={};self.calls=0
    @staticmethod
    def key(e,c):return (c,bool(e.get('is_shared')),min(e['Me'],9))
    @staticmethod
    def okey(e,c):return (c,min(7,int(math.ceil(math.log2(max(1,e['Me']))))))
    @staticmethod
    def exactkey(e,c):return (c,e.get('id'),e['Me'],e.get('H'),e.get('F'))
    def predict(self,e,c,nominal,**kw):
        self.calls+=1;mean=self.means[c] if self.means[c] is not None else nominal
        if self.name=='random':return self.rng.uniform(0,2*mean)
        if self.name=='static':return mean
        if self.name in ('btb','ema'):return self.table.get(self.key(e,c),mean)
        if self.name=='oracle':return self.profile.get(self.exactkey(e,c),nominal)
        # The supplied cost is the explicit phase resource model; it includes cold data latency.
        return nominal*self.correction.get(self.okey(e,c),1.0)
    def on_complete(self,e,c,predicted,actual,nominal=None):
        n=self.counts[c];self.counts[c]=min(65535,n+1)
        self.means[c]=actual if self.means[c] is None else self.means[c]+(actual-self.means[c])/max(1,self.counts[c])
        if self.name=='btb':self.table[self.key(e,c)]=actual
        if self.name=='ema':
            k=self.key(e,c);old=self.table.get(k,actual);self.table[k]=old+(actual-old)/4
        if self.name=='ours':
            k=self.okey(e,c);old=self.correction.get(k,1.0)
            nominal=nominal if nominal is not None else predicted/max(old,1e-9)
            target=max(.25,min(4.0,actual/max(nominal,1e-6)))
            # Explicit Q16.16 coefficient: fixed storage, saturating update.
            self.correction[k]=round(65536*(old+(target-old)/4))/65536
    def on_progress(self,e,c,elapsed,quarter,remaining):
        if self.name!='ours' or quarter<=0:return None
        # Measured service rate projected over the uncompleted work. Does not assert completion.
        return max(1.0,elapsed*(1-quarter)/quarter)
    def absorb_profile(self,result,workload):
        for t in result.get('tasks',[]):
            e=workload['experts'][t['expert_index']]
            self.profile[self.exactkey(e,t['core'])]=t['actual_cycles']
    def state_bits(self):
        # Installed table sizes, not only entries touched in one run. Core means+sample counts+valid.
        base=2*(32+16+1)
        if self.name=='random':return base+64
        if self.name=='static':return base
        if self.name in ('btb','ema'):return base+2*2*9*(32+1)
        if self.name=='ours':return base+2*8*(32+1)+2*(32*3+2)
        return None # profile-guided reference is not synthesizable hardware.
