import numpy as np
import pytest
import ref_numerics as rn


def fixture(seed=29):
    rng=np.random.default_rng(seed); T,d,I,r=3,544,160,8
    x=rn.bf16_round(rng.normal(size=(T,d)).astype(np.float32)*.1)
    ws={k:rn.bf16_round(rng.normal(size=(I,d) if k!='d' else (d,I)).astype(np.float32)*.03) for k in ('g','u','d')}
    comp={k:(rn.quantize_factor(rng.normal(size=(w.shape[1],r)).astype(np.float32)*.02,'mxint4',0),rn.bf16_round(rng.normal(size=(r,w.shape[0])).astype(np.float32)*.02)) for k,w in ws.items()}
    return x,ws,comp,rng.uniform(.1,.8,T).astype(np.float32)


def test_hw_contract_algorithm_tolerance():
    x,w,c,g=fixture()
    out,state=rn.expert_ffn_hw(x,w['g'],w['u'],w['d'],c,g,return_state=True)
    ref=rn.expert_ffn(x,w['g'],w['u'],w['d'],c,g)
    assert rn.rel_err(out,ref)<5e-3
    for k in ('X','Z','U_g','U_u','U_d'):
        assert np.array_equal(state[k],rn.bf16_round(state[k]))
    assert np.array_equal(out,rn.expert_ffn_hw(x,w['g'],w['u'],w['d'],c,g))


def test_streamed_rank_tails_are_explicit():
    x,w,c,g=fixture(); out,state=rn.expert_ffn_hw(x,w['g'],w['u'],w['d'],c,g,L=4,z_mode='streamed',return_state=True)
    assert state['rank_segments']['d']==[[0,1,2,3]]
    assert state['tail_segments']['d']==[[4,5,6,7]]
    assert rn.rel_err(out,rn.expert_ffn(x,w['g'],w['u'],w['d'],c,g))<5e-3


def test_merge_uses_completion_order():
    y={0:np.array([[1e20]],np.float32),1:np.array([[-1e20]],np.float32),2:np.array([[1]],np.float32)}; ids={0:[0],1:[0],2:[0]}
    assert rn.combine_hw(y,ids,[0,1,2])[0,0]==1
    assert rn.combine_hw(y,ids,[0,2,1])[0,0]==0
    with pytest.raises(ValueError): rn.combine_hw(y,ids,[0,1,1])


def test_gate_compensation_before_nonlinearity():
    x,w,c,g=fixture(); y=rn.expert_ffn_hw(x,w['g'],w['u'],w['d'],c,g)
    wrong=rn.expert_ffn(x,w['g'],w['u'],w['d'],c,g,order='after_silu')
    assert rn.rel_err(y,wrong)>1e-3


def test_streamed_me7_down_k2_tails_partial_merge():
    rng=np.random.default_rng(83); T,H,F,r,L=7,160,544,12,4
    x=rn.bf16_round(rng.normal(size=(T,H)).astype(np.float32)*.1)
    w={k:rn.bf16_round(rng.normal(size=(F,H) if k!='d' else (H,F)).astype(np.float32)*.02) for k in ('g','u','d')}
    comp={k:(rn.quantize_factor(rng.normal(size=(v.shape[1],r)).astype(np.float32)*.03,'mxint4',0),rn.bf16_round(rng.normal(size=(r,v.shape[0])).astype(np.float32)*.03)) for k,v in w.items()}
    y,s=rn.expert_ffn_hw(x,w['g'],w['u'],w['d'],comp,L=L,z_mode='streamed',return_state=True)
    assert s['rank_segments']['d']==[[],list(range(4))]
    assert s['tail_segments']['d']==[list(range(4,8)),list(range(8,12))]
    c=rn.down_partial_contributions_hw(s['Z'],s['U_d'],w['d'],comp['d'][1],s['rank_segments']['d'],s['tail_segments']['d'])
    assert np.array_equal(c['output'],y)
    events=[]
    for main,ks,ranks in c['source_keys']:
        for col in range(0,H,32):
            events.append({'expert_id':9,'col':col,'cols':min(32,H-col),'tokens':list(range(T)), 'partial':True,
                           'sources':[{'n':n,'k_segment':ks,'main':main,'ranks':list(ranks)} for n in range(col,min(H,col+32),4)]})
    got=rn.combine_partial_hw({9:c},{9:list(range(T))},events)
    assert np.array_equal(got,y)
    with pytest.raises(ValueError,match='exactly once'):
        rn.combine_partial_hw({9:c},{9:list(range(T))},events[:-1])
    with pytest.raises(ValueError,match='twice'):
        rn.combine_partial_hw({9:c},{9:list(range(T))},events+events[:1])


def test_partial_combine_preserves_interexpert_atomic_order():
    a=np.array([[1e20]],np.float32); b=np.array([[-1e20]],np.float32); one=np.array([[1]],np.float32)
    c={0:{'parts':[a,one],'source_keys':[(True,0,()),(False,0,(0,))],'output':a},1:b}
    e0={'expert_id':0,'col':0,'cols':1,'partial':True,'sources':[{'n':0,'k_segment':0,'main':True,'ranks':[]}]}
    e1={'expert_id':1,'col':0,'cols':1,'partial':False}
    e2={'expert_id':0,'col':0,'cols':1,'partial':True,'sources':[{'n':0,'k_segment':0,'main':False,'ranks':[0]}]}
    assert rn.combine_partial_hw(c,{0:[0],1:[0]},[e0,e1,e2])[0,0]==1
    assert rn.combine_partial_hw(c,{0:[0],1:[0]},[e0,e2,e1])[0,0]==0
