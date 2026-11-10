"""Train or assess the roll using complete episodes in the Arena roller simulation."""
import argparse
from copy import copy, deepcopy
from functools import lru_cache
import json
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np
from arena import runner
from arena.course import compile_stage
from arena.manifest import load_event
from arena.timeline import Start
from policy import Actor, command, PHASES, START_STEP, STEPS

EVENT=load_event('got-a-talent')
START=Start(x=-.85,y=0,yaw_deg=0)
COMMANDS=[command(i) for i in range(STEPS)]
DT=.02
SUB_DT=.005


@lru_cache(maxsize=32)
def entry(seed):
    model=compile_stage(EVENT,[('rollers',START)])
    sim,_,_=runner._start(EVENT,model,[Actor(np.zeros(12))],[0],seed,[COMMANDS[0]])
    trace=np.empty((START_STEP,21),np.float32)
    for i in range(START_STEP):
        sim.step([COMMANDS[i]])
        sim.record(trace[i:i+1])
    neutral=mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,neutral,0)
    mujoco.mj_forward(model,neutral)
    assert model.mesh(model.geom_dataid[6]).name=='top_head_shell'
    return sim,trace,neutral.geom_xmat[6].reshape(3,3)[2].copy()


def run(actor,seed):
    pre,intro,local_up=entry(seed)
    sim=copy(pre)
    sim.model,sim.data=copy(pre.model),copy(pre.data)
    sim.ducks=deepcopy(pre.ducks,{id(pre.model):sim.model,id(pre.ducks[0].policy):actor})
    d=sim.data
    trunk=sim.ducks[0].adr.trunk
    masks=[]
    def contact():
        mask=0
        for c in d.contact:
            a,b=c.geom
            if 13 not in (a,b) or c.dist>.0005:
                continue
            geom=a if b==13 else b
            if geom in (4,5): mask|=1
            elif geom in (11,12): mask|=2
            elif geom in (6,7,8): mask|=8
            else: mask|=4
        masks.append(mask)
        return False
    trace=np.empty((STEPS-START_STEP,21),np.float32)
    signals=np.empty((len(trace),10))
    for i in range(len(trace)):
        sim.step([COMMANDS[START_STEP+i]],contact)
        sim.record(trace[i:i+1])
        head=d.geom_xpos[6]
        crown_up=d.geom_xmat[6].reshape(3,3)@local_up
        crown=False
        mask=0
        for c in d.contact:
            a,b=c.geom
            if 13 not in (a,b) or c.dist>.0005: continue
            g=a if b==13 else b
            if g==6 and (c.pos-head)@crown_up>.009: crown=True
            if g in (4,5): mask|=1
            if g in (11,12): mask|=2
        rot=d.xmat[trunk].reshape(3,3)
        overhead=min(d.geom_xpos[4:6,2].mean(),d.geom_xpos[11:13,2].mean())-head[2]
        signals[i]=[np.arctan2(-rot[2,0],rot[0,0]),rot[2,2],abs(rot[2,1]),overhead,
                    crown,bool(mask&1),bool(mask&2),0,0,d.subtree_com[trunk,0]]
    return (intro,trace,signals),np.array(masks,dtype=np.uint8)


def metrics(result,masks):
    _,trace,f=result
    pitch=np.degrees(np.unwrap(f[:,0]))
    inverted=(f[:,1]<-.65)&(f[:,3]>.055)
    crowns=np.flatnonzero(inverted & (f[:,4]>0))
    t0=(crowns[0]+1)*4 if len(crowns) else 4*np.argmin(f[:,1])
    t0=min(int(t0),len(masks)-1)
    left=np.flatnonzero((masks[t0:]&1)>0)+t0
    right=np.flatnonzero((masks[t0:]&2)>0)+t0
    touch=bool(len(left) and len(right))
    first=min(left[0],right[0]) if touch else len(masks)-1
    between=abs(int(left[0])-int(right[0]))*SUB_DT if touch else 2.
    heads=np.flatnonzero((masks[t0:first+1]&8)>0)+t0
    gap=(first-heads[-1])*SUB_DT if len(heads) and touch else 2.
    scrape=np.count_nonzero(masks[t0:]&4)*SUB_DT
    after_head=bool(np.any(masks[first+16:]&8))
    feet=(masks&3)>0
    support=feet[first:first+60].mean()
    upright=(f[:,1]>.85)&(trace[:,2]>.085)
    secure=np.flatnonzero(np.convolve(upright.astype(int),np.ones(8,int),'valid')==8)
    secure=secure[secure>=first//4]
    rise=max(0,(int(secure[0])*4-first)*SUB_DT) if len(secure) else 2.
    exit_speed=float((f[-1,9]-f[-16,9])/.30)
    early=np.flatnonzero(pitch>45)
    late=np.flatnonzero(pitch>315)
    path=pitch[early[0]:late[0]+1] if len(early) and len(late) else np.array([])
    smooth=len(path)>6 and min(path[6:]-path[:-6])>8
    rotated=bool(smooth and (len(path)-1)*DT<=.85 and 320<np.median(pitch[-15:])<400)
    rolling=bool(upright[-20:].all() and ((f[-20:,5]+f[-20:,6])>0).mean()>=.6)
    success=bool(len(crowns) and touch and rotated and rolling and not after_head and scrape==0
                 and between<=.1 and gap<=.23 and rise<=.22 and support>=.5 and exit_speed>.13)
    reward=(4*success + 1.5*rolling + 1.5*rotated + 1.0*bool(len(crowns)) +
            np.clip(1-gap/.3,-1,1)+np.clip(1-rise/.3,-1,1)+support-
            14*min(.5,scrape)-1.5*after_head+np.clip(exit_speed/.45,-1,1))
    return dict(direct=success,reward=round(float(reward),5),body_ms=round(1000*scrape),
                head_to_skates_ms=round(gap*1000),between_feet_ms=round(between*1000),
                stable_after_ms=round(rise*1000),exit_mps=round(exit_speed,3))


def episode(job):
    theta,seed=job
    return metrics(*run(Actor(theta),int(seed)))


def batches(pool,vectors,seeds):
    flat=pool.map(episode,[(v,seed) for v in vectors for seed in seeds],chunksize=2)
    return [flat[i:i+len(seeds)] for i in range(0,len(flat),len(seeds))]


def ranking(rows):
    return np.mean([r['reward'] for r in rows]) + 2*np.mean([r['direct'] for r in rows])


def train(out,workers,generations):
    rng=np.random.default_rng(292507)
    shape=PHASES*3
    centre=np.ones(shape)
    spread=np.full(shape,.56)
    best=centre.copy()
    best_score=-np.inf
    history=[]
    with Pool(workers) as pool:
        for g in range(generations):
            pop=np.clip(centre+spread*rng.normal(size=(32,shape)),-1.2,3.5)
            pop[:2]=[centre,best]
            seeds=rng.choice(np.arange(2400,2448),6,replace=False)
            scores=[ranking(rows) for rows in batches(pool,pop,seeds)]
            elite=np.argsort(scores)[-6:]
            centre=.25*centre+.75*pop[elite].mean(0)
            spread=np.maximum(.12,.68*spread+.32*pop[elite].std(0))
            finalists=[pop[elite[-1]],centre]
            valid=batches(pool,finalists,range(2750,2766))
            for v,rows in zip(finalists,valid):
                score=ranking(rows)
                if score>best_score:
                    best_score=score
                    best=v.copy()
                    out.write_text(json.dumps(dict(weights=best.tolist(),generation=g+1,selection_score=score),indent=2)+'\n')
            history.append(dict(generation=g+1,best_selection=best_score))
            print(g+1,round(best_score,4),flush=True)
    return history


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--generations',type=int,default=18)
    args=parser.parse_args()
    train(args.output,args.workers,args.generations)
