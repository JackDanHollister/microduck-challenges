"""Reference-tracking feedback for the roll and the Roller Sprint network for skating."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import os

import numpy as np
from arena.policy import load_policy
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
BASE_REPO = 'JackDH/microduck-roller-sprint-20261110-best'
BASE_REV = '43e9b57825fbfe08cf4b510368b643ff9ae47d93'
BASE_SHA = '0b724f5cebe7f50815f6dd9631009e47c4b0a607f81b22d0f4048ffd6d5a77d6'
START_STEP = 70
STEPS = 174
DT = .02
PHASES = 4
FEATURE_GROUP = np.array([0,0,0,1,1,1,2,2,2,2,2,2,2,2])
with np.load(ROOT/'reference.npz') as ref:
    INPUT = ref['observation']
    ACTION = ref['joints']
    INDICES = ref['indices']
    WARM = ref['initial_feedback']
FRAME_PHASE = np.minimum(np.arange(len(ACTION))*PHASES//len(ACTION), PHASES-1)


@lru_cache(maxsize=1)
def base_path():
    local = os.environ.get('MICRODUCK_SKATE_ONNX')
    path = Path(local or hf_hub_download(BASE_REPO,'policy.onnx',revision=BASE_REV))
    if sha256(path.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError('Roller Sprint policy checksum differs from checkpoint 500')
    return path


@lru_cache(maxsize=1)
def skater():
    return load_policy(base_path())


def feedback(theta):
    gains = np.asarray(theta,np.float32).reshape(PHASES,3)
    return WARM * gains[FRAME_PHASE][:,FEATURE_GROUP,None]


def command(step):
    out=np.zeros(13,np.float32)
    if step>=10:
        out[0]=.65
    slot=step-START_STEP
    if 0<=slot<len(ACTION):
        out[-1]=(slot+1)/16
    return out


class Actor:
    def __init__(self, theta):
        self.gains=feedback(theta)

    def act(self, obs):
        slot=int(round(float(obs[-1])*16))-1
        if 0<=slot<len(ACTION):
            delta=obs[INDICES]-INPUT[slot]
            correction=np.clip(delta @ self.gains[slot],-.8,.8)
            return ACTION[slot]+correction
        model=skater()
        return model.session.run([model._output_name],{model._input_name:obs.reshape(1,61)})[0][0]
