"""Build the single 61-input, 14-output Arena policy plus its timed roll cue."""
import argparse
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper, TensorProto as T
import policy as P


def export(checkpoint,folder):
    theta=json.loads(Path(checkpoint).read_text())['weights']
    model=onnx.load(str(P.base_path()))
    nodes=[]
    constants=list(model.graph.initializer)
    source=model.graph.input[0].name
    skate_output=model.graph.output[0].name
    nodes.extend(model.graph.node)
    def constant(name,values,dtype=np.float32):
        constants.append(numpy_helper.from_array(np.asarray(values,dtype=dtype),name))
        return name
    def op(kind,name,*args,**kwargs):
        nodes.append(h.make_node(kind,list(args),[name],name=name,**kwargs))
        return name
    cue=op('Gather','roll_cue',source,constant('cue_index',[60],np.int64),axis=1)
    raw=op('Mul','scaled_cue',cue,constant('sixteen',16))
    raw=op('Sub','one_based',raw,constant('one',1))
    raw=op('Cast','index64',op('Round','rounded',raw),to=T.INT64)
    raw=op('Reshape','frame',raw,constant('scalar_shape',[],np.int64))
    zero=constant('index_zero',0,np.int64)
    limit=constant('last_index',len(P.ACTION)-1,np.int64)
    idx=op('Clip','safe_index',raw,zero,limit)
    use=op('And','is_trick',op('GreaterOrEqual','after_start',raw,zero),op('LessOrEqual','before_end',raw,limit))
    obs=op('Gather','motor_sensors',source,constant('sensor_indices',P.INDICES,np.int64),axis=1)
    def table(name,arr):
        return op('Gather',name,constant(name+'_table',arr),idx,axis=0)
    delta=op('Sub','sensor_error',obs,table('expected',P.INPUT))
    correction=op('MatMul','feedback',delta,table('gains',P.feedback(theta)))
    clipped=op('Clip','bounded_feedback',correction,constant('feedback_min',-.8),constant('feedback_max',.8))
    joints=op('Add','roll_joints',table('trained_joints',P.ACTION),clipped)
    op('Where','motor_targets',use,joints,skate_output)
    graph=h.make_graph(nodes,'crown_roll',list(model.graph.input),
                       [h.make_tensor_value_info('motor_targets',T.FLOAT,[1,14])],constants)
    built=h.make_model(graph,ir_version=max(8,model.ir_version),opset_imports=[h.make_opsetid('',18)])
    onnx.checker.check_model(built, full_check=True)
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    onnx.save(built,str(folder/'policy.onnx'))
    frames=[]
    previous=None
    for i in range(P.STEPS):
        cmd=P.command(i)
        if previous is None or not np.array_equal(previous,cmd):
            frames.append(dict(t=round(i*P.DT,2),twist=[.65 if cmd[0] else 0,0,0],
                               body=[0,0,0,0,0,float(cmd[-1])]))
            previous=cmd
    act=dict(timeline_version=2,duration_s=P.STEPS*P.DT,
             ducks=[dict(policy='self',start=dict(x=-.85,y=0,yaw=0),keyframes=frames)])
    (folder/'timeline.json').write_text(json.dumps(act,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('folder',type=Path)
    args=parser.parse_args()
    export(args.checkpoint,args.folder)
