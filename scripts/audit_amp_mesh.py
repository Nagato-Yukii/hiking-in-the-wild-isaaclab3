"""Independent AMP math, same-step history, and terrain topology diagnostics."""
import argparse, json, random, re
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from hikinglab3.motions import MotionDataset


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mesh',action='store_true')
    parser.add_argument('--legacy-motion',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    run=Path('/home/ye/Desktop/HikingLab3/logs/g1-newton-2048-20k-20261104_062959')
    names=json.loads((run/'motion_manifest.json').read_text())['joint_names']
    defaults={'.*_hip_pitch_joint':-.312,'.*_knee_joint':.669,'.*_ankle_pitch_joint':-.363,'.*_elbow_joint':.6,'left_shoulder_roll_joint':.2,'right_shoulder_roll_joint':-.2,'.*_shoulder_pitch_joint':.2}
    q0=torch.tensor([[next((v for k,v in defaults.items() if re.fullmatch(k,n)),0.) for n in names]])
    dataset_class=MotionDataset
    if args.legacy_motion:
        import importlib.util
        spec=importlib.util.spec_from_file_location('legacy_motion_audit',args.legacy_motion)
        legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
        dataset_class=legacy.MotionDataset
    from hikinglab3.motions import DEFAULT_MOTION
    data=dataset_class(names,q0,'cpu',DEFAULT_MOTION)
    import ast
    from hikinglab3.official_policy import PHYSX_JOINTS
    constants={}
    for node in ast.parse(Path('/home/ye/Desktop/HikingLab3/vendor/InstinctLab/source/instinctlab/instinctlab/assets/unitree_g1.py').read_text()).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id.startswith('G1_29Dof_TorsoBase_symmetric_augmentation_joint_'):
            constants[node.targets[0].id]=ast.literal_eval(node.value)
    mapping=constants['G1_29Dof_TorsoBase_symmetric_augmentation_joint_mapping']
    signs=constants['G1_29Dof_TorsoBase_symmetric_augmentation_joint_reverse_buf']
    for i,name in enumerate(names):
        j=PHYSX_JOINTS.index(name)
        assert names[int(data.mirror_order[i])]==PHYSX_JOINTS[mapping[j]]
        assert float(data.mirror_sign[i])==signs[j]

    raw=np.load(data.manifest['path']); rotations=Rotation.from_quat(raw['base_quat_w'][:,[1,2,3,0]])
    q=raw['joint_pos'][:,[raw['joint_names'].tolist().index(n) for n in names]]
    def diff(x):return np.concatenate((np.diff(x,axis=0)*50,np.zeros_like(x[:1])))
    w=np.concatenate(((rotations[1:]*rotations[:-1].inv()).as_rotvec()*50,np.zeros((1,3))))
    expected={'projected_gravity':rotations.inv().apply(np.tile([0.,0.,-1.],(len(q),1))), 'joint_pos_rel':q-q0.numpy(), 'joint_vel':diff(q)*.05,'base_lin_vel':rotations.inv().apply(diff(raw['base_pos_w'])),'base_ang_vel':rotations.inv().apply(w)}
    math_errors={k:float(np.max(np.abs(data.features[k].numpy()-v))) for k,v in expected.items()}
    first=data.advance(4,0,torch.ones(4,dtype=torch.bool)).clone(); cursor=data._cursor.clone()
    again=data.advance(4,0,torch.ones(4,dtype=torch.bool)).clone()
    same_step={'max_error':float((first-again).abs().max()),'cursors_changed':int((cursor!=data._cursor).sum())}
    data._cursor.fill_(100)
    data.advance(4,1,torch.tensor([True,False,False,False])); before=data._cursor.clone(); hist=data._reference_output.clone()
    repeated=data.advance(4,1,torch.tensor([True,False,False,False]))
    partial={'max_error':float((hist-repeated).abs().max()),'unreset_cursor_delta':(data._cursor-before)[1:].tolist()}
    from instinct_rl.modules.discriminator import Discriminator
    cfg=json.loads((run/'agent.json').read_text())['algorithm']
    disc=Discriminator(json.loads((run/'observation_format.json').read_text())['amp_policy'],**cfg['discriminator_kwargs'])
    disc.load_state_dict(torch.load(run/'model_15000.pt',map_location='cpu',weights_only=False)['discriminator']);disc.eval()
    torch.manual_seed(947)
    samples=data.sample(1000)
    static=[]
    offset=0
    for k,v in data.features.items():
        width=v.shape[-1]
        frame=samples[:,offset:offset+10*width].reshape(-1,10,width)[:,-1].clone()
        offset+=10*width
        if k in ('joint_vel','base_lin_vel','base_ang_vel'):frame.zero_()
        static.append(frame[:,None].expand(-1,10,-1).flatten(1))
    def score(x):
        with torch.inference_mode():
            d=disc(x).flatten();amp=.25*(1-.25*(d-1).square()).clamp(min=0)
        return {'D_mean':float(d.mean()),'amp_per_step_mean':float(amp.mean()),'amp_quantiles':torch.quantile(amp,torch.tensor([0.,.1,.5,.9,1.])).tolist()}
    result={'official_mirror_mapping_matches':True,'independent_feature_max_abs_errors':math_errors,'same_step_all_reset':same_step,'same_step_partial_reset':partial,'reference_windows':score(samples),'synthetic_static_reference_poses':score(torch.cat(static,dim=-1))}
    if args.mesh:
        from isaaclab.terrains import TerrainGenerator
        from hikinglab3.terrain_cfg import ROUGH_TERRAINS_CFG
        from hikinglab3.edges import GreedyconcatEdgeCylinder
        cfg=ROUGH_TERRAINS_CFG.copy();cfg.seed=42;cfg.use_cache=True
        mesh=TerrainGenerator(cfg,device='cpu').terrain_mesh
        clean=mesh.copy();clean.merge_vertices();clean.update_faces(clean.unique_faces());clean.remove_unreferenced_vertices()
        def edge_coords(m):
            pairs=m.face_adjacency_edges[m.face_adjacency_angles>np.deg2rad(70.)]
            return m.vertices[pairs].reshape(-1,6)
        def canonical(edges):
            e=np.round(edges.reshape(-1,2,3),6); rev=np.array([tuple(a)>tuple(b) for a,b in e]);e[rev]=e[rev,::-1]
            return {tuple(x) for x in e.reshape(-1,6)}
        raw_edges=edge_coords(mesh);clean_edges=edge_coords(clean)
        a=canonical(raw_edges);b=canonical(clean_edges)
        ecfg=SimpleNamespace(angle_threshold=70.,adjacent_angle_threshold=30.,min_points=2,cylinder_radius=.05,num_grid_cells=64**3)
        det=GreedyconcatEdgeCylinder(ecfg)
        random.seed(42);pa=det.process_edges(raw_edges)
        random.seed(42);pb=det.process_edges(clean_edges)
        ca=canonical(pa);cb=canonical(pb)
        result['mesh']={'raw_vertices':len(mesh.vertices),'clean_vertices':len(clean.vertices),'raw_faces':len(mesh.faces),'clean_faces':len(clean.faces),'raw_sharp_edges':len(a),'clean_sharp_edges':len(b),'sharp_added':len(b-a),'sharp_removed':len(a-b),'raw_processed_edges':len(pa),'clean_processed_edges':len(pb),'processed_added':len(cb-ca),'processed_removed':len(ca-cb)}
        np.savez_compressed(args.output/'edge_sets.npz',raw=pa,clean=pb)
    (args.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
