"""Convert the original G1 URDF using the standalone Newton USD converter."""

from pathlib import Path
from urdf_usd_converter import Converter
from pxr import Usd, UsdPhysics, UsdGeom

root = Path(__file__).resolve().parents[1]
urdf = (
    root
    / "vendor/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf"
)
asset = Converter(layer_structure=False, scene=False).convert(str(urdf), str(root / "assets/g1"))
stage = Usd.Stage.Open(asset.path)
# Flatten rigid-body namespace while retaining world transforms and relationship targets.
cache = UsdGeom.XformCache()
bodies = [
    (str(p.GetPath()), p.GetName(), cache.GetLocalToWorldTransform(p))
    for p in stage.Traverse()
    if p.HasAPI(UsdPhysics.RigidBodyAPI)
]
for old, name, world in sorted(bodies, key=lambda item: item[0].count("/"), reverse=True):
    new = "/g1/" + name
    editor = Usd.NamespaceEditor(stage)
    editor.MovePrimAtPath(old, new)
    if not editor.ApplyEdits():
        raise RuntimeError(f"Cannot move {old}")
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(new))
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(world)
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.RevoluteJoint):
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(0.0)
# The low-level converter anchors the root to /g1 (a non-rigid world scope).
# Remove that anchor explicitly for locomotion; an empty body0 check alone misses it.
for prim in list(stage.Traverse()):
    if prim.IsA(UsdPhysics.FixedJoint):
        j = UsdPhysics.Joint(prim)
        parents = j.GetBody0Rel().GetTargets()
        children = j.GetBody1Rel().GetTargets()
        if (
            children
            and str(children[0]) == "/g1/torso_link"
            and (not parents or not stage.GetPrimAtPath(parents[0]).HasAPI(UsdPhysics.RigidBodyAPI))
        ):
            stage.RemovePrim(prim.GetPath())
stage.GetRootLayer().Save()
(root / "assets/robot_path.txt").write_text(str(Path(asset.path).resolve()))
print("Converted", asset.path)
