import bpy
import sys
import numpy as np
from os import listdir, path

# 使用方法：把fbx集中在一个目录里，把fbx2bvh.bat放进去，执行
# FOR %%f IN (*.bvh) DO "H:\Program Files\Blender Foundation\Blender 3.2\blender.exe" -b --python "I:\ganimator\把fbx2bvh.py" -- "%%f"

# 使用方法：blender -b -P fbx2bvh.py
# https://github.com/DeepMotionEditing/deep-motion-editing/issues/25

def fbx2bvh(fbx_in, bvh_out):
    # 这里的axis_forward和axis_up，填什么对bvh的结果无影响
    bpy.ops.import_scene.fbx(filepath=fbx_in, use_anim=True, axis_forward='Z', axis_up='Y')

    frame_start = 9999
    frame_end = -9999
    action = bpy.data.actions[-1]
    if action.frame_range[1] > frame_end:
      frame_end = action.frame_range[1]
    if action.frame_range[0] < frame_start:
      frame_start = action.frame_range[0]

    #frame_end = np.max([60, frame_end])
    bpy.ops.export_anim.bvh(filepath=bvh_out, frame_start=int(frame_start), frame_end=int(frame_end),
                            rotate_mode='XYZ',
                            root_transform_only=False)
    #bpy.data.actions.remove(bpy.data.actions[-1])


# Get command line arguments
argv = sys.argv
argv = argv[argv.index("--") + 1:]
fbx_in = argv[0]
bvh_out = fbx_in.split(".fbx")[0]+".bvh"

fbx2bvh(fbx_in, bvh_out)