import bpy
import sys

# 使用方法：把bvh集中在一个目录里，把bvh2fbx.bat放进去，执行
# FOR %%f IN (*.bvh) DO "H:\Program Files\Blender Foundation\Blender 3.2\blender.exe" -b --python "I:\ganimator\把bvh2fbx.py" -- "%%f"

# https://alastaira.wordpress.com/2014/04/25/batch-conversion-of-bvh-to-fbx-motion-capture-files/
# https://docs.blender.org/api/current/index.html

# Get command line arguments
argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "—"
bvh_in = argv[0]
fbx_out = bvh_in.split(".bvh")[0]+".fbx"

# Import the BVH file
# https://docs.blender.org/api/current/bpy.ops.import_anim.html
# See http://www.blender.org/documentation/blender_python_api_2_60_0/bpy.ops.import_anim.html
bpy.ops.import_anim.bvh(filepath=bvh_in, filter_glob="*.bvh", global_scale=1, frame_start=1, use_fps_scale=False,
                        use_cyclic=False, rotate_mode='NATIVE', axis_forward='-Z', axis_up='Y')

# Export as FBX
# https://docs.blender.org/api/current/bpy.ops.export_scene.html  删去了use_anim=True, use_default_take=False
# See http://www.blender.org/documentation/blender_python_api_2_62_1/bpy.ops.export_scene.html
bpy.ops.export_scene.fbx(filepath=fbx_out, axis_forward='-Z', axis_up='Y', use_selection=True)
