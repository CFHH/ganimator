"""
This code is modified from:
http://theorangeduck.com/page/deep-learning-framework-character-motion-synthesis-and-editing

by Daniel Holden et al
"""


import re
import numpy as np
from bvh.Quaternions import Quaternions

channelmap = {
    'Xrotation' : 'x',
    'Yrotation' : 'y',
    'Zrotation' : 'z'   
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x': 0,
    'y': 1,
    'z': 2,
}


class Animation:
    def __init__(self, rotations, positions, orients, offsets, parents, names, frametime):
        self.rotations = rotations
        self.positions = positions
        self.orients   = orients
        self.offsets   = offsets
        self.parent    = parents
        self.names     = names
        self.frametime = frametime

    @property
    def shape(self):
        return self.rotations.shape


def load(filename, start=None, end=None, order=None, world=False, need_quater=False) -> Animation:
    """
    Reads a BVH file and constructs an animation

    Parameters
    ----------
    filename: str
        File to be opened

    start : int
        Optional Starting Frame, [start, end)

    end : int
        Optional Ending Frame, [start, end)

    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'

    world : bool
        If set to true euler angles are applied
        together in world space rather than local
        space
    Returns
    -------

    (animation, joint_names, frametime)
        Tuple of loaded animation and joint names
    """

    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    orients = Quaternions.id(0)
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)
    orders = []

    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        """ Modified line read to handle mixamo data """
        #        rmatch = re.match(r"ROOT (\w+)", line)
        rmatch = re.match(r"ROOT (\w+:?\w+)", line)
        if rmatch:
            names.append(rmatch.group(1)) # 根骨骼名字，比如mixamorig7:Hips。Joe的最终数量是65
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0) # 先随便设个0，后面会读取OFFSET修改
            orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0) # Quaternions是四元数数组，qs是个二维数组
            parents = np.append(parents, active) # 根骨骼没有父节点，-1
            active = (len(parents) - 1) # 0
            continue

        if "{" in line: continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))]) # 修改当前active骨骼的offset，之前是000
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))

            channelis = 0 if channels == 3 else 3
            channelie = 3 if channels == 3 else 6
            parts = line.split()[2 + channelis:2 + channelie] # ['Xrotation', 'Yrotation', 'Zrotation']
            if any([p not in channelmap for p in parts]):
                continue
            order = "".join([channelmap[p] for p in parts]) # 'xyz'
            orders.append(order) # 一堆'xyz'
            continue

        """ Modified line read to handle mixamo data """
        #        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        jmatch = re.match("\s*JOINT\s+(\w+:?\w+)", line)
        if jmatch:
            names.append(jmatch.group(1)) # 根骨骼名字，比如mixamorig7:Spine
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0) # 先随便设个0，后面会读取OFFSET修改
            orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active) # 父节点索引
            # orders是例外，不是先随便append一个，然后再修改
            active = (len(parents) - 1) # 当前节点
            continue

        if "End Site" in line:
            # End Site好像没什么用
            # 配置中下面的OFFSET是什么意思？没什么用
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            fnum = int(fmatch.group(1))
            if start and end:
                if start <= 0:
                    start = 0
                if end <= 0 or end > fnum:
                    end = fnum
                fnum = (end - start)  # [start, end)
                assert fnum > 0
            jnum = len(parents)
            # offsets是(65, 3), positions是(fnum, 65, 3)
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            # orients含65个，rotations是(fnum, 65, 3)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        # 接下来是取各帧的数据，i原始为0
        if (start and end) and (i < start or i >= end):
            i += 1
            continue

        # dmatch = line.strip().split(' ')
        dmatch = line.strip().split() #65骨骼的Joe，一行是198个数，198 = 3 + 65 * 3，先根骨骼偏移，再各骨骼旋转（3个欧拉角，单位是角度！！！！！！！！！！）
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i # fi = i
            if channels == 3: # 这里的channels，使用了骨骼解析过程中的最后一次赋值，原则上不对
                positions[fi, 0:1] = data_block[0:3]            # 修改第fi帧的根骨骼偏移，其他骨骼保持不变
                rotations[fi, :] = data_block[3:].reshape(N, 3) # 修改第fi帧的各骨骼旋转
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    # 按照xyz来重新组织rotations是，如果本来就是xyz就不用处理了
    all_rotations = []
    canonical_order = 'xyz'
    for i, order in enumerate(orders):
        rot = rotations[:, i:i + 1] # rotations是(fnum, 65, 3)，rot是(fnum, 1, 3)，第i根骨骼在各帧的旋转
        if need_quater:
            quat = Quaternions.from_euler(np.radians(rot), order=order, world=world) # 此处说明旋转是用欧拉角表示的，而且是角度
            all_rotations.append(quat)
            continue
        elif order != canonical_order:
            quat = Quaternions.from_euler(np.radians(rot), order=order, world=world)
            rot = np.degrees(quat.euler(order=canonical_order))
        all_rotations.append(rot)
    # all_rotations是65个(fnum, 1, 3)，再变回(fnum, 65, 3)
    rotations = np.concatenate(all_rotations, axis=1)

    return Animation(rotations, positions, orients, offsets, parents, names, frametime)


def get_frame_info(filename):
    frame_num = 0
    frame_time = 0.0
    f = open(filename, "r")
    cur_step = 0
    for line in f:
        if cur_step == 0:
            if "MOTION" in line:
                cur_step = 1
                continue
        if cur_step == 1:
            cur_step = 2
            fmatch = re.match("\s*Frames:\s+(\d+)", line)
            frame_num = int(fmatch.group(1))
            continue
        if cur_step == 2:
            fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
            frame_time = float(fmatch.group(1))
            break
    return frame_num, frame_time


def save(filename, anim, names=None, frametime=1.0/24.0, order='zyx', positions=False, orients=True):
    """
    Saves an Animation to file as BVH
    
    Parameters
    ----------
    filename: str
        File to be saved to
        
    anim : Animation
        Animation to save
        
    names : [str]
        List of joint names
    
    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'
    
    frametime : float
        Optional Animation Frame time
        
    positions : bool
        Optional specfier to save bone
        positions for each frame
        
    orients : bool
        Multiply joint orients to the rotations
        before saving.
        
    """
    
    if names is None:
        names = ["joint_" + str(i) for i in range(len(anim.parents))]
    
    with open(filename, 'w') as f:

        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, names[0]))
        f.write("%s{\n" % t)
        t += '\t'

        f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[0,0], anim.offsets[0,1], anim.offsets[0,2]) )
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % 
            (t, channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))

        for i in range(anim.shape[1]):
            if anim.parents[i] == 0:
                t = save_joint(f, anim, names, t, i, order=order, positions=positions)

        t = t[:-1]
        f.write("%s}\n" % t)

        f.write("MOTION\n")
        f.write("Frames: %i\n" % anim.shape[0]);
        f.write("Frame Time: %f\n" % frametime);
            
        #if orients:        
        #    rots = np.degrees((-anim.orients[np.newaxis] * anim.rotations).euler(order=order[::-1]))
        #else:
        #    rots = np.degrees(anim.rotations.euler(order=order[::-1]))
        rots = np.degrees(anim.rotations.euler(order=order[::-1]))
        poss = anim.positions
        
        for i in range(anim.shape[0]):
            for j in range(anim.shape[1]):
                
                if positions or j == 0:
                
                    f.write("%f %f %f %f %f %f " % (
                        poss[i,j,0],                  poss[i,j,1],                  poss[i,j,2], 
                        rots[i,j,ordermap[order[0]]], rots[i,j,ordermap[order[1]]], rots[i,j,ordermap[order[2]]]))
                
                else:
                    
                    f.write("%f %f %f " % (
                        rots[i,j,ordermap[order[0]]], rots[i,j,ordermap[order[1]]], rots[i,j,ordermap[order[2]]]))

            f.write("\n")
    
    
def save_joint(f, anim, names, t, i, order='zyx', positions=False):
    
    f.write("%sJOINT %s\n" % (t, names[i]))
    f.write("%s{\n" % t)
    t += '\t'
  
    f.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[i,0], anim.offsets[i,1], anim.offsets[i,2]))
    
    if positions:
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % (t, 
            channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
    else:
        f.write("%sCHANNELS 3 %s %s %s\n" % (t, 
            channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
    
    end_site = True
    
    for j in range(anim.shape[1]):
        if anim.parents[j] == i:
            t = save_joint(f, anim, names, t, j, order=order, positions=positions)
            end_site = False
    
    if end_site:
        f.write("%sEnd Site\n" % t)
        f.write("%s{\n" % t)
        t += '\t'
        f.write("%sOFFSET %f %f %f\n" % (t, 0.0, 0.0, 0.0))
        t = t[:-1]
        f.write("%s}\n" % t)
  
    t = t[:-1]
    f.write("%s}\n" % t)
    
    return t
