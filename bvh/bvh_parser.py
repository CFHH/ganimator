import torch
import bvh.bvh_io as bvh_io
import numpy as np
from bvh.Quaternions import Quaternions
from bvh.skeleton_database import SkeletonDatabase
from models.kinematics import ForwardKinematicsJoint
from models.transforms import quat2repr6d, quat2mat, degeuler2mat
from models.contact import foot_contact
from bvh.bvh_writer import WriterWrapper


class Skeleton:
    def __init__(self, names, parent, offsets, joint_reduction=True):
        self._names = names
        self.original_parent = parent
        self._offsets = offsets
        self._parent = None
        self._ee_id = None
        self.contact_names = []

        for i, name in enumerate(self._names):
            if ':' in name:
                self._names[i] = name[name.find(':')+1:] # mixamorig7:Hips -> Hips

        if joint_reduction:
            self.skeleton_type, match_num = SkeletonDatabase.match(names)
            corps_names = SkeletonDatabase.corps_names[self.skeleton_type] # 这个只有24根骨骼，比names少了很多不重要的
            self.contact_names = SkeletonDatabase.contact_names[self.skeleton_type] # 检测是否贴地的骨骼
            self.contact_threshold = SkeletonDatabase.contact_thresholds[self.skeleton_type]

            self.contact_id = []
            for i in self.contact_names:
                self.contact_id.append(corps_names.index(i))
        else:
            self.skeleton_type = -1
            corps_names = self._names

        self.details = []    # joints that does not belong to the corps (we are not interested in them)
        for i, name in enumerate(self._names):
            if name not in corps_names: self.details.append(i)

        self.corps = [] # SkeletonDatabase中的骨骼顺序在训练数据骨骼中的索引
        self.simplified_name = [] # 最后就是SkeletonDatabase中的顺序
        self.simplify_map = {} # 如55->1，55是训练数据骨骼的索引，1是SkeletonDatabase中的索引
        self.inverse_simplify_map = {} # 如1->55

        # Repermute the skeleton id according to the databse
        for name in corps_names:
            for j in range(len(self._names)):
                if name in self._names[j]:
                    self.corps.append(j)
                    break
        if len(self.corps) != len(corps_names):
            for i in self.corps:
                print(self._names[i], end=' ')
            print(self.corps, self.skeleton_type, len(self.corps), sep='\n')
            raise Exception('Problem in this skeleton')

        self.joint_num_simplify = len(self.corps)
        for i, j in enumerate(self.corps):
            self.simplify_map[j] = i
            self.inverse_simplify_map[i] = j
            self.simplified_name.append(self._names[j])
        self.inverse_simplify_map[0] = -1 # 把 0->0 改成0->-1，没用。ForwardKinematicsJoint.forward()
        for i in range(len(self._names)):
            if i in self.details:
                self.simplify_map[i] = -1

    @property
    def parent(self):
        if self._parent is None:
            self._parent = self.original_parent[self.corps].copy() # self.original_parent[self.corps] 重组顺序
            for i in range(self._parent.shape[0]):
                if i >= 1:
                    self._parent[i] = self.simplify_map[self._parent[i]] #最终是db骨骼系统中的父节点索引
            self._parent = tuple(self._parent)
        return self._parent

    @property
    def offsets(self): # 相当于根据SkeletonDatabase中的骨骼顺序重新组织了_offsets
        return torch.tensor(self._offsets[self.corps], dtype=torch.float)

    @property
    def names(self):
        return self.simplified_name

    @property
    def ee_id(self):
        raise Exception('Abaddoned')
        # if self._ee_id is None:
        #     self._ee_id = []
        #     for i in SkeletonDatabase.ee_names[self.skeleton_type]:
        #         self.ee_id._ee_id(corps_names[self.skeleton_type].index(i))


class BVH_file:
    def __init__(self, file_path, no_scale=False, requires_contact=False, joint_reduction=True, start_frame=None, end_frame=None):
        self.anim = bvh_io.load(file_path, start=start_frame, end=end_frame)
        self._names = self.anim.names
        self.frametime = self.anim.frametime
        self.skeleton = Skeleton(self.anim.names, self.anim.parent, self.anim.offsets, joint_reduction)

        # Downsample to 30 fps for our application
        if self.frametime < 0.0084:
            self.frametime *= 2
            self.anim.positions = self.anim.positions[::2]
            self.anim.rotations = self.anim.rotations[::2]
        if self.frametime < 0.017:
            self.frametime *= 2
            self.anim.positions = self.anim.positions[::2]
            self.anim.rotations = self.anim.rotations[::2]

        # Scale by 1/100 if it's raw exported bvh from blender
        if not no_scale and self.skeleton.offsets[0, 1] > 10: # [0,1]也就是根骨骼的Y偏移，99.79
            self.scale(1. / 100)

        self.requires_contact = requires_contact

        if requires_contact:
            self.contact_names = self.skeleton.contact_names
        else:
            self.contact_names = []

        self.fk = ForwardKinematicsJoint(self.skeleton.parent, self.skeleton.offsets)
        self.writer = WriterWrapper(self.skeleton.parent, self.skeleton.offsets)
        if self.requires_contact:
            gl_pos = self.joint_position()
            self.contact_label = foot_contact(gl_pos[:, self.skeleton.contact_id],
                                              threshold=self.skeleton.contact_threshold)
            self.gl_pos = gl_pos

    def local_pos(self):
        gl_pos = self.joint_position()
        local_pos = gl_pos - gl_pos[:, 0:1, :]
        return local_pos[:, 1:]

    def scale(self, ratio):
        self.anim.offsets *= ratio
        self.anim.positions *= ratio

    def to_tensor(self, repr='euler', rot_only=False):
        if repr not in ['euler', 'quat', 'quaternion', 'repr6d', 'mat']:
            raise Exception('Unknown rotation representation')
        positions = self.get_position() # 根骨骼偏移(frame, 3)
        rotations = self.get_rotation(repr=repr) # 各骨骼旋转(frame, 24, 6)

        if rot_only:
            return rotations.reshape(rotations.shape[0], -1)

        if self.requires_contact:
            virtual_contact = torch.zeros_like(rotations[:, :len(self.skeleton.contact_id)]) # (frame, 4, 6)，rotations的第二维是24，这里取前4
            virtual_contact[..., 0] = self.contact_label # self.contact_label是(frame, 4)，各帧的贴地数据。每帧数据是4行6列，这里把第一列的4个数换成了各帧的贴地数据
            rotations = torch.cat([rotations, virtual_contact], dim=1)  # (frame, 28, 6)，原来一帧数据是24行6列，现在增加的4行，这4行的第一列是贴地数据，其他全是0

        rotations = rotations.reshape(rotations.shape[0], -1) # (frame, 28*6)，6个6个排起来的
        return torch.cat((rotations, positions), dim=-1) # (frame, 168) + (frame, 3) = (frame, 171)，旋转在前，位置在后

    def joint_position(self):
        positions = torch.tensor(self.anim.positions[:, 0, :], dtype=torch.float) # 取根骨骼，(frame=648, 3)
        rotations = self.anim.rotations[:, self.skeleton.corps, :] # (frame, 24, 3)
        rotations = Quaternions.from_euler(np.radians(rotations)).qs # (frame, 24, 4)
        rotations = torch.tensor(rotations, dtype=torch.float)
        j_loc = self.fk.forward(rotations, positions) # 计算各骨骼节点的世界坐标位置(frame, 24, 3)
        return j_loc

    def get_rotation(self, repr='quat'):
        rotations = self.anim.rotations[:, self.skeleton.corps, :]
        if repr == 'quaternion' or repr == 'quat' or repr == 'repr6d':
            rotations = Quaternions.from_euler(np.radians(rotations)).qs
            rotations = torch.tensor(rotations, dtype=torch.float)
        if repr == 'repr6d':
            rotations = quat2repr6d(rotations) # (frame, 24, 6)
        if repr == 'euler':
            rotations = torch.tensor(rotations, dtype=torch.float)
        if repr == 'mat':
            #ZZW TODO rotations.shape=(帧数，骨骼数24，3)单位是角度，处理后shape=(帧数，骨骼数，9)还需要转成tenor
            rotations = degeuler2mat(rotations)
            rotations = torch.tensor(rotations, dtype=torch.float)
        return rotations

    def get_position(self):
        return torch.tensor(self.anim.positions[:, 0, :], dtype=torch.float)

    def dfs(self, x, vis, dist):
        fa = self.skeleton.parent
        vis[x] = 1
        for y in range(len(fa)):
            if (fa[y] == x or fa[x] == y) and vis[y] == 0:
                dist[y] = dist[x] + 1
                self.dfs(y, vis, dist)

    def get_neighbor(self, threshold, enforce_contact=False):
        fa = self.skeleton.parent
        neighbor_list = []
        for x in range(0, len(fa)):
            vis = [0 for _ in range(len(fa))] #在dfs()里用来作为是否该位置已经计算的标志（is visited），0：未计算
            dist = [0 for _ in range(len(fa))] #x节点到其他节点的距离
            self.dfs(x, vis, dist)
            neighbor = []
            for j in range(0, len(fa)):
                if dist[j] <= threshold: #默认参数是2，小于等于2认为相邻
                    neighbor.append(j)
            neighbor_list.append(neighbor)

        # 在数据中，对贴地检测的骨骼节点（比如4个），每个附加了6（取决于对节点旋转的表示方法）个数字，所以当作24+4=28的骨骼系统
        # neighbor_list只有24，所以下面再加4个，同时自己跟自己相邻
        contact_list = []
        if self.requires_contact:
            for i, p_id in enumerate(self.skeleton.contact_id): # p_id是贴地检测骨骼节点在骨骼系统中的序号
                v_id = len(neighbor_list) # 虚拟的骨骼序号，也代表接下来的重复数据在neighbor_list中的序号
                neighbor_list[p_id].append(v_id) # 自己重复一遍，自己跟自己相邻
                neighbor_list.append(neighbor_list[p_id]) # 从24扩展到28
                contact_list.append(v_id) # 记录重复的

        # 根骨骼节点也重复一遍
        root_neighbor = neighbor_list[0]
        id_root = len(neighbor_list) # 代表接下来的重复数据在neighbor_list中的序号

        if enforce_contact: # 默认为True
            root_neighbor = root_neighbor + contact_list # 两个list合并
            for j in contact_list:
                neighbor_list[j] = list(set(neighbor_list[j])) # 没实际效果，除了顺序有可能变化

        root_neighbor = list(set(root_neighbor)) #没实际效果
        for j in root_neighbor:
            neighbor_list[j].append(id_root) # 互为邻居，一个重复了一份，它的邻居都得添加
        root_neighbor.append(id_root)
        neighbor_list.append(root_neighbor)  # Neighbor for root position，已经存在，再添加一遍
        return neighbor_list
