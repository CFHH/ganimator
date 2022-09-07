import os
from os.path import join as pjoin
import numpy as np
import torch
import torch.nn.functional as F
from bvh.bvh_parser import BVH_file
from models.transforms import interpolate_6d
from models.utils import gaussian_filter_wrapper
from bvh.bvh_writer import WriterWrapper


class MotionData:
    def __init__(self, filename, repr='quat', padding=False, #'repr6d', True,
                 use_velo=False, no_scale=False, contact=False, keep_y_pos=False, #True, False, True, True,
                 joint_reduction=True, start_frame=None, end_frame=None): #True
        if not (start_frame is None) and not (end_frame is None):
            print("MotionData, loading %s, [%d, %d), contact=%d" % (filename, start_frame, end_frame, contact))
        else:
            print("MotionData, loading %s, contact=%d" % (filename, contact))
        self.bvh_file = BVH_file(filename, no_scale, requires_contact=contact,
                                 joint_reduction=joint_reduction, start_frame=start_frame, end_frame=end_frame)
        self.contact = contact
        self.filename = filename
        self.raw_motion = self.bvh_file.to_tensor(repr=repr) #(frame, 171)训练用数据，171 = (24骨骼 + 4贴地额外) * 6 + 3，位置在后

        # ['euler', 'quat', 'quaternion', 'repr6d', 'mat']
        if repr == 'repr6d':
            self.n_rot = 6
        elif repr == 'euler':
            self.n_rot = 3
        elif repr == 'mat':
            self.n_rot = 9
        else:
            self.n_rot = 4

        self.repr = repr
        self.use_velo = use_velo
        self.keep_y_pos = keep_y_pos

        self.writer = self.bvh_file.writer
        self.raw_motion = self.raw_motion.permute(1, 0) # 改变维度，(frame, 171)->(171, frame)，现在每列是一帧的数据。一维卷积是在最后维度上扫的
        self.raw_motion.unsqueeze_(0)     # Shape = (1, n_channel, n_frames), (171, frame) -> (1, 171, frame), batch
        if self.use_velo:
            self.velo_mask = [-3, -2, -1] if not keep_y_pos else [-3, -1] # y位置不动
            # 第2个维度的最后3个数字是位置， 所以-1是最后一个是z，-3是倒数第3个是x，这里后一帧减前一帧
            self.raw_motion[:, self.velo_mask, 1:] = self.raw_motion[:, self.velo_mask, 1:] - \
                                                     self.raw_motion[:, self.velo_mask, :-1]
            self.begin_pos = self.raw_motion[:, self.velo_mask, 0].clone() #保存初始位置(1,2)
            self.raw_motion[:, self.velo_mask, 0] = self.raw_motion[:, self.velo_mask, 1]
            # 到此，除了第0帧，其他各帧的xz表示的是相对前一帧的偏移；第0帧，使用第1帧的数据，以免解读为偏移时数值过大


        if padding:
            self.n_pad = self.n_rot - 3 # pad position channels
            paddings = torch.zeros_like(self.raw_motion[:, :self.n_pad]) #(1,3,frame)
            self.raw_motion = torch.cat((self.raw_motion, paddings), dim=1) #(1, 171, frame)->(1, 174, frame)，添3个0
        else:
            self.n_pad = 0

        self.raw_motion = torch.cat((self.raw_motion[:, :-3-self.n_pad], self.raw_motion[:, -3-self.n_pad:]), dim=1) #这操作没任何效果
        #最终，(1, 174, frame)

        if contact:
            self.n_contact = len(self.bvh_file.skeleton.contact_id)
        else:
            self.n_contact = 0

    def slerp(self, input, size):
        res = torch.empty(input.shape[:-1] + (size,), device=input.device, dtype=input.dtype)
        n_other = 3 + self.n_pad + self.n_contact * 6
        rot = input[..., :-n_other, :]
        other = input[..., -n_other:, :]

        rot_res = interpolate_6d(rot, size)
        other_res = F.interpolate(other, size=size, mode='linear', align_corners=True)
        res[..., :-n_other, :] = rot_res
        res[..., -n_other:, :] = other_res
        return res

    def sample(self, size=None, slerp=False, input=None):
        raw_motion = self.raw_motion #(1, 174, frame)
        if input is not None:
            raw_motion = input
        if size is None:
            return raw_motion
        else:
            if slerp:
                motion = self.slerp(raw_motion, size=size)
            else:
                motion = F.interpolate(raw_motion, size=size, mode='linear', align_corners=False) #(1, 174, size)
            return motion

    @property
    def n_channels(self):
        return self.raw_motion.shape[1]

    def velo2pos(self, motion):
        if not self.use_velo:
            return motion
            # raise Exception('No need to call this function')
        res = motion.clone()
        mask = [i - self.n_pad for i in self.velo_mask]
        res[:, mask, 0] = self.begin_pos.to(motion.device)
        res[:, mask] = torch.cumsum(res[:, mask], dim=-1)
        return res

    def inject_contact(self, source, target):
        batch_size = source.shape[0]
        temporal_size = source.shape[-1]
        source = source.reshape(batch_size, -1, self.n_rot, temporal_size)
        target = target.clone().reshape(batch_size, -1, self.n_rot, temporal_size)
        target[:, -(self.n_contact+1):-1] = source[:, -(self.n_contact+1):-1]
        target = target.reshape(batch_size, -1, temporal_size)
        return target

    def parse(self, motion, keep_velo=False,):
        """
        ZZW TODO BATCH
        No batch support here!!!
        :returns pos, rot, contact (if exists)
        """
        motion = motion.clone()

        if self.n_pad:
            motion = motion[:, :-self.n_pad]
        if self.use_velo and not keep_velo:
            motion[:, self.velo_mask, 0] = self.begin_pos.to(motion.device)
            motion[:, self.velo_mask] = torch.cumsum(motion[:, self.velo_mask], dim=-1)
        motion = motion.squeeze().permute(1, 0)
        pos = motion[..., -3:]
        rot = motion[..., :-3].reshape(motion.shape[0], -1, self.n_rot)
        if self.contact:
            contact = rot[..., -self.n_contact:, 0]
            rot = rot[..., :-self.n_contact, :]
        else:
            contact = None
        return pos, rot, contact

    def write(self, filename, motion, scale100=False, fix_euler=False):
        pos, rot, contact = self.parse(motion)
        if self.contact:
            np.save(filename + '.contact', contact.detach().cpu().numpy())
        self.writer.write(filename, rot, pos, names=self.bvh_file.skeleton.names, repr=self.repr, frametime=self.bvh_file.frametime, scale100=scale100, fix_euler=fix_euler)

    def __len__(self):
        return self.raw_motion.shape[-1]


def load_multiple_dataset(prefix, name_list, **kargs):
    with open(name_list, 'r') as f:
        names = [line.strip() for line in f.readlines()]
    datasets = []
    for f in names:
        kargs['filename'] = pjoin(prefix, f)
        print("load_multiple_dataset, %s" % kargs['filename'])
        datasets.append(MotionData(**kargs))
    return datasets

def load_slice_dataset(start_frame, end_frame, slice_frame_num, **kargs):
    slice_num = int((end_frame - start_frame) / slice_frame_num)
    assert slice_num > 0
    datasets = []
    for i in range(slice_num):
        start = start_frame + i * slice_frame_num
        end = start + slice_frame_num
        kargs['start_frame'] = start
        kargs['end_frame'] = end
        #print("load_slice_dataset, %s, slice[%d], [%d, 5d)" % (kargs['filename'], i, start, end))
        datasets.append(MotionData(**kargs))
    return datasets
