import os
import sys
import torch
from dataset.motion import MotionData, load_multiple_dataset, load_slice_dataset
from models import create_model, create_layered_model, get_group_list
from models.architecture import get_pyramid_lengths, joint_train
from models.utils import get_interpolator
from option import TrainOptionParser
from os.path import join as pjoin
import time
from torch.utils.tensorboard import SummaryWriter
from loss_recorder import LossRecorder
from demo import load_all_from_path
from utils import get_device_info
from bvh.bvh_io import get_frame_info

#
# 训练单个动作
# python train.py --bvh_prefix=./data/Joe --bvh_name=Salsa-Dancing-6 --save_path=./results/Joe --device=cuda:0
# python demo.py --save_path=./results/Joe
#
# python train.py --bvh_prefix=./data/aist --bvh_name=gWA_sFM_cAll_d25_mWA4_ch05 --save_path=./results/gWA_sFM_cAll_d25_mWA4_ch05 --device=cuda:0
# python demo.py --save_path=./results/gWA_sFM_cAll_d25_mWA4_ch05 --target_length=960
#
# gBR_sBM_cAll_d04_mBR0_ch01
# nohup python CMD > logtrain.log 2>&1 &
# 不需要contact添加参数 --contact=0 --enforce_contact=0
# 旋转表达 --repr=mat
# 切割 --slice=1 --slice_time_len=2.0
#
# python train.py --bvh_prefix=./data/aist --bvh_name=gWA_sFM_cAll_d26_mWA4_ch12 --save_path=./results/gWA_sFM_cAll_d26_mWA4_ch12 --device=cuda:0
# python demo.py --save_path=./results/gWA_sFM_cAll_d26_mWA4_ch12 --target_length=960
#
# nohup python train.py --bvh_prefix=./data/aist --bvh_name=gWA_sFM_cAll_d27_mWA4_ch19 --save_path=./results/gWA_sFM_cAll_d27_mWA4_ch19 --device=cuda:0 > logtrain.log 2>&1 &
# python demo.py --save_path=./results/gWA_sFM_cAll_d27_mWA4_ch19 --target_length=960
#
#
#
# 训练多个动作
# python train.py --bvh_prefix=./data/aist --bvh_name=list_1.txt --save_path=./results/list_1 --device=cuda:0 --multiple_sequences=1
# python demo.py --save_path=./results/list_1
# python train.py --bvh_prefix=./data/aist_noscale --bvh_name=list_1.txt --save_path=./results/list_1_noscale --device=cuda:0 --multiple_sequences=1
# python demo.py --save_path=./results/list_1_noscale
#
# list_1.txt
#gWA_sFM_cAll_d25_mWA4_ch05.bvh
#gWA_sFM_cAll_d26_mWA4_ch12.bvh
#gWA_sFM_cAll_d27_mWA4_ch19.bvh
#
#
# 一些资料
# https://smpl.is.tue.mpg.de/index.html
# https://blog.csdn.net/weixin_43955293/article/details/124670725
# https://chowdera.com/2022/131/202205110634189207.html
# https://github.com/facebookresearch/fairmotion
#

def main():
    start_time = time.time()

    parser = TrainOptionParser()
    args = parser.parse_args()
    device = torch.device(args.device)

    cpu_str, gpu_str = get_device_info()
    print(f'CPU :{cpu_str}\nGPU: {gpu_str}')

    parser.save(pjoin(args.save_path, 'args.txt'))
    os.makedirs(args.save_path, exist_ok=True)

    if not args.multiple_sequences: # 是否多个动作，默认是0，针对一个动作文件进行训练
        file_name = pjoin(args.bvh_prefix, f'{args.bvh_name}.bvh')
        if args.slice: # 把一个动作切成多个动作来进行训练
            frame_num, frame_time = get_frame_info(file_name)
            slice_frame_num = round(args.slice_time_len / frame_time)
            start_frame = 0 if args.start_frame <= 0 else args.start_frame
            end_frame = frame_num if (args.end_frame < 0 or args.end_frame > frame_num) else args.end_frame
            assert end_frame > start_frame
            multiple_data = load_slice_dataset(start_frame, end_frame, slice_frame_num,
                                               filename = file_name,
                                               padding=args.skeleton_aware, use_velo=args.use_velo, repr=args.repr,
                                               contact=args.contact, keep_y_pos=args.keep_y_pos,
                                               joint_reduction=args.joint_reduction)
            motion_data = multiple_data[0]
        else:
            motion_data = MotionData(file_name,
                                     padding=args.skeleton_aware, use_velo=args.use_velo, repr=args.repr, #1, 1, 'repr6d',
                                     contact=args.contact, keep_y_pos=args.keep_y_pos, #1, 1,
                                     joint_reduction=args.joint_reduction) #1
            multiple_data = [motion_data]
    else: # 训练多个动作
        multiple_data = load_multiple_dataset(prefix=args.bvh_prefix, name_list=pjoin(args.bvh_prefix, args.bvh_name),
                                              padding=args.skeleton_aware, use_velo=args.use_velo, repr=args.repr,
                                              contact=args.contact, keep_y_pos=args.keep_y_pos,
                                              joint_reduction=args.joint_reduction)
        motion_data = multiple_data[0]

    # motion_data.shape = (1, channels, frame)
    # channels = (24骨骼节点 + 4贴地标志) * 6（旋转） + 3（唯一，相对上一帧的偏移）+ 3（全0，pad） = 174
    # 贴地标志，6个数字，第一个数字用1表示贴着地，其他是0
    # 所谓贴地，就是先算出各帧各节点的世界坐标，后一帧减前一帧的向量的模若小于某阈值，则是1
    # 174 = 29*6，29 = 24骨骼 + 4贴地骨骼 + 1根骨骼，这是训练模型中29的缘由

    interpolator = get_interpolator(args) #是个插值(采样)函数，默认args.nearest_interpolation=0

    lengths = []
    min_len = 10000
    for i in range(len(multiple_data)):
        # 对每份数据，算得帧数序列，如648 => [129, 172, 229, 305, 406, 541, 648]
        new_length = get_pyramid_lengths(args, len(multiple_data[i]))
        min_len = min(min_len, len(new_length))
        if args.num_stages_limit != -1: #默认-1
            new_length = new_length[:args.num_stages_limit]
        lengths.append(new_length)

    for i in range(len(multiple_data)):
        lengths[i] = lengths[i][-min_len:] #让所有数据的帧数序列的长度一样，比如都是7个；对应于相同的序号，对原数据的缩放是差不多的

    if not args.silent:
        print('Levels:', lengths)

    log_path = pjoin(args.save_path, './logs')
    if os.path.exists(log_path):
        os.system(f'rm -r {log_path}')
    writer = SummaryWriter(pjoin(args.save_path, './logs'))
    loss_recorder = LossRecorder(writer)

    if len(args.path_to_existing) and args.layered_generator: # 2个条件都不满足
        ConGen = load_all_from_path(args.path_to_existing, args.device, use_class=True)
    else:
        ConGen = None

    gans = []
    gens = []
    amps = [[] for _ in range(len(multiple_data))]
    if args.full_zstar: #n_channels是property，每帧数据的长度174，lengths[i][0]是lengths[i]的最小帧数
        z_star = [torch.randn((1, motion_data.n_channels, lengths[i][0]), device=device) for i in range(len(multiple_data))] # z_star[0].shape = (1, 174, 129)
    else:
        z_star = [torch.randn((1, 1, lengths[i][0]), device=device).repeat(1, motion_data.n_channels, 1) for i in range(len(multiple_data))]
    torch.save(z_star, pjoin(args.save_path, 'z_star.pt'))
    reals = [[] for _ in range(len(multiple_data))]
    gt_deltas = [[] for _ in range(len(multiple_data))]
    training_groups = get_group_list(args, len(lengths[0])) #[[129, 172, 229, 305, 406, 541, 648]]两个一组，所以其值是[[0, 1], [2, 3], [4, 5], [6]]

    #模型部分，以648帧，每帧174个数据（channel）为例
    #首先，根据帧数[129, 172, 229, 305, 406, 541, 648]，有7个generator和discriminator，第一个直接由随机数生成，后续的根据随机数和前面的生成结果来生成
    #然后，每个generator根据channel（在__init__.py的get_channels_list()里）[174, 145, 261, 261, 174]做输入输出有4层
    #     每个discriminator根据[174, 145, 261, 261, 1]也是4层
    #     每个generator和每个discriminator，除了discriminator的最后一层是nn.Conv1d之外，其他每层都是SkeletonConv
    for step in range(len(lengths[0])): #[129, 172, 229, 305, 406, 541, 648]，step从0到6
        for i in range(len(multiple_data)):
            length = lengths[i][step]
            motion_data = multiple_data[i] # (174, 648)
            reals[i].append(motion_data.sample(size=length).to(device)) #把原动作数据按帧数降采样，(174, length)
            last_real = reals[i][-2] if step > 0 else torch.zeros_like(reals[i][-1]) #reals[i]右移一个，最前头补一个0
            amps[i].append(torch.nn.MSELoss()(reals[i][-1], interpolator(last_real, length)) ** 0.5) # 均方损失函数，就是差的平方和，loss(input, target)，单个浮点数
            if step == 0 and args.correct_zstar_gen: #默认参数false
                z_star[i] *= amps[i][0]
            gt_deltas[i].append(reals[i][-1] - interpolator(last_real, length)) #把上一个lenghth的真实数据插值成现在length的数据，然后与现在的真实数据做差

        # 默认args.layered_generator=0，所以这里是create_model（在models\__init__.py里）
        create = create_layered_model if args.layered_generator and step < args.num_layered_generator else create_model
        """
        gen : generator
        disc: discriminator
        gan : WGAN-GP
        """
        gen, disc, gan_model = create(args, motion_data, evaluation=False) #gan_model里含了gan和disc

        gens.append(gen)
        gans.append(gan_model)

    amps = torch.tensor(amps)
    if not args.requires_noise_amp: #requires_noise_amp默认1
        amps = torch.ones_like(amps)
    torch.save(amps, pjoin(args.save_path, 'amps.pt'))

    last_stage = 0
    for group in training_groups: # [[0, 1], [2, 3], [4, 5], [6]]
        curr_stage = last_stage + len(group)
        group_gan_models = [gans[i] for i in group]
        joint_train(reals, gens[:curr_stage], group_gan_models, lengths,
                    z_star, amps, args, loss_recorder, gt_deltas, ConGen)

        for i, gan_model in enumerate(group_gan_models):
            torch.save(gan_model.gen.state_dict(), pjoin(args.save_path, f'gen{group[i]:03d}.pt'))

        last_stage = curr_stage

    end_time = time.time()
    if not args.silent:
        print(f'Training time: {end_time - start_time:.07f}s')


if __name__ == '__main__':
    main()
