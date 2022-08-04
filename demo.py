import torch
import os
from os.path import join as pjoin
from dataset.motion import MotionData, load_multiple_dataset, load_slice_dataset
from models import create_model, create_layered_model
from models.architecture import draw_example, get_pyramid_lengths, FullGenerator
from option import EvaluateOptionParser, TrainOptionParser
from fix_contact import fix_contact_on_file
from bvh.bvh_io import get_frame_info
import random
from tqdm import tqdm, trange


def load_all_from_path(save_path, device, use_class=False):
    train_parser = TrainOptionParser()
    args = train_parser.load(pjoin(save_path, 'args.txt')) #训练时的参数设置
    args.device = device
    args.save_path = save_path
    device = torch.device(args.device)

    if not args.multiple_sequences:
        file_name = pjoin(args.bvh_prefix, f'{args.bvh_name}.bvh')
        if args.slice:
            frame_num, frame_time = get_frame_info(file_name)
            slice_frame_num = round(args.slice_time_len / frame_time)
            start_frame = 0 if args.start_frame <= 0 else args.start_frame
            end_frame = frame_num if (args.end_frame < 0 or args.end_frame > frame_num) else args.end_frame
            assert end_frame > start_frame
            multiple_data = load_slice_dataset(start_frame, end_frame, slice_frame_num,
                                               filename=file_name,
                                               padding=args.skeleton_aware, use_velo=args.use_velo, repr=args.repr,
                                               contact=args.contact, keep_y_pos=args.keep_y_pos,
                                               joint_reduction=args.joint_reduction)
            motion_data = multiple_data[0]
        else:
            motion_data = MotionData(file_name,
                                     padding=args.skeleton_aware, use_velo=args.use_velo, repr=args.repr,
                                     contact=args.contact, keep_y_pos=args.keep_y_pos)
            multiple_data = [motion_data]
    else:
        multiple_data = load_multiple_dataset(prefix=args.bvh_prefix, name_list=pjoin(args.bvh_prefix, args.bvh_name),
                                              padding=args.skeleton_aware, use_velo=args.use_velo, repr=args.repr,
                                              contact=args.contact, keep_y_pos=args.keep_y_pos,
                                              no_scale=True)
        motion_data = multiple_data[0]

    lengths = []
    min_len = 10000
    for i in range(len(multiple_data)):
        new_length = get_pyramid_lengths(args, len(multiple_data[i]))
        min_len = min(min_len, len(new_length))
        if args.num_stages_limit != -1:
            new_length = new_length[:args.num_stages_limit]
        lengths.append(new_length)

    for i in range(len(multiple_data)):
        lengths[i] = lengths[i][-min_len:]

    gens = [] #7个
    for step, length in enumerate(lengths[0]):
        create = create_layered_model if args.layered_generator and step < args.num_layered_generator else create_model
        gen = create(args, motion_data, evaluation=True)
        try:
            gen_sate = torch.load(pjoin(args.save_path, f'gen{step:03d}.pt'), map_location=device)
        except FileNotFoundError:
            gen_sate = torch.load(pjoin(args.save_path, f'gen{step}.pt'), map_location=device)
        gen.load_state_dict(gen_sate)
        gens.append(gen)
    z_star = torch.load(pjoin(args.save_path, 'z_star.pt'), map_location=device)
    amps = torch.load(pjoin(args.save_path, 'amps.pt'), map_location=device)
    if use_class:
        if isinstance(z_star, list):
            z_star = z_star[0]
        if len(amps.shape) != 1:
            amps = amps[0]
        return FullGenerator(args, motion_data, gens, z_star, amps)
    else:
        if len(amps.shape) == 1:
            amps = amps.unsqueeze(0)
        if isinstance(z_star, torch.Tensor) and len(z_star.shape) == 3:
            z_star = z_star.unsqueeze(0)
        return args, multiple_data, gens, z_star, amps, lengths


def write_multires(imgs, prefix, writer, interpolator, full_lengths=None, requires_con_loss=True):
    os.makedirs(prefix, exist_ok=True)
    length = imgs[-1].shape[-1] if full_lengths is None else full_lengths
    res = []
    for step, img in enumerate(imgs):
        full_length = interpolator(img, length)
        writer(pjoin(prefix, f'{step:02d}.bvh'), full_length)
        velo = full_length[:, -6:-3].norm(dim=1)
        res.append(velo)
    if requires_con_loss:
        res = torch.cat(res, dim=0)
        consistency_loss = torch.nn.MSELoss()(res[1], res[0])
        return consistency_loss


def gen_noise(n_channel, length, full_noise, device):
    if full_noise:
        res = torch.randn((1, n_channel, length)).to(device)
    else:
        res = torch.randn((1, 1, length)).repeat(1, n_channel, 1).to(device)
    return res


def main():
    eval_parser = EvaluateOptionParser()
    eval_args = eval_parser.parse_args()

    #加载训练产生的数据
    train_args, multiple_data, gens, z_stars, amps, lengths = load_all_from_path(eval_args.save_path, eval_args.device)

    device = torch.device(train_args.device)
    n_total_levels = len(gens)

    motion_data = multiple_data[0]

    noise_channel = z_stars[0].shape[1] if train_args.full_noise else 1

    if len(train_args.path_to_existing):
        ConGen = load_all_from_path(train_args.path_to_existing, train_args.device, use_class=True)
    else:
        ConGen = None

    print('levels:', lengths)
    save_path = pjoin(train_args.save_path, 'bvh') #bvh文件夹
    os.makedirs(save_path, exist_ok=True)

    # Evaluate with reconstruct noise
    conds_rec = None
    for i in range(len(multiple_data)):
        #这for的代码没有用处，就是测试一下
        motion_data = multiple_data[i]
        #'rec'改成'random'
        imgs = draw_example(gens, 'rec', z_stars[i], lengths[i] + [1], amps[i], 1, train_args, all_img=True, conds=conds_rec,
                            full_noise=train_args.full_noise)
        real = motion_data.sample(size=len(motion_data), slerp=train_args.slerp).to(device)
        motion_data.write(pjoin(save_path, f'gt_{i}.bvh'), real, scale100=True, fix_euler=True) #真实动作
        motion_data.write(pjoin(save_path, f'rec_{i}.bvh'), imgs[-1], scale100=True, fix_euler=eval_args.fix_euler) #生成动作

        if imgs[-1].shape[-1] == real.shape[-1]:
            rec_loss = torch.nn.MSELoss()(imgs[-1], real).detach().cpu().numpy()
            print(f'rec_loss: {rec_loss.item():.07f}')

    target_len = eval_args.target_length #目标生成动作的帧数
    target_length = get_pyramid_lengths(train_args, target_len) #帧数序列
    while len(target_length) > n_total_levels:
        target_length = target_length[1:]


    generate_num = eval_args.gen_num
    for index in trange(generate_num):
        base_id = random.randint(0, len(multiple_data) - 1)
        z_length = target_length[0] #最短的那个帧数
        z_target = gen_noise(noise_channel, z_length, train_args.full_noise, device)
        z_target *= amps[base_id][0]

        amps2 = amps[base_id].clone()
        amps2[1:] = 0

        #这里上真正的生成代码
        imgs = draw_example(gens, 'random', z_stars[base_id], target_length, amps2, 1, train_args, all_img=True,
                            conds=None, full_noise=train_args.full_noise, given_noise=[z_target])
        motion_data.write(pjoin(save_path, 'result_%03d.bvh' % index), imgs[-1], scale100=False, fix_euler=False)
        motion_data.write(pjoin(save_path, 'result_%03d_unfixed.bvh' % index), imgs[-1], scale100=True, fix_euler=eval_args.fix_euler)
        if train_args.contact:
            fix_contact_on_file(save_path, name=f'result_%03d' % index, scale100=True, fix_euler=eval_args.fix_euler)


if __name__ == '__main__':
    main()
