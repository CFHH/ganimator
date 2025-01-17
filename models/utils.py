import torch
import torch.nn as nn
import random
from models.transforms import repr6d2quat, euler2mat
import torch.nn.functional as F
from functools import partial
from scipy.ndimage.filters import gaussian_filter


def gaussian_filter_wrapper(x: torch.Tensor, sigma: float):
    res = []
    for i in range(x.shape[1]):
        nx = x[:, [i]]
        n_shape = nx.shape
        nx = nx.reshape(-1)
        nx = gaussian_filter(nx.numpy(), sigma=sigma, mode='nearest')
        nx = torch.tensor(nx)
        nx = nx.reshape(n_shape)
        res.append(nx)
    res = torch.cat(res, dim=1)
    return res


def get_layered_mask(layer_mode, n_rot=6):
    """
    Get index of given channels in tensor representation
    :param layer_mode: 'loc' or 'locrot'
    :param n_rot: number of channels for a rotation
    :return:
    """
    if 'xz' in layer_mode:
        mask_loc = [-6, -4]
    else:
        mask_loc = [-6, -5, -4]

    if 'locrot' in layer_mode:
        mask_layered = mask_loc + list(range(n_rot))
    elif 'loc' in layer_mode:
        mask_layered = mask_loc
    elif 'all' in layer_mode:
        mask_layered = slice(None)
    else:
        raise Exception('Unknown layer mode')
    return mask_layered


def get_interpolator(args=None, linear=None, nearest=None):
    """
    Return 1D interpolator based on given parameters. Note that args has highest priority
    """
    if args is not None:
        nearest = args.nearest_interpolation
        linear = not args.nearest_interpolation
    if linear and nearest:
        raise Exception('Cannot use both linear and nearest interpolator')

    if linear:
        return partial(F.interpolate, mode='linear', align_corners=False)
    elif nearest:
        return partial(F.interpolate, mode='nearest')
    else:
        raise Exception('Unknown interpolate mode')


class ImagePool:
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


class GAN_loss(nn.Module):
    def __init__(self, gan_mode, real_label=1.0, fake_label=0.0):
        super(GAN_loss, self).__init__()
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgan-gp':
            self.loss = self.wgan_loss
            real_label = 1
            fake_label = 0
        elif gan_mode == 'none':
            self.loss = None
        else:
            raise Exception('Unknown GAN mode')

        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))

    @staticmethod
    def wgan_loss(prediction, target):
        # target不是True就是False
        lmbda = torch.ones_like(target)
        lmbda[target == 1] = -1
        return (prediction * lmbda).mean()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss


class VeloLabelConsistencyLoss(nn.Module):
    """
    ZZW TODO BATCH
    This class does not support batching!!!!
    """
    def __init__(self, motion_data, detach_label=False, use_sigmoid=False, use_6d_fk=False): # 默认use_sigmoid=true
        super(VeloLabelConsistencyLoss, self).__init__()
        self.bvh_file = motion_data.bvh_file
        self.motion_data = motion_data
        self.use_6d_fk = use_6d_fk
        self.register_buffer('offset', self.bvh_file.skeleton.offsets)
        self.detach_label = detach_label
        self.use_sigmoid = use_sigmoid

    def __call__(self, motion):
        """
        ZZW TODO 这里要把rot要用quat表示，euler和mat没处理
        ['euler', 'quat', 'quaternion', 'repr6d', 'mat']:
        """
        pos, rot, contact = self.motion_data.parse(motion)
        if rot.shape[-1] == 6 and not self.use_6d_fk:
            rot = repr6d2quat(rot)
        pos_all = self.bvh_file.fk.forward(rot, pos, self.offset) # 调用kinematics
        pos_contact = pos_all[:, self.bvh_file.skeleton.contact_id]

        from models.contact import velocity
        velo = velocity(pos_contact, padding=True)
        if self.detach_label:
            contact = contact.detach()
        if self.use_sigmoid:
            contact = torch.sigmoid((contact - 0.5) * 2 * 6)
        return (velo * contact).mean()


class RecLoss(nn.Module):
    def __init__(self, motion_data, velo_cumsum=False, loss_type='L2', lambda_pos=0., extra_velo=False):
        super(RecLoss, self).__init__()
        self.motion_data = motion_data
        self.velo_cumsum = velo_cumsum
        self.lambda_pos = lambda_pos
        self.extra_velo = extra_velo
        if loss_type == 'L2':
            self.criteria = nn.MSELoss()
        elif loss_type == 'L1':
            self.criteria = nn.L1Loss() # 做差，取绝对值，再平均
        else:
            raise Exception('Unknown loss type')

    def __call__(self, a, b):
        if self.lambda_pos > 0: #实际是0，不执行
            a_pos = self.motion_data.parse(a, keep_velo=self.extra_velo)[0]
            b_pos = self.motion_data.parse(b, keep_velo=self.extra_velo)[0]
            loss_pos = self.criteria(a_pos, b_pos)
        else:
            loss_pos = 0.

        if self.velo_cumsum: #实际是False，不执行
            # ZZW TODO BATCH
            a = self.motion_data.velo2pos(a)
            b = self.motion_data.velo2pos(b)

        # 最终就是self.criteria(a, b)
        return self.criteria(a, b) + loss_pos * self.lambda_pos


class DeltaLoss(nn.Module):
    def __init__(self, delta_loss_mode, n_rot, uniform, deduction, gt_delta=None, eps=1e-6, previous_length=None):
        super(DeltaLoss, self).__init__()
        self.eps = eps
        self.uniform = uniform
        self.deduction = deduction
        self.channels = get_layered_mask(delta_loss_mode, n_rot)
        self.gt_delta = self.get_gt_delta(gt_delta) if gt_delta is not None else None
        self.normalizer = None
        self.previous_length = previous_length

    def get_gt_delta(self, gt_delta):
        gt_delta = gt_delta[:, self.channels]
        gt_delta = (gt_delta ** 2).mean(dim=-1, keepdim=True)
        return gt_delta

    def set_gt_delta(self, gt_delta):
        self.gt_delta = self.get_gt_delta(gt_delta)
        self.normalizer = self.gt_delta.clone()
        self.normalizer[self.normalizer < self.eps] = self.eps

    def __call__(self, res):
        if self.gt_delta is None:
            return torch.tensor(0., device=res.device)
        if self.previous_length is not None:
            interpolator = get_interpolator(linear=True)
            res = interpolator(res, self.previous_length)
        res = res[:, self.channels]
        res = res**2
        if self.deduction:
            res = res - self.gt_delta
        if self.uniform:
            res = res / self.normalizer
        res = torch.relu(res)
        return res.mean()
