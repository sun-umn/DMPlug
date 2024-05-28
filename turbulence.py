import argparse, os, yaml
import torch
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
from util.img_utils import Blurkernel, clear_color, generate_tilt_map, mask_generator
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from ddim_sampler import *
import shutil
import torch.nn.functional as F
from motionblur.motionblur import Kernel
import lpips

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def dmplug(model, scheduler, logdir, img='00000', eta=0, lr=1e-2,lrk=1e-1, lrt=1e-7, dataset='celeba',img_model_config=None,task_config=None,device='cuda'):
    dtype = torch.float32
    gt_img_path = './data/{}/{}.png'.format(dataset,img)
    gt_img = Image.open(gt_img_path).convert("RGB")
    shutil.copy(gt_img_path, os.path.join(logdir, 'gt.png'))
    ref_numpy = np.array(gt_img) / 255.0
    x = ref_numpy * 2 - 1
    x = x.transpose(2, 0, 1)
    ref_img = torch.Tensor(x).to(dtype).to(device).unsqueeze(0)
    ref_img.requires_grad = False
    # Prepare Operator and noise
    kernel_size = task_config["kernel_size"]
    intensity = task_config["intensity"]
    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])

    # tile_map requires loop for generation that could be slow.
    img_size = x.shape[-1]
    tilt = generate_tilt_map(img_h=img_size, img_w=img_size, kernel_size=7, device=device)
    tilt = torch.clip(tilt, -2.5, 2.5)
    # blur kernel
    conv = Blurkernel('gaussian', kernel_size=kernel_size, device=device, std=intensity)
    kernel = conv.get_kernel().type(torch.float32)
    kernel = kernel.to(device).view(1, 1, kernel_size, kernel_size)
    y = operator.forward(ref_img,kernel,tilt)
    y_n = noiser(y)
    y_n.requires_grad = False
    plt.imsave(os.path.join(logdir, 'measurement.png'), clear_color(y_n))

    # DMPlug
    Z = torch.randn((1, 3, img_model_config['image_size'], img_model_config['image_size']), device=device, dtype=dtype, requires_grad=True)
    trainable_kernel = torch.randn((1, kernel_size * kernel_size), device=device, dtype=dtype, requires_grad=True)
    trainable_tilt = torch.randn(1, 2, 256, 256, device=device) * 0.01
    trainable_tilt.requires_grad = True
    criterion = torch.nn.MSELoss().to(device)
    params_group1 = {'params': Z, 'lr': lr}
    params_group2 = {'params': trainable_kernel, 'lr': lrk}
    params_group3 = {'params': trainable_tilt, 'lr': lrt}
    optimizer = torch.optim.Adam([params_group1,params_group2,params_group3])

    epochs = 10000
    psnrs = []
    ssims = []
    losses = []
    lpipss = []
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    for iterator in range(epochs):
        model.eval()
        optimizer.zero_grad()

        for i, tt in enumerate(scheduler.timesteps):
            t = (torch.ones(1) * tt).cuda()
            if i == 0:
                noise_pred = model(Z, t)
            else:
                noise_pred = model(x_t, t)
            noise_pred = noise_pred[:, :3]
            if i == 0:
                x_t = scheduler.step(noise_pred, tt, Z, return_dict=True, use_clipped_model_output=True, eta=eta).prev_sample
            else:
                x_t = scheduler.step(noise_pred, tt, x_t, return_dict=True, use_clipped_model_output=True, eta=eta).prev_sample

        output = torch.clamp(x_t, -1, 1)
        kernel_output = F.softmax(trainable_kernel, dim=1)
        out_k = kernel_output.view(1, 1, kernel_size, kernel_size)
        loss = criterion(operator.forward(output, out_k, trainable_tilt), y_n)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        with torch.no_grad():
            output_numpy = output.detach().cpu().squeeze().numpy()
            output_numpy = (output_numpy + 1) / 2
            output_numpy = np.transpose(output_numpy, (1, 2, 0))
            # calculate psnr
            tmp_psnr = peak_signal_noise_ratio(ref_numpy, output_numpy)
            psnrs.append(tmp_psnr)
            # calculate ssim
            tmp_ssim = structural_similarity(ref_numpy, output_numpy, channel_axis=2, data_range=1)
            ssims.append(tmp_ssim)
            # calculate lpips
            rec_img_torch = torch.from_numpy(output_numpy).permute(2, 0, 1).unsqueeze(0).float().to(device)
            gt_img_torch = torch.from_numpy(ref_numpy).permute(2, 0, 1).unsqueeze(0).float().to(device)
            rec_img_torch = rec_img_torch * 2 - 1
            gt_img_torch = gt_img_torch * 2 - 1
            lpips_alex = loss_fn_alex(gt_img_torch, rec_img_torch).item()
            lpipss.append(lpips_alex)

            if len(psnrs) == 1 or (len(psnrs) > 1 and tmp_psnr > np.max(psnrs[:-1])):
                best_img = output_numpy
    plt.imsave(os.path.join(logdir, "rec_img.png"), best_img)

    plt.plot(np.array(losses), label='all')
    plt.legend()
    plt.savefig(os.path.join(logdir, 'loss.png'))
    plt.close()

    plt.plot(np.array(psnrs))
    plt.title('Max PSNR: {}'.format(np.max(np.array(psnrs))))
    plt.savefig(os.path.join(logdir, 'psnr.png'))
    plt.close()

    psnr_res = np.max(psnrs)
    ssim_res = np.max(ssims)
    lpips_res = np.min(lpipss)
    print('PSNR: {}, SSIM: {}, LPIPS: {}'.format(psnr_res, ssim_res, lpips_res))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=0.0
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="logdir",
        default="./results"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="?",
        help="dataset",
        default="celeba"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fast sampling",
        default=3
    )
    parser.add_argument(
        "--lr",
        type=float,
        nargs="?",
        help="lr of z",
        default=0.01
    )
    parser.add_argument(
        "--img",
        type=int,
        nargs="?",
        help="image id",
        default=0
    )
    parser.add_argument(
        "--lrk",
        type=float,
        nargs="?",
        help="lr of the blur kernel",
        default=1e-1
    )
    parser.add_argument(
        "--lrt",
        type=float,
        nargs="?",
        help="lr of tilt map",
        default=1e-7
    )
    return parser
def torch_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == "__main__":
    torch_seed(123)
    # Load configurations
    parser = get_parser()
    device = torch.device("cuda")
    opt, unknown = parser.parse_known_args()
    img_model_config = 'configs/model_config_{}.yaml'.format(opt.dataset)
    task_config = 'configs/tasks/turbulence_config.yaml'
    img_model_config = load_yaml(img_model_config)
    model = create_model(**img_model_config)
    model = model.to(device)
    model.eval()
    task_config = load_yaml(task_config)
    # Define the DDIM scheduler
    scheduler = DDIMScheduler()
    scheduler.set_timesteps(opt.custom_steps)
    img = str(opt.img).zfill(5)
    logdir = os.path.join(opt.logdir, 'turbulence', opt.dataset, img)
    os.makedirs(logdir,exist_ok=True)
    # DMPlug
    dmplug(model, scheduler, logdir, img=img, eta=opt.eta, lr=opt.lr, lrk=opt.lrk, lrt=opt.lrt, dataset=opt.dataset, img_model_config=img_model_config, task_config = task_config, device=device)