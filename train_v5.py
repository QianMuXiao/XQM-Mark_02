# /Users/qianmu/Desktop/Mark02/compair_model/FDA_V2/train.py


import os
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler as DSampler
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
from Sample import cont_sample, style_sample
from Models_v2 import AutoencoderKL
from generative.networks.nets import PatchDiscriminator
# from Memory_pair_v3 import MemorySharedParts, seg_idx_1  # 更新：精简版记忆库
from Memory_pair_v7 import MemorySharedPartsV7, SEG_IDX_V7, SEG_GROUPS_MAJOR


from loss_fn import compute_kl_loss, discriminator_loss
from generative.losses import PerceptualLoss, PatchAdversarialLoss
from pytorch_msssim import ssim
from lpips import LPIPS
from mri_dataset_v4 import PairMRIDataset

# torch.autograd.set_detect_anomaly(True)

# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim

# --------------------------
# 一些通用小工具 / 损失
# --------------------------
def is_master():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def hinge_d_loss(real_logits, fake_logits):
    loss_real = F.relu(1.0 - real_logits).mean()
    loss_fake = F.relu(1.0 + fake_logits).mean()
    return loss_real + loss_fake

intensity_loss = torch.nn.L1Loss()


# def setup_distributed():
#     """
#     初始化分布式训练，自动检测 GPU 并设置当前进程的设备
#     """
#     if not dist.is_initialized():
#         dist.init_process_group(backend='nccl')  # 使用 NCCL 后端进行 GPU 通信
#     local_rank = int(os.environ['LOCAL_RANK'])  # 通过环境变量获取本地进程的 rank
#     torch.cuda.set_device(local_rank)  # 将当前进程绑定到对应的 GPU
#     device = torch.device(f'cuda:{local_rank}')
#     return device


def setup_distributed():
    """
    初始化分布式训练，自动检测 GPU 并设置当前进程的设备
    """

    if not dist.is_initialized():
        backend = 'gloo' if (os.name == 'nt') else 'nccl'  # Windows 用 gloo
        dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))  # 单卡/调试时兜底为0
    torch.cuda.set_device(local_rank)  # 将当前进程绑定到对应的 GPU
    device = torch.device(f'cuda:{local_rank}')
    return device

def init_memory_list(seg_idx, kdim, vdim, memory_slots, device, seed=42):
    """
    在 rank0 初始化每类记忆 (key,val_a,val_b)，并广播到其它 rank。
    返回: memory_size_list, memorys(list of tuples)
    """
    if isinstance(memory_slots, int):
        memory_size_list = [memory_slots] * len(seg_idx)
    else:
        assert len(memory_slots) == len(seg_idx), "memory_slots 长度需与 seg_idx 一致"
        memory_size_list = list(memory_slots)

    world_ok = dist.is_initialized()
    rank = dist.get_rank() if world_ok else 0
    g = torch.Generator(device=device).manual_seed(seed) if rank == 0 else None

    memorys = []
    for Mi in memory_size_list:
        if rank == 0:
            key   = torch.randn((Mi, kdim), generator=g, device=device)
            key   = torch.nn.functional.normalize(key, dim=1)
            val_a = torch.randn((Mi, vdim), generator=g, device=device) * 0.02
            val_b = torch.randn((Mi, vdim), generator=g, device=device) * 0.02
        else:
            key   = torch.empty((Mi, kdim), device=device)
            val_a = torch.empty((Mi, vdim), device=device)
            val_b = torch.empty((Mi, vdim), device=device)
        if world_ok:
            dist.broadcast(key,   src=0)
            dist.broadcast(val_a, src=0)
            dist.broadcast(val_b, src=0)
        memorys.append((key, val_a, val_b))
    return memory_size_list, memorys

# def sync_memory_list(memorys):
#     """
#     跨卡同步记忆库：对每类 (key,val_a,val_b) 做 all_reduce 求均值，并重归一化 key。
#     注意：不原地修改输入，返回 clone().detach() 的副本，避免破坏计算图。
#     """
#     if not dist.is_initialized() or dist.get_world_size() == 1:
#         # 仍返回脱钩副本
#         return [(k.detach().clone(), va.detach().clone(), vb.detach().clone()) for (k, va, vb) in memorys]
#     world = dist.get_world_size()
#     synced = []
#     with torch.no_grad():
#         for (k, va, vb) in memorys:
#             k  = k.detach().clone()
#             va = va.detach().clone()
#             vb = vb.detach().clone()
#             dist.all_reduce(k,  op=dist.ReduceOp.SUM);  k  /= world
#             dist.all_reduce(va, op=dist.ReduceOp.SUM);  va /= world
#             dist.all_reduce(vb, op=dist.ReduceOp.SUM);  vb /= world
#             k = torch.nn.functional.normalize(k, dim=1)
#             synced.append((k, va, vb))
#     return synced

def sync_memory_list(memorys):
    """
    跨卡同步记忆库：对每类 (key,val_a,val_b) 做 all_reduce 求均值，并重归一化 key。
    优化：当每类 memory 的 slot 数一致时，将 (3 * num_classes) 次 all_reduce 融合成 3 次。
    注意：不原地修改输入，返回 detach().clone() 的副本，避免破坏计算图。
    """
    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_world_size() == 1:
        return [(k.detach().clone(), va.detach().clone(), vb.detach().clone()) for (k, va, vb) in memorys]

    world = dist.get_world_size()

    # 检查每类 slots 数是否一致；不一致则回退到原来的逐类 reduce（保证健壮、语义不变）
    sizes = [k.shape[0] for (k, _, _) in memorys]
    same_size = all(s == sizes[0] for s in sizes)

    with torch.no_grad():
        if same_size:
            # [C, M, D]
            K = torch.stack([k.detach() for (k, _, _) in memorys], dim=0).contiguous()
            VA = torch.stack([va.detach() for (_, va, _) in memorys], dim=0).contiguous()
            VB = torch.stack([vb.detach() for (_, _, vb) in memorys], dim=0).contiguous()

            dist.all_reduce(K, op=dist.ReduceOp.SUM);  K /= world
            dist.all_reduce(VA, op=dist.ReduceOp.SUM); VA /= world
            dist.all_reduce(VB, op=dist.ReduceOp.SUM); VB /= world

            # 对每个 slot 的 key 向量做归一化：dim=-1（最后一维是 kdim）
            K = torch.nn.functional.normalize(K, dim=-1)

            synced = [(K[i].clone(), VA[i].clone(), VB[i].clone()) for i in range(K.size(0))]
            return synced

        # fallback：原始写法（逐类逐张量 all_reduce）
        synced = []
        for (k, va, vb) in memorys:
            k  = k.detach().clone()
            va = va.detach().clone()
            vb = vb.detach().clone()

            dist.all_reduce(k,  op=dist.ReduceOp.SUM);  k  /= world
            dist.all_reduce(va, op=dist.ReduceOp.SUM);  va /= world
            dist.all_reduce(vb, op=dist.ReduceOp.SUM);  vb /= world

            k = torch.nn.functional.normalize(k, dim=1)
            synced.append((k, va, vb))
        return synced

def tahn2sigmoid(input):
    return (input + 1) / 2

def calculate_psnr(img1, img2):
    mse = nn.functional.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def build_mirror_gauss_sigma(mu, sigma):
    """
    FDA 伪镜像：翻转均值，协方差不变（保持 sigma）
    """
    return -mu, sigma


# --------------------------
# 训练主函数
# --------------------------
def train_FDA_V2(
    train_loader, 
    train_sampler,
    cont_sampler,
    style_sampler,
    val_loader,
    val_sampler,
    generator,
    disc_A,
    disc_B,
    cont_disc,
    adv_loss,
    memory_module,
    memory_list,                       # 记忆库列表（外部维护）
    opt_gen,
    opt_dis_A,
    opt_dis_B,
    opt_cont_sampler,
    opt_style_sampler,
    scheduler_gen=None,
    scheduler_dis_A=None,
    scheduler_dis_B=None,
    perceptual_loss = None,
    lpips = None,
    d_train_freq=1,
    device=None,
    Max_Epoch=1501,
    loss_cfg=None,
    warm_up_epoch=3,
    writer=None,
    model_save_path = None,
    model_save_interval=10,
):
    """
    loss_cfg: 可选损失权重字典，例如：
        loss_cfg = dict(
            w_kl=1.0, w_fda=1.0,
            w_rec=10.0, w_cnt=1.0,
            w_adv=1.0, w_pair=10.0,
            w_mirror=0.5, w_style_align=0.2,
            fda_cov_mode="sigma",   # or "logvar"
            use_random_branch=True
        )
    """
    if loss_cfg is None:
        loss_cfg = dict(
            w_kl=1e-7, 
            w_fda=0.01,
            w_rec_cs=0.1,
            w_rec_raw=0.01,
            w_trans=1.0,
            w_prece=0.01,
            w_adv=0.01, 
            w_key=0.1,
            w_value=0.1,
            w_kl_diff=0.01,
            fda_cov_mode="sigma",
            use_random_branch=True
        )

    scaler = None  # 如需 AMP，可替换为 torch.cuda.amp.GradScaler()

    current_memorys = memory_list  # 训练过程中不断更新的记忆库列表
    
    try:
        
        best_a2b_psnr = 0
    
        for epoch_nums in range(Max_Epoch):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch_nums)

            generator.train()
            disc_A.train()
            disc_B.train()
            # memory_module.train()  # 只有 forward 内部的 IN1d 起作用；库本身通过外部张量更新

            # --- memory usage stats accumulator (warmup 后才统计) ---
            mem_pix_acc = None
            mem_hit_acc = None
            if epoch_nums >= warm_up_epoch and hasattr(memory_module, 'memory_size'):
                mem_pix_acc = torch.zeros((len(memory_module.seg_idx),), device=device, dtype=torch.float32)
                mem_hit_acc = torch.zeros((int(sum(memory_module.memory_size)),), device=device, dtype=torch.float32)

            pbar = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch_nums}', dynamic_ncols=True) if is_master() else train_loader

            for batch_idx, (image_A, image_B, mask, _, _) in enumerate(pbar):
                image_A = image_A.to(device, non_blocking=True)
                image_B = image_B.to(device, non_blocking=True)
                mask    = mask.to(device, non_blocking=True)   # [B, N, H, W]

                # --------------------------
                # 1) 编码得到两域分布同时得到两个分布的翻转分布
                # 
                # --------------------------
                z_mu_A,  z_sigma_A = generator.module.encode(image_A)  # [B,C,H,W] x2
                z_mu_B,  z_sigma_B = generator.module.encode(image_B)

                fake_z_mu_A2B, fake_z_sigma_A2B = build_mirror_gauss_sigma(z_mu_A, z_sigma_A)  # 伪镜像
                fake_z_mu_B2A, fake_z_sigma_B2A = build_mirror_gauss_sigma(z_mu_B, z_sigma_B)
                
                # KL + FDA 对称性（sigma 版本）


                # --------------------------
                # 2) 真实对分布上的随机方向解耦（两次独立采样）
                # z_samp_A, z_samp_B: 从真实图片的特征分布上分别采样点
                # cont_real: 从一对真实图片上解码的内容特征
                # style_A_real, style_B_real: 从一对真实图片上解码的风格特征
                # 用于正确的更新记忆库
                # --------------------------
                z_samp_A = generator.module.sampling(z_mu_A, z_sigma_A)  # reparam
                z_samp_B = generator.module.sampling(z_mu_B, z_sigma_B)

                cont_real, _ = cont_sampler(
                    z_mu_A, z_mu_B, z_sigma_A, z_sigma_B, z_samp_A, z_samp_B
                )                                   # [B,C,H,W]
                style_A_real, style_B_real, _, _ = style_sampler(
                    z_mu_A, z_mu_B, z_sigma_A, z_sigma_B, z_samp_A, z_samp_B
                )                                   # [B,C,H,W], [B,C,H,W]

                # --------------------------
                # 2.1) 镜像分布特征解耦
                # 变量命名规则
                #
                # sample_fake_AA: 再次从 A 域真实分布采样（用于验证阶段仅有A输入时）
                # sample_fake_AB: 从 A 域的镜像分布采样（用于跨域合成，形成伪分布对）
                # sample_fake_BB: 再次从 B 域真实分布采样（用于验证阶段仅有B输入时）
                # sample_fake_BA: 从 B 域的镜像分布采样（用于跨域合成，形成伪分布对）
                #
                # cont_mirror_A2B: 从 A 域真实分布和 A 域镜像分布解耦得到的内容特征（用于跨域合成）
                # style_fake_AA: 从 A 域真实分布和 A 域镜像分布解耦得到的A风格特征（理论上用于推理阶段自重建，但是暂时不用）
                # style_fake_AB: 从 A 域真实分布和 A 域镜像分布解耦得到的B风格特征（用于跨域合成）
                #
                # cont_mirror_B2A: 从 B 域真实分布和 B 域镜像分布解耦得到的内容特征（用于跨域合成）
                # style_fake_BB: 从 B 域真实分布和 B 域镜像分布解耦得到的B风格特征（理论上用于推理阶段自重建，但是暂时不用）
                # style_fake_BA: 从 B 域真实分布和 B 域镜像分布解耦得到的A风格特征（用于跨域合成）
                #
                #
                # --------------------------
                
                # sample_fake_AA = generator.module.sampling(z_mu_A, z_sigma_A)
                sample_fake_AA = z_samp_A
                sample_fake_AB = generator.module.sampling(fake_z_mu_A2B, fake_z_sigma_A2B)
                
                # sample_fake_BB = generator.module.sampling(z_mu_B, z_sigma_B)
                sample_fake_BB = z_samp_B
                sample_fake_BA = generator.module.sampling(fake_z_mu_B2A, fake_z_sigma_B2A)
                
                
                
                cont_mirror_A2B, _ = cont_sampler(z_mu_A, fake_z_mu_A2B, z_sigma_A, fake_z_sigma_A2B, sample_fake_AA, sample_fake_AB)
                
                style_fake_AA, style_fake_AB, _, _ = style_sampler(z_mu_A, fake_z_mu_A2B, z_sigma_A, fake_z_sigma_A2B, sample_fake_AA, sample_fake_AB)
                
                
                
                
                cont_mirror_B2A, _ = cont_sampler(fake_z_mu_B2A, z_mu_B, fake_z_sigma_B2A, z_sigma_B, sample_fake_BA, sample_fake_BB)
                
                style_fake_BA, style_fake_BB, _, _ = style_sampler(fake_z_mu_B2A, z_mu_B, fake_z_sigma_B2A, z_sigma_B, sample_fake_BA, sample_fake_BB)
                

                # --------------------------
                # 3) warm_up写库 + 读库（真实对）
                # 这里需要补充对内容和风格特征的通道变换投影层的设计。
                # 
                # warm_up_epoch 之后开始使用记忆库:
                # 对于大于 warm_up_epoch 的 epoch：
                # 使用从真实图片对中解耦得到的内容和风格特征进行记忆库的读写，并更新记忆库
                # 
                # --------------------------
                
                if epoch_nums >= warm_up_epoch:
                    
                    # 这里是当超过 warm_up_epoch 后，才开始使用记忆库并进行读写，其中自重建的管线读写记忆库，跨域合成的时候仅读取记忆库
                    update_memorys, read_real_sty_A, read_real_sty_B, _, _, key_loss, value_loss = memory_module(
                        cont_real, style_A_real, style_B_real, mask, current_memorys)

                    # accumulate lightweight memory stats (per-batch)
                    if mem_pix_acc is not None and getattr(memory_module, 'last_stats', None) is not None:
                        st = memory_module.last_stats
                        if st is not None and 'pix_per_class' in st and 'hit_global' in st:
                            mem_pix_acc = mem_pix_acc + st['pix_per_class']
                            mem_hit_acc = mem_hit_acc + st['hit_global']
                    

                    _, read_fake_sty_AB = memory_module.forward_second(cont_mirror_A2B, mask, current_memorys)
                    read_fake_sty_BA, _ = memory_module.forward_second(cont_mirror_B2A, mask, current_memorys)

                    style_fake_AB = read_fake_sty_AB
                    style_fake_BA = read_fake_sty_BA
                    
                    style_A_real = read_real_sty_A
                    style_B_real = read_real_sty_B


                rec_AA_raw = generator.module.decode_A_raw(z_samp_A)
                rec_BB_raw = generator.module.decode_B_raw(z_samp_B)
                
                # rec_AA_cs_1 = generator.module.decode_A_cs(cont_mirror_A2B, style_fake_AA) 
                # rec_BB_cs_1= generator.module.decode_B_cs(cont_mirror_B2A, style_fake_BB)
                
                # rec_AA_cs_2 = generator.module.decode_A_cs(cont_real, style_A_real) 
                # rec_BB_cs_2 = generator.module.decode_B_cs(cont_real, style_B_real)
                
                rec_AA_cs = generator.module.decode_A_cs(cont_real, style_A_real) 
                rec_BB_cs = generator.module.decode_B_cs(cont_real, style_B_real)
                
                trans_AB = generator.module.decode_B_cs(cont_mirror_A2B, style_fake_AB)
                trans_BA = generator.module.decode_A_cs(cont_mirror_B2A, style_fake_BA)

                # --------------------------
                # 4) 计算并更新判别器
                # --------------------------
                d_total_loss_A = torch.zeros(1).to(device)
                d_total_loss_B = torch.zeros(1).to(device)
                d_total_loss_A_real = torch.zeros(1).to(device)
                d_total_loss_B_real = torch.zeros(1).to(device)
                d_total_loss_A_fake = torch.zeros(1).to(device)
                d_total_loss_B_fake = torch.zeros(1).to(device)
                
                
                
                for _ in range(d_train_freq):
                    # 判别器 B
                    d_A2B_loss, d_A2B_fake, d_A2B_real = discriminator_loss(
                        trans_AB, image_B, disc_net=disc_B)
                    
                    d_B_loss = d_A2B_loss
                    d_B_fake = d_A2B_fake
                    d_B_real = d_A2B_real
                    
                    opt_dis_B.zero_grad()
                    d_B_loss.backward()
                    opt_dis_B.step()

                    d_total_loss_B += d_B_loss
                    d_total_loss_B_real += d_B_real
                    d_total_loss_B_fake += d_B_fake
                    
                    
                    # 判别器 A
                    d_B2A_loss, d_B2A_fake, d_B2A_real = discriminator_loss(
                        trans_BA, image_A, disc_net=disc_A)
                    
                    d_A_loss = d_B2A_loss
                    d_A_fake = d_B2A_fake
                    d_A_real = d_B2A_real

                    opt_dis_A.zero_grad()
                    d_A_loss.backward()
                    opt_dis_A.step()
                    
                    
                    d_total_loss_A += d_A_loss
                    d_total_loss_A_real += d_A_real
                    d_total_loss_A_fake += d_A_fake
                    
                # --------------------------
                # 5) 计算并损失并更新生成器（含编码器、解码器）
                # --------------------------
                
                
                kl_A = compute_kl_loss(z_mu_A, z_sigma_A)
                kl_B = compute_kl_loss(z_mu_B, z_sigma_B)
                kl_loss = (kl_A + kl_B) * 0.5

                trans_loss_A2B = intensity_loss(trans_AB, image_B)
                trans_loss_B2A = intensity_loss(trans_BA, image_A)
                trans_loss = (trans_loss_A2B + trans_loss_B2A) * 0.5
                
                rec_loss_AA_raw = intensity_loss(rec_AA_raw, image_A)
                rec_loss_BB_raw = intensity_loss(rec_BB_raw, image_B)
                
                
                # rec_loss_AA_cs_1 = intensity_loss(rec_AA_cs_1, image_A)
                # rec_loss_BB_cs_1 = intensity_loss(rec_BB_cs_1, image_B)
                # rec_loss_AA_cs_2 = intensity_loss(rec_AA_cs_2, image_A)
                # rec_loss_BB_cs_2 = intensity_loss(rec_BB_cs_2, image_B)
                
                
                # rec_loss_AA_cs_2 = intensity_loss(rec_AA_cs_2, image_A)
                # rec_loss_BB_cs_2 = intensity_loss(rec_BB_cs_2, image_B)
                
                # rec_loss_raw = (rec_loss_AA_raw + rec_loss_BB_raw) * 0.5
                # rec_loss_cs = (rec_loss_AA_cs_1 + rec_loss_BB_cs_1 + rec_loss_AA_cs_2 + rec_loss_BB_cs_2) * 0.25


                # rec_loss_AA = (rec_loss_AA_raw + rec_loss_AA_cs_1 + rec_loss_AA_cs_2) / 3
                # rec_loss_BB = (rec_loss_BB_raw + rec_loss_BB_cs_1 + rec_loss_BB_cs_2) / 3
                # rec_loss = (rec_loss_AA + rec_loss_BB) * 0.5

                
                rec_loss_AA_cs = intensity_loss(rec_AA_cs, image_A)
                rec_loss_BB_cs = intensity_loss(rec_BB_cs, image_B)
                
                rec_loss_AA = (rec_loss_AA_raw + rec_loss_AA_cs) / 2
                rec_loss_BB = (rec_loss_BB_raw + rec_loss_BB_cs) / 2
                rec_loss = (rec_loss_AA + rec_loss_BB) * 0.5
                
                rec_loss_raw = (rec_loss_AA_raw + rec_loss_BB_raw) * 0.5
                rec_loss_cs = (rec_loss_AA_cs + rec_loss_BB_cs) * 0.5
                
                
                logits_fake_A2B = disc_B(trans_AB)[-1]
                logits_fake_B2A = disc_A(trans_BA)[-1]
                adv_A2B_loss = adv_loss(logits_fake_A2B, target_is_real = True, for_discriminator=False)
                adv_B2A_loss = adv_loss(logits_fake_B2A, target_is_real = True, for_discriminator=False)
                adv_loss_total = (adv_A2B_loss + adv_B2A_loss) * 0.5
                
                perce_A2B_loss = perceptual_loss(trans_AB.float(), image_B.float())
                perce_B2A_loss = perceptual_loss(trans_BA.float(), image_A.float())
                perce_loss = (perce_A2B_loss + perce_B2A_loss) * 0.5
                
                kl_diff_mu = intensity_loss(z_mu_A, -z_mu_B)  # 均值对称
                kl_diff_sigma = intensity_loss(z_sigma_A.pow(2), z_sigma_B.pow(2))  # 方差相等
                # kl_diff_sigma = intensity_loss(z_sigma_A, z_sigma_B)  # 方差相等
                
                kl_diff_loss = kl_diff_mu / 2 + kl_diff_sigma
                
                if epoch_nums >= warm_up_epoch:
                    total_loss = (loss_cfg['w_kl'] * kl_loss +
                                loss_cfg['w_adv'] * adv_loss_total +
                                loss_cfg['w_rec_raw'] * rec_loss_raw +
                                loss_cfg['w_rec_cs'] * rec_loss_cs +
                                loss_cfg['w_trans'] * trans_loss +
                                loss_cfg['w_kl_diff'] * kl_diff_loss + 
                                loss_cfg['w_key'] * key_loss +
                                loss_cfg['w_value'] * value_loss +
                                loss_cfg['w_prece'] * perce_loss)
                    
                    opt_gen.zero_grad(set_to_none=True)
                    opt_cont_sampler.zero_grad(set_to_none=True)
                    opt_style_sampler.zero_grad(set_to_none=True)
                    total_loss.backward()
                    opt_gen.step()
                    opt_cont_sampler.step()
                    opt_style_sampler.step()
                    
                    current_memorys = sync_memory_list(update_memorys)

                else:
                    total_loss = (loss_cfg['w_kl'] * kl_loss +
                                loss_cfg['w_adv'] * adv_loss_total +
                                loss_cfg['w_rec_raw'] * rec_loss_raw +
                                loss_cfg['w_rec_cs'] * rec_loss_cs +
                                loss_cfg['w_trans'] * trans_loss +
                                loss_cfg['w_kl_diff'] * kl_diff_loss +
                                loss_cfg['w_prece'] * perce_loss)
                    
                    opt_gen.zero_grad()
                    opt_cont_sampler.zero_grad()
                    opt_style_sampler.zero_grad()
                    total_loss.backward()
                    opt_gen.step()
                    opt_cont_sampler.step()
                    opt_style_sampler.step()
                

                
                # --------------------------
                # 6) 记录日志
                # --------------------------
                
                # losses_to_average = {
                #     'g_loss': total_loss.detach(),
                #     'd_A_loss': d_A_loss.item() / d_train_freq,
                #     'd_B_loss': d_B_loss.item() / d_train_freq,
                #     'gan_loss': adv_loss_total.detach(),
                #     'gan_A_real_loss': d_total_loss_A_real.detach() / d_train_freq,
                #     'gan_A_fake_loss': d_total_loss_A_fake.detach() / d_train_freq,
                #     'gan_B_real_loss': d_total_loss_B_real.detach() / d_train_freq,
                #     'gan_B_fake_loss': d_total_loss_B_fake.detach() / d_train_freq,
                #     'gan_A2B_loss': adv_A2B_loss.detach(),
                #     'gan_B2A_loss': adv_B2A_loss.detach(),
                #     'kl_loss': kl_loss.detach(),
                #     'kl_A_loss': kl_A.detach(),
                #     'kl_B_loss': kl_B.detach(),
                #     'kl_diff_mu_loss': kl_diff_mu.detach(),
                #     'kl_diff_sigma_loss': kl_diff_sigma.detach(),
                #     'perce_loss': perce_loss.detach(),
                #     'perce_A2B_loss': perce_A2B_loss.detach(),
                #     'perce_B2A_loss': perce_B2A_loss.detach(),
                #     'recon_loss': rec_loss.detach(),
                #     'trans_loss': trans_loss.detach(),
                #     'recon_loss_AA': rec_loss_AA.detach(),
                #     'recon_loss_BB': rec_loss_BB.detach(),
                #     'trans_loss_A2B': trans_loss_A2B.detach(),
                #     'trans_loss_B2A': trans_loss_B2A.detach(),
                # }
                
                # 统一保持为 Tensor（不调用 .item()），便于 all_reduce
                losses_to_average = {
                    'g_loss': total_loss.detach(),
                    'd_A_loss': (d_total_loss_A / d_train_freq).detach(),
                    'd_B_loss': (d_total_loss_B / d_train_freq).detach(),
                    'gan_loss': adv_loss_total.detach(),
                    'gan_A_real_loss': (d_total_loss_A_real / d_train_freq).detach(),
                    'gan_A_fake_loss': (d_total_loss_A_fake / d_train_freq).detach(),
                    'gan_B_real_loss': (d_total_loss_B_real / d_train_freq).detach(),
                    'gan_B_fake_loss': (d_total_loss_B_fake / d_train_freq).detach(),
                    'gan_A2B_loss': adv_A2B_loss.detach(),
                    'gan_B2A_loss': adv_B2A_loss.detach(),
                    'kl_loss': kl_loss.detach(),
                    'kl_A_loss': kl_A.detach(),
                    'kl_B_loss': kl_B.detach(),
                    'kl_diff_mu_loss': kl_diff_mu.detach(),
                    'kl_diff_sigma_loss': kl_diff_sigma.detach(),
                    'perce_loss': perce_loss.detach(),
                    'perce_A2B_loss': perce_A2B_loss.detach(),
                    'perce_B2A_loss': perce_B2A_loss.detach(),
                    'recon_loss': rec_loss.detach(),
                    'trans_loss': trans_loss.detach(),
                    'recon_loss_AA': rec_loss_AA.detach(),
                    'recon_loss_AA_raw': rec_loss_AA_raw.detach(),
                    # 'recon_loss_AA_cs_1': rec_loss_AA_cs_1.detach(),
                    # 'recon_loss_AA_cs_2': rec_loss_AA_cs_2.detach(),
                    'recon_loss_AA_cs': rec_loss_AA_cs.detach(),
                    'recon_loss_BB': rec_loss_BB.detach(),
                    'recon_loss_BB_raw': rec_loss_BB_raw.detach(),
                    # 'recon_loss_BB_cs_1': rec_loss_BB_cs_1.detach(),
                    # 'recon_loss_BB_cs_2': rec_loss_BB_cs_2.detach(),
                    'recon_loss_BB_cs': rec_loss_BB_cs.detach(),
                    'trans_loss_A2B': trans_loss_A2B.detach(),
                    'trans_loss_B2A': trans_loss_B2A.detach(),
                }
                
                if epoch_nums >= warm_up_epoch:
                    losses_to_average.update({
                        'key_loss': key_loss.detach(),
                        'value_loss': value_loss.detach(),
                    })
                
                # for key in losses_to_average:
                #     losses_to_average[key] = losses_to_average[key].clone()
                #     torch.distributed.all_reduce(losses_to_average[key], op=torch.distributed.ReduceOp.SUM)
                #     losses_to_average[key] /= dist.get_world_size()
                
                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    keys = list(losses_to_average.keys())  # dict 插入顺序在各 rank 一致即可
                    buf = torch.stack([losses_to_average[k].detach().float().mean() for k in keys])  # [K]
                    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
                    buf /= dist.get_world_size()
                    losses_to_average = {k: buf[i] for i, k in enumerate(keys)}
                else:
                    # 单卡/未 init 分布式：保持 detach 即可
                    losses_to_average = {k: v.detach() for k, v in losses_to_average.items()}
                
                if dist.get_rank() == 0:
                    pbar.set_postfix({
                        'G_Loss': losses_to_average['g_loss'].item(),
                        'D_A_Loss': losses_to_average['d_A_loss'].item(),
                        'D_B_Loss': losses_to_average['d_B_loss'].item(),
                        'KL_Loss': losses_to_average['kl_loss'].item(),
                        'Recon_Loss': losses_to_average['recon_loss'].item(),
                        'Trans_loss': losses_to_average['trans_loss'].item(),
                    })
                    
                    if writer:
                        global_step = epoch_nums * len(train_loader) + batch_idx
                        
                        
                        writer.add_scalar('Train_loss/gengerator', losses_to_average['g_loss'].item(), global_step)
                        current_lr = scheduler_gen.get_last_lr()[0] if scheduler_gen else opt_gen.param_groups[0]['lr']
                        writer.add_scalar('Train_lr/gengerator', current_lr, epoch_nums)
                        
                        writer.add_scalars(
                            'Train_loss/l1_loss_parts',{
                                'recon_loss': losses_to_average['recon_loss'].item(),
                                'trans_loss': losses_to_average['trans_loss'].item(),
                            }, global_step
                        )
                        
                        writer.add_scalars(
                            'Train_loss/recon_trans_parts',{
                                'recon_loss_AA': losses_to_average['recon_loss_AA'].item(),
                                'recon_loss_BB': losses_to_average['recon_loss_BB'].item(),
                                'trans_loss_A2B': losses_to_average['trans_loss_A2B'].item(),
                                'trans_loss_B2A': losses_to_average['trans_loss_B2A'].item(),
                            }, global_step
                        )
                        writer.add_scalars(
                            'Train_loss/recon_parts_2',{
                                # 'recon_loss_AA_cs_1': losses_to_average['recon_loss_AA_cs_1'].item(),
                                # 'recon_loss_BB_cs_1': losses_to_average['recon_loss_BB_cs_1'].item(),
                                # 'recon_loss_AA_cs_2': losses_to_average['recon_loss_AA_cs_2'].item(),
                                # 'recon_loss_BB_cs_2': losses_to_average['recon_loss_BB_cs_2'].item(),
                                
                                
                                'recon_loss_AA_cs': losses_to_average['recon_loss_AA_cs'].item(),
                                'recon_loss_BB_cs': losses_to_average['recon_loss_BB_cs'].item(),
                                
                                
                                'recon_loss_AA_raw': losses_to_average['recon_loss_AA_raw'].item(),
                                'recon_loss_BB_raw': losses_to_average['recon_loss_BB_raw'].item(),
                            }, global_step
                        )
                        writer.add_scalar('Train_loss/gan_loss', losses_to_average['gan_loss'].item(), global_step)
                            
                        writer.add_scalars(
                            'Train_loss/gan_parts',{
                                'gan_A2B_loss': losses_to_average['gan_A2B_loss'].item(),
                                'gan_B2A_loss': losses_to_average['gan_B2A_loss'].item(),
                            }, global_step
                        )
                        
                        writer.add_scalar('Train_loss/perce_loss', losses_to_average['perce_loss'].item(), global_step)
                        writer.add_scalars(
                            'Train_loss/perce_parts',{
                                'perce_A2B_loss': losses_to_average['perce_A2B_loss'].item(),
                                'perce_B2A_loss': losses_to_average['perce_B2A_loss'].item(),
                            }, global_step
                        )
                        
                        writer.add_scalar('Train_loss/kl_loss', losses_to_average['kl_loss'].item(), global_step)
                        writer.add_scalars(
                            'Train_loss/kl_parts',{
                                'kl_A_loss': losses_to_average['kl_A_loss'].item(),
                                'kl_B_loss': losses_to_average['kl_B_loss'].item(),
                            }, global_step
                        )
                        
                        writer.add_scalars(
                            'Train_loss/kl_diff_parts',{
                                'kl_diff_mu_loss': losses_to_average['kl_diff_mu_loss'].item(),
                                'kl_diff_sigma_loss': losses_to_average['kl_diff_sigma_loss'].item(),
                            }, global_step
                        )
                        
                        writer.add_scalar('Train_loss/d_A_loss', losses_to_average['d_A_loss'].item(), global_step)
                        writer.add_scalar('Train_loss/d_B_loss', losses_to_average['d_B_loss'].item(), global_step)
                        writer.add_scalars(
                            'Train_loss/gan_A_parts',{
                                'gan_A_real_loss': losses_to_average['gan_A_real_loss'].item(),
                                'gan_A_fake_loss': losses_to_average['gan_A_fake_loss'].item(),
                            }, global_step
                        )
                        writer.add_scalars(
                            'Train_loss/gan_B_parts',{
                                'gan_B_real_loss': losses_to_average['gan_B_real_loss'].item(),
                                'gan_B_fake_loss': losses_to_average['gan_B_fake_loss'].item(),
                            }, global_step
                        )
                        
                        if epoch_nums >= warm_up_epoch:
                            writer.add_scalar('Train_loss/key_loss', losses_to_average['key_loss'].item(), global_step)
                            writer.add_scalar('Train_loss/value_loss', losses_to_average['value_loss'].item(), global_step)

            # --- epoch-end memory stats: DDP reduce + rank0 print ---
            if mem_pix_acc is not None and mem_hit_acc is not None:
                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    dist.all_reduce(mem_pix_acc, op=dist.ReduceOp.SUM)
                    dist.all_reduce(mem_hit_acc, op=dist.ReduceOp.SUM)

                if is_master():
                    offsets = [0]
                    for m in memory_module.memory_size:
                        offsets.append(offsets[-1] + int(m))

                    print(f"[MemStats] Epoch {epoch_nums} (after warmup)")
                    for ci, seg_id in enumerate(memory_module.seg_idx):
                        pix = float(mem_pix_acc[ci].item())
                        msize = int(memory_module.memory_size[ci])
                        sl = mem_hit_acc[offsets[ci]:offsets[ci + 1]]
                        used = int((sl > 0).sum().item())

                        if pix <= 0 or msize <= 0:
                            top1_share = 0.0
                            ent_norm = 0.0
                            eff_slots = 0.0
                        else:
                            top1_share = float(sl.max().item()) / (pix + 1e-12)
                            p = (sl / (pix + 1e-12))
                            pnz = p[p > 0]
                            ent = float(-(pnz * pnz.log()).sum().item()) if pnz.numel() > 0 else 0.0
                            ent_norm = ent / (math.log(msize) + 1e-12) if msize > 1 else 0.0
                            eff_slots = math.exp(ent)

                        print(
                            f"  seg={int(seg_id):>2}  pix={pix:>9.0f}  used={used:>3}/{msize:<3}  "
                            f"top1={top1_share:>.3f}  H={ent_norm:>.3f}  eff={eff_slots:>.2f}"
                        )
                            
            scheduler_gen.step() if scheduler_gen else None
            scheduler_dis_A.step() if scheduler_dis_A else None
            scheduler_dis_B.step() if scheduler_dis_B else None
            
            torch.distributed.barrier()
            generator.eval()
            disc_A.eval()
            disc_B.eval()
                        
                
            # --------------------------
            # 7) 验证循环
            # --------------------------
            
            val_a2a_l1_loss = 0.0
            val_b2b_l1_loss = 0.0
            val_a2b_l1_loss = 0.0
            val_b2a_l1_loss = 0.0 
            
            val_a2a_ssim_loss = 0.0
            val_b2b_ssim_loss = 0.0
            val_a2b_ssim_loss = 0.0
            val_b2a_ssim_loss = 0.0
            
            val_a2a_psnr_loss = 0.0
            val_b2b_psnr_loss = 0.0
            val_a2b_psnr_loss = 0.0
            val_b2a_psnr_loss = 0.0
            
            val_a2b_perce_loss = 0.0
            val_b2a_perce_loss = 0.0
            
            val_a2b_lpips_loss = 0.0
            val_b2a_lpips_loss = 0.0
            
            total_samples = 0
            
            
            val_sampler.set_epoch(epoch_nums)
            
            if dist.get_rank() == 0:
                pbar = tqdm(val_loader, total=len(val_loader), desc='Epoch %d'%epoch_nums, dynamic_ncols=True)
            else:
                pbar = val_loader
                
            with torch.no_grad():
                for batch_idx, (image_A, image_B, mask, _, _) in enumerate(pbar):
                    image_A = image_A.to(device, non_blocking=True)
                    image_B = image_B.to(device, non_blocking=True)
                    mask    = mask.to(device, non_blocking=True)   # [B, N, H, W]


                    batch_size = image_A.size(0)
                    total_samples += batch_size
                    
                    
                    # --------------------------
                    # 7.1) 编码得到两域分布（mu, sigma）
                    # --------------------------
                    z_mu_A,  z_sigma_A = generator.module.encode(image_A)  # [B,C,H,W] x2
                    z_mu_B,  z_sigma_B = generator.module.encode(image_B)

                    fake_z_mu_A2B, fake_z_sigma_A2B = build_mirror_gauss_sigma(z_mu_A, z_sigma_A)  # 伪镜像
                    fake_z_mu_B2A, fake_z_sigma_B2A = build_mirror_gauss_sigma(z_mu_B, z_sigma_B)

                    # --------------------------
                    # 7.2) 对两个镜像分布对进行两次的随机方向解耦然后进行双向跨域合成
                    # --------------------------

                    sample_fake_A2B_A = generator.module.sampling(z_mu_A, z_sigma_A)
                    sample_fake_A2B_B = generator.module.sampling(fake_z_mu_A2B, fake_z_sigma_A2B)
                    
                    sample_fake_B2A_B = generator.module.sampling(z_mu_B, z_sigma_B)
                    sample_fake_B2A_A = generator.module.sampling(fake_z_mu_B2A, fake_z_sigma_B2A)
                    
                    cont_fake_A2B, _ = cont_sampler(z_mu_A, fake_z_mu_A2B, z_sigma_A, fake_z_sigma_A2B, sample_fake_A2B_A, sample_fake_A2B_B)
                    style_fake_A2B_A, style_fake_A2B_B, _, _ = style_sampler(z_mu_A, fake_z_mu_A2B, z_sigma_A, fake_z_sigma_A2B, sample_fake_A2B_A, sample_fake_A2B_B)
                    
                    
                    
                    cont_fake_B2A, _ = cont_sampler(fake_z_mu_B2A, z_mu_B, fake_z_sigma_B2A, z_sigma_B, sample_fake_B2A_A, sample_fake_B2A_B)
                    style_fake_B2A_A, style_fake_B2A_B, _, _ = style_sampler(fake_z_mu_B2A, z_mu_B, fake_z_sigma_B2A, z_sigma_B, sample_fake_B2A_A, sample_fake_B2A_B)


                    if epoch_nums >= warm_up_epoch:
                        read_style_A2B_A, read_style_A2B_B = memory_module.forward_second(cont_fake_A2B, mask, current_memorys)
                        read_style_B2A_A, read_style_B2A_B = memory_module.forward_second(cont_fake_B2A, mask, current_memorys)
                        

                        val_rec_A2A = generator.module.decode_A_cs(cont_fake_A2B, read_style_A2B_A)
                        val_rec_B2B = generator.module.decode_B_cs(cont_fake_B2A, read_style_B2A_B)
                        val_trans_A2B = generator.module.decode_B_cs(cont_fake_A2B, read_style_A2B_B)
                        val_trans_B2A = generator.module.decode_A_cs(cont_fake_B2A, read_style_B2A_A)
                        
                    else:
                        # feature_fake_A2A = torch.concat([cont_fake_A2B, style_fake_A2B_A], dim=1)
                        # feature_fake_B2B = torch.concat([cont_fake_B2A, style_fake_B2A_B], dim=1)
                        # feature_fake_A2B = torch.concat([cont_fake_A2B, style_fake_A2B_B], dim=1)
                        # feature_fake_B2A = torch.concat([cont_fake_B2A, style_fake_B2A_A], dim=1)
                        
                        val_rec_A2A = generator.module.decode_A_cs(cont_fake_A2B, style_fake_A2B_A)
                        val_rec_B2B = generator.module.decode_B_cs(cont_fake_B2A, style_fake_B2A_B)
                        val_trans_A2B = generator.module.decode_B_cs(cont_fake_A2B, style_fake_A2B_B)
                        val_trans_B2A = generator.module.decode_A_cs(cont_fake_B2A, style_fake_B2A_A)

                    
                    a2a_rec_loss = intensity_loss(val_rec_A2A, image_A)
                    b2b_rec_loss = intensity_loss(val_rec_B2B, image_B)
                    a2b_trans_loss = intensity_loss(val_trans_A2B, image_B)
                    b2a_trans_loss = intensity_loss(val_trans_B2A, image_A)

                    a2b_perce_val_loss = perceptual_loss(val_trans_A2B.float(), image_B.float())
                    b2a_perce_val_loss = perceptual_loss(val_trans_B2A.float(), image_A.float())
                    
                    a2b_lpips_val_loss = lpips(val_trans_A2B.float(), image_B.float())
                    b2a_lpips_val_loss = lpips(val_trans_B2A.float(), image_A.float())
                    
                    image_A = tahn2sigmoid(image_A)
                    image_B = tahn2sigmoid(image_B)
                    val_rec_A2A = tahn2sigmoid(val_rec_A2A)
                    val_rec_B2B = tahn2sigmoid(val_rec_B2B)
                    val_trans_A2B = tahn2sigmoid(val_trans_A2B)
                    val_trans_B2A = tahn2sigmoid(val_trans_B2A)
                    
                    a2a_ssim = ssim(val_rec_A2A, image_A, data_range=1.0, size_average=True, win_size=11)
                    b2b_ssim = ssim(val_rec_B2B, image_B, data_range=1.0, size_average=True, win_size=11)
                    a2b_ssim = ssim(val_trans_A2B, image_B, data_range=1.0, size_average=True, win_size=11)
                    b2a_ssim = ssim(val_trans_B2A, image_A, data_range=1.0, size_average=True, win_size=11)
                    
                    
                    a2a_psnr = calculate_psnr(val_rec_A2A, image_A)
                    b2b_psnr = calculate_psnr(val_rec_B2B, image_B)
                    a2b_psnr = calculate_psnr(val_trans_A2B, image_B)
                    b2a_psnr = calculate_psnr(val_trans_B2A, image_A)
                    
                    val_a2a_l1_loss += a2a_rec_loss.item() * batch_size
                    val_b2b_l1_loss += b2b_rec_loss.item() * batch_size
                    val_a2b_l1_loss += a2b_trans_loss.item() * batch_size
                    val_b2a_l1_loss += b2a_trans_loss.item() * batch_size
                    
                    val_a2a_ssim_loss += a2a_ssim.item() * batch_size
                    val_b2b_ssim_loss += b2b_ssim.item() * batch_size
                    val_a2b_ssim_loss += a2b_ssim.item() * batch_size
                    val_b2a_ssim_loss += b2a_ssim.item() * batch_size
                    
                    val_a2a_psnr_loss += a2a_psnr.item() * batch_size
                    val_b2b_psnr_loss += b2b_psnr.item() * batch_size
                    val_a2b_psnr_loss += a2b_psnr.item() * batch_size
                    val_b2a_psnr_loss += b2a_psnr.item() * batch_size
                    
                    val_a2b_perce_loss += a2b_perce_val_loss.item() * batch_size
                    val_b2a_perce_loss += b2a_perce_val_loss.item() * batch_size
                    
                    val_a2b_lpips_loss += a2b_lpips_val_loss.mean().item() * batch_size
                    val_b2a_lpips_loss += b2a_lpips_val_loss.mean().item() * batch_size
                    
            
            metrics = torch.tensor([
                val_a2a_l1_loss,
                val_b2b_l1_loss,
                val_a2b_l1_loss,
                val_b2a_l1_loss,
                val_a2a_ssim_loss,
                val_b2b_ssim_loss,
                val_a2b_ssim_loss,
                val_b2a_ssim_loss,
                val_a2a_psnr_loss,
                val_b2b_psnr_loss,
                val_a2b_psnr_loss,
                val_b2a_psnr_loss,
                val_a2b_perce_loss,
                val_b2a_perce_loss,
                val_a2b_lpips_loss,
                val_b2a_lpips_loss,
            ]).to(device)
            
            
            # 汇总total_samples
            total_samples_tensor = torch.tensor(total_samples).to(device)
            torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM)
            total_samples = total_samples_tensor.item()
            
            
            # 提取指标
            torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
            
            (
                val_a2a_l1_loss,
                val_b2b_l1_loss,
                val_a2b_l1_loss,
                val_b2a_l1_loss,
                val_a2a_ssim_loss,
                val_b2b_ssim_loss,
                val_a2b_ssim_loss,
                val_b2a_ssim_loss,
                val_a2a_psnr_loss,
                val_b2b_psnr_loss,
                val_a2b_psnr_loss,
                val_b2a_psnr_loss,
                val_a2b_perce_loss,
                val_b2a_perce_loss,
                val_a2b_lpips_loss,
                val_b2a_lpips_loss,
            ) = metrics.tolist()
            
            # 计算平均值
            avg_val_a2a_l1_loss = val_a2a_l1_loss / total_samples
            avg_val_b2b_l1_loss = val_b2b_l1_loss / total_samples
            avg_val_a2b_l1_loss = val_a2b_l1_loss / total_samples
            avg_val_b2a_l1_loss = val_b2a_l1_loss / total_samples
            
            avg_val_a2a_ssim = val_a2a_ssim_loss / total_samples
            avg_val_b2b_ssim = val_b2b_ssim_loss / total_samples
            avg_val_a2b_ssim = val_a2b_ssim_loss / total_samples
            avg_val_b2a_ssim = val_b2a_ssim_loss / total_samples
            
            avg_val_a2a_psnr = val_a2a_psnr_loss / total_samples
            avg_val_b2b_psnr = val_b2b_psnr_loss / total_samples
            avg_val_a2b_psnr = val_a2b_psnr_loss / total_samples
            avg_val_b2a_psnr = val_b2a_psnr_loss / total_samples
            
            avg_val_a2b_perce_loss = val_a2b_perce_loss / total_samples
            avg_val_b2a_perce_loss = val_b2a_perce_loss / total_samples
            
            avg_val_a2b_lpips_loss = val_a2b_lpips_loss / total_samples
            avg_val_b2a_lpips_loss = val_b2a_lpips_loss / total_samples
            
            if writer and dist.get_rank() == 0:
                writer.add_scalars(
                    'Val_loss/l1_loss_parts',{
                        'A2A_l1_loss': avg_val_a2a_l1_loss,
                        'B2B_l1_loss': avg_val_b2b_l1_loss,
                        'A2B_l1_loss': avg_val_a2b_l1_loss,
                        'B2A_l1_loss': avg_val_b2a_l1_loss,
                    }, epoch_nums
                )
                
                writer.add_scalars(
                    'Val_loss/ssim_parts',{
                        'A2A_ssim': avg_val_a2a_ssim,
                        'B2B_ssim': avg_val_b2b_ssim,
                        'A2B_ssim': avg_val_a2b_ssim,
                        'B2A_ssim': avg_val_b2a_ssim,
                    }, epoch_nums
                )
                
                writer.add_scalars(
                    'Val_loss/psnr_parts',{
                        'A2A_psnr': avg_val_a2a_psnr,
                        'B2B_psnr': avg_val_b2b_psnr,
                        'A2B_psnr': avg_val_a2b_psnr,
                        'B2A_psnr': avg_val_b2a_psnr,
                    }, epoch_nums
                )
                
                writer.add_scalars(
                    'Val_loss/perce_parts',{
                        'A2B_perce_loss': avg_val_a2b_perce_loss,
                        'B2A_perce_loss': avg_val_b2a_perce_loss,
                    }, epoch_nums
                )
                
                writer.add_scalars(
                    'Val_loss/lpips_parts',{
                        'A2B_lpips_loss': avg_val_a2b_lpips_loss,
                        'B2A_lpips_loss': avg_val_b2a_lpips_loss,
                    }, epoch_nums
                )
            if avg_val_a2b_psnr > best_a2b_psnr:
                best_a2b_psnr = avg_val_a2b_psnr
                
                if dist.get_rank() == 0:
                    print(f'New best A2B PSNR: {best_a2b_psnr:.4f} at epoch {epoch_nums}, saving model...')
                    torch.save(generator.module.state_dict(), os.path.join(model_save_path, 'best_generator_%d.pth'%epoch_nums))
                    torch.save(disc_A.module.state_dict(), os.path.join(model_save_path, 'best_disc_A_%d.pth'%epoch_nums))
                    torch.save(disc_B.module.state_dict(), os.path.join(model_save_path, 'best_disc_B_%d.pth'%epoch_nums))
                    torch.save(cont_sampler.module.state_dict(), os.path.join(model_save_path, 'best_cont_sampler_%d.pth'%epoch_nums))
                    torch.save(style_sampler.module.state_dict(), os.path.join(model_save_path, 'best_style_sampler_%d.pth'%epoch_nums))
                    torch.save([(k.detach().cpu(), va.detach().cpu(), vb.detach().cpu()) for (k,va,vb) in current_memorys],
                                os.path.join(model_save_path, f'best_memory_list_{epoch_nums}.pt'))
                    
                    torch.save({
                        'epoch': epoch_nums,
                        'best_a2b_psnr': best_a2b_psnr,

                        'generator': generator.module.state_dict(),
                        'disc_A': disc_A.module.state_dict(),
                        'disc_B': disc_B.module.state_dict(),
                        'cont_sampler': cont_sampler.module.state_dict(),
                        'style_sampler': style_sampler.module.state_dict(),

                        'opt_gen': opt_gen.state_dict(),
                        'opt_dis_A': opt_dis_A.state_dict(),
                        'opt_dis_B': opt_dis_B.state_dict(),
                        'opt_cont_sampler': opt_cont_sampler.state_dict(),
                        'opt_style_sampler': opt_style_sampler.state_dict(),

                        'sch_gen': scheduler_gen.state_dict() if scheduler_gen else None,
                        'sch_dis_A': scheduler_dis_A.state_dict() if scheduler_dis_A else None,
                        'sch_dis_B': scheduler_dis_B.state_dict() if scheduler_dis_B else None,

                        'memory_list': [(k.detach().cpu(), va.detach().cpu(), vb.detach().cpu())
                                        for (k, va, vb) in current_memorys],
                    }, os.path.join(model_save_path, 'latest_checkpoint/checkpoint_best.pth'))
                
            if epoch_nums % model_save_interval == 0 and epoch_nums != 0 and dist.get_rank() == 0:
                print('Saving model at epoch:', epoch_nums)
                torch.save(generator.module.state_dict(), os.path.join(model_save_path, 'generator_%d.pth'%epoch_nums))
                torch.save(disc_A.module.state_dict(), os.path.join(model_save_path, 'disc_A_%d.pth'%epoch_nums))
                torch.save(disc_B.module.state_dict(), os.path.join(model_save_path, 'disc_B_%d.pth'%epoch_nums))
                torch.save(cont_sampler.module.state_dict(), os.path.join(model_save_path, 'cont_sampler_%d.pth'%epoch_nums))
                torch.save(style_sampler.module.state_dict(), os.path.join(model_save_path, 'style_sampler_%d.pth'%epoch_nums))
                torch.save([(k.detach().cpu(), va.detach().cpu(), vb.detach().cpu()) for (k,va,vb) in current_memorys],
                            os.path.join(model_save_path, f'memory_list_{epoch_nums}.pt'))
                



                
                
        if dist.is_initialized():
                dist.destroy_process_group() 
    
    except KeyboardInterrupt:
        if epoch_nums >= 10 and dist.get_rank() == 0:
            print('Training has been stopped at epoch:', epoch_nums)
            print('Saving model')
            torch.save(generator.module.state_dict(), os.path.join(model_save_path, 'generator_%d.pth'%epoch_nums))
            torch.save(disc_A.module.state_dict(), os.path.join(model_save_path, 'disc_A_%d.pth'%epoch_nums))
            torch.save(disc_B.module.state_dict(), os.path.join(model_save_path, 'disc_B_%d.pth'%epoch_nums))
            torch.save(cont_sampler.module.state_dict(), os.path.join(model_save_path, 'cont_sampler_%d.pth'%epoch_nums))
            torch.save(style_sampler.module.state_dict(), os.path.join(model_save_path, 'style_sampler_%d.pth'%epoch_nums))
            torch.save([(k.detach().cpu(), va.detach().cpu(), vb.detach().cpu()) for (k,va,vb) in current_memorys],
                        os.path.join(model_save_path, f'memory_list_{epoch_nums}.pt'))

            if dist.is_initialized():
                dist.destroy_process_group()
            
        else:
            if dist.get_rank() == 0:
                print('Training has been stopped at epoch:', epoch_nums)
            if dist.is_initialized():
                dist.destroy_process_group() 


if __name__ == '__main__':
    
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device = setup_distributed()
    lr = 0.0001
    batch_size = 5
    num_workers = 32
    resume_latest = False
    
    
    
    phase_1 = "pre"
    phase_2 = "c_a"
    
    data_path = ""
    lesion_patient_file = "./lesion_patient_list.txt"
    
    
    model_save_path = './model_weight/'+ phase_1 + '2' + phase_2 + '_No_memory' +'/'
    checkpoint_save_path = os.path.join(model_save_path, 'latest_checkpoint')
    os.makedirs(checkpoint_save_path, exist_ok=True)

    
    if dist.get_rank() == 0:
        writer = SummaryWriter()
    else:
        writer = None
    
    val_ratio = 0.2
    image_size = (256, 256)
    random_seed = 42
    
    train_dataset = PairMRIDataset(
        data_path=data_path,
        lesion_patient_file=lesion_patient_file,
        split='train',
        val_ratio=0.2,
        image_size=image_size,
        random_seed=42,
        phase_1=phase_1,
        phase_2=phase_2,
        roi_aug_prob=0
    )
    
    val_dataset = PairMRIDataset(
        data_path=data_path,
        lesion_patient_file=lesion_patient_file,
        split='val',
        val_ratio=0.2,
        image_size=image_size,
        random_seed=42,
        phase_1=phase_1,
        phase_2=phase_2
    )
    
    train_sampler = DSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, 
                            batch_size = batch_size, 
                            sampler = train_sampler,
                            shuffle = False,
                            num_workers =num_workers,
                            pin_memory = True,
                            drop_last=True
                            )
    val_sampler = DSampler(val_dataset, shuffle=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            sampler=val_sampler,
                            shuffle=False, 
                            num_workers=num_workers, 
                            pin_memory = True
                            )

    adv_loss = PatchAdversarialLoss(criterion="bce")
    lpips = LPIPS(net='alex')
    lpips = lpips.to(device)
    
    sty_channel = 64
    content_channel = 128
    
    generator = AutoencoderKL(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        latent_channels=content_channel,
        num_channels=[64, 128, 256],
        num_res_blocks=2,
        norm_num_groups=32,
        norm_eps=1e-6,
        attention_levels=[False, False, False],
        with_encoder_nonlocal_attn=True,
        with_decoder_nonlocal_attn=True,
        use_convtranspose=True,
        use_checkpointing=False,
        use_flash_attention=True,  # True 需 CUDA + xformers
        # ------ big-kernel (optional) ------
        stem_kernel=3,
        use_big_kernels=False,
        big_kernel_levels=[False, False, False],   # 前两层用大核
        big_kernel_size=[3, 3, 3],               # 对应每层大核尺寸
        big_kernel_blocks=[0, 0, 0],             # 每层前1个 ResBlock 用大核
        # ---------- 新增：风格注入相关 ----------
        sty_ch=sty_channel,
    ).to(device)

    # per_class_slots = 256
    # memory_size_list, memory_list = init_memory_list(
    #     seg_idx=seg_idx_1, kdim=96, vdim=sty_channel, memory_slots=per_class_slots, device=device, seed=42
    # )
    # memory_slots = [20, 8, 10, 12, 8, 10, 8, 6, 6, 6, 6, 6, 6, 6, 6]
    # memory_slots = [40,16,12,24,10,10,8,12,12,12,12,12,12,12,12]
    # memory_slots = [100,32,24,48,20,20,16,24,12,12,12,12,12,12,24]
    # memory_slots = [10,5,5,6,4,4,3,2,2,2,2,2,2,2,2]
    memory_slots = [2, 20, 8, 8, 16, 8, 8, 6, 4, 2,2,2,2,2,2,3]
    memory_size_list, memory_list = init_memory_list(
        seg_idx=SEG_IDX_V7, kdim=content_channel, vdim=sty_channel, memory_slots=memory_slots, device=device, seed=42
    )


    # memory_module = MemorySharedParts(
    #     memory_size=memory_size_list,
    #     kdim = 96,
    #     vdim = sty_channel,
    #     seg_idx=seg_idx_1
    # ).to(device)
    
    memory_module = MemorySharedPartsV7(
        memory_size=memory_size_list,
        kdim = content_channel,
        vdim = sty_channel,
        seg_idx=SEG_IDX_V7
    ).to(device)
    
    
    cont_sampler = cont_sample(in_ch=content_channel, out_ch=content_channel, norm_num_groups = 32).to(device)
    style_sampler = style_sample(in_ch=content_channel, out_ch=sty_channel, norm_num_groups = 2).to(device)
    
    
    
    disc_A = PatchDiscriminator(
    spatial_dims=2,
    num_layers_d=3,
    num_channels=32,
    in_channels=1,
    out_channels=1,
    norm="INSTANCE"
    ).to(device)

    disc_B = PatchDiscriminator(
    spatial_dims=2,
    num_layers_d=3,
    num_channels=32,
    in_channels=1,
    out_channels=1,
    norm="INSTANCE"
    ).to(device)
    
    perceptual_loss = PerceptualLoss(
    spatial_dims=2,
    network_type="resnet50",
    is_fake_3d=True,
    fake_3d_ratio=0.2,
    pretrained=False,
    pretrained_path=None,
    pretrained_state_dict_key="state_dict"
    ).to(device)
    

    generator = DDP(generator, device_ids=[device], find_unused_parameters=True)
    disc_A = DDP(disc_A, device_ids = [device], find_unused_parameters=False)
    disc_B = DDP(disc_B, device_ids = [device], find_unused_parameters=False)
    cont_sampler = DDP(cont_sampler, device_ids = [device], find_unused_parameters=False)
    style_sampler = DDP(style_sampler, device_ids = [device], find_unused_parameters=False)
    # memory_module = DDP(memory_module, device_ids = [device], find_unused_parameters=False)
    
    
    optimizer_gen = optim.Adam(
        generator.parameters(), 
        lr=lr, 
        betas=(0.5, 0.999)
    )
    optimizer_disc_ct = optim.Adam(
        disc_A.parameters(), 
        lr=lr, 
        betas=(0.5, 0.999)
    )
    
    optimizer_disc_mr = optim.Adam(
        disc_B.parameters(), 
        lr=lr, 
        betas=(0.5, 0.999)
    )
    
    optimizer_cont_sampler = optim.Adam(
        cont_sampler.parameters(), 
        lr=lr, 
        betas=(0.5, 0.999)
    )
    
    optimizer_style_sampler = optim.Adam(
        style_sampler.parameters(), 
        lr=lr, 
        betas=(0.5, 0.999)
    )
    
    def lr_lambda(epoch):
        if epoch < 15:
            return 1.0
        # elif epoch < 30:
        #     return 0.5
        else:
            return 1.0
        
    scheduler_gen = optim.lr_scheduler.LambdaLR(optimizer_gen, lr_lambda=lr_lambda)
    scheduler_dis_A = optim.lr_scheduler.LambdaLR(optimizer_disc_ct, lr_lambda=lr_lambda)
    scheduler_dis_B = optim.lr_scheduler.LambdaLR(optimizer_disc_mr, lr_lambda=lr_lambda)
    
    # -------- after resume: override lr (optional) --------
    override_lr_after_resume = True  # True: 读完ckpt后强制用当前脚本 lr；False: 完全沿用ckpt里的lr

    def _set_optimizer_lr(optimizer, new_lr: float):
        for pg in optimizer.param_groups:
            pg["lr"] = float(new_lr)

    def _sync_scheduler_base_lrs(scheduler, new_lr: float):
        # 避免 scheduler 继续用 ckpt 的 base_lrs
        if scheduler is None:
            return
        if hasattr(scheduler, "base_lrs") and scheduler.base_lrs is not None:
            scheduler.base_lrs = [float(new_lr) for _ in scheduler.base_lrs]
    
    
    start_epoch = 0
    init_best_a2b_psnr = 0.0

    checkpoint_path = '/root/autodl-tmp/Mark_02/FDA_VAE_v2.1_Style_inster/model_weight/pre2c_a_half/latest_checkpoint/checkpoint_best.pth'
    
    if resume_latest and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)

        generator.module.load_state_dict(ckpt['generator'])
        disc_A.module.load_state_dict(ckpt['disc_A'])
        disc_B.module.load_state_dict(ckpt['disc_B'])
        cont_sampler.module.load_state_dict(ckpt['cont_sampler'])
        style_sampler.module.load_state_dict(ckpt['style_sampler'])

        optimizer_gen.load_state_dict(ckpt['opt_gen'])
        optimizer_disc_ct.load_state_dict(ckpt['opt_dis_A'])
        optimizer_disc_mr.load_state_dict(ckpt['opt_dis_B'])
        optimizer_cont_sampler.load_state_dict(ckpt['opt_cont_sampler'])
        optimizer_style_sampler.load_state_dict(ckpt['opt_style_sampler'])

        if ckpt.get('sch_gen') is not None:
            scheduler_gen.load_state_dict(ckpt['sch_gen'])
        if ckpt.get('sch_dis_A') is not None:
            scheduler_dis_A.load_state_dict(ckpt['sch_dis_A'])
        if ckpt.get('sch_dis_B') is not None:
            scheduler_dis_B.load_state_dict(ckpt['sch_dis_B'])

        # ---- override lr AFTER loading optimizer/scheduler states ----
        if override_lr_after_resume:
            if dist.get_rank() == 0:
                old_lrs = {
                    "gen": optimizer_gen.param_groups[0].get("lr", None),
                    "disc_A": optimizer_disc_ct.param_groups[0].get("lr", None),
                    "disc_B": optimizer_disc_mr.param_groups[0].get("lr", None),
                    "cont": optimizer_cont_sampler.param_groups[0].get("lr", None),
                    "style": optimizer_style_sampler.param_groups[0].get("lr", None),
                }
                print(f"[LR override] before={old_lrs}, set lr={lr}")

            _set_optimizer_lr(optimizer_gen, lr)
            _set_optimizer_lr(optimizer_disc_ct, lr)
            _set_optimizer_lr(optimizer_disc_mr, lr)
            _set_optimizer_lr(optimizer_cont_sampler, lr)
            _set_optimizer_lr(optimizer_style_sampler, lr)

            _sync_scheduler_base_lrs(scheduler_gen, lr)
            _sync_scheduler_base_lrs(scheduler_dis_A, lr)
            _sync_scheduler_base_lrs(scheduler_dis_B, lr)

        memory_list = [(k.to(device), va.to(device), vb.to(device))
                       for (k, va, vb) in ckpt['memory_list']]
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        init_best_a2b_psnr = float(ckpt.get('best_a2b_psnr', 0.0))

        if dist.get_rank() == 0:
            print(f"Resume from {checkpoint_path}, start_epoch={start_epoch}, best_a2b_psnr={init_best_a2b_psnr:.4f}")
            print(f"[LR override] after gen_lr={optimizer_gen.param_groups[0]['lr']}")
    
    train_FDA_V2(
        train_loader = train_loader, 
        train_sampler = train_sampler,
        cont_sampler = cont_sampler,
        style_sampler = style_sampler,
        val_loader = val_loader,
        val_sampler = val_sampler,
        generator = generator,
        disc_A = disc_A,
        disc_B = disc_B,
        cont_disc = None,
        adv_loss = adv_loss,
        memory_module = memory_module,
        memory_list = memory_list,                       # 记忆库列表（外部维护）
        opt_gen = optimizer_gen,
        opt_dis_A = optimizer_disc_ct,
        opt_dis_B = optimizer_disc_mr,
        opt_cont_sampler=optimizer_cont_sampler,
        opt_style_sampler=optimizer_style_sampler,
        scheduler_gen = scheduler_gen,
        scheduler_dis_A = scheduler_dis_A,
        scheduler_dis_B = scheduler_dis_B,
        perceptual_loss = perceptual_loss,
        lpips = lpips,
        d_train_freq=1,
        device = device,
        Max_Epoch=1501,
        loss_cfg=None,
        warm_up_epoch=1000,
        writer = writer,
        model_save_path = model_save_path,
        model_save_interval=10,
    )