import torch
import torch.nn.functional as F
import torch.nn as nn
from monai.networks.blocks import Convolution
import time 

def content_sampler(muA, muB, sigmaA, sigmaB, sampA, sampB, clip: bool = False, eps: float = 1e-8):
    """
    Args:
        muA, muB, sigmaA, sigmaB, sampA, sampB: [B, C, H, W]  (sigma 为 std)
    Returns:
        content_point: [B, C, H, W]，线段 [sampA, sampB] 上使 pA(x)*pB(x) 最大的点
        t_opt        : [B, 1, H, W]，对应每像素的标量 t
    备注：通道 C 视作特征维，所有二次型均在 dim=1 上归约，确保每像素一个 t。
    """
    assert muA.shape == muB.shape == sigmaA.shape == sigmaB.shape == sampA.shape == sampB.shape, \
        f"shape mismatch: {muA.shape}, {muB.shape}, {sigmaA.shape}, {sigmaB.shape}, {sampA.shape}, {sampB.shape}"
    assert muA.dim() == 4, f"expect [B,C,H,W], got {muA.shape}"

    d = sampB - sampA                      # [B,C,H,W]
    a = sampA                              # [B,C,H,W]
    inv1 = 1.0 / (sigmaA * sigmaA + eps)   # [B,C,H,W]
    inv2 = 1.0 / (sigmaB * sigmaB + eps)   # [B,C,H,W]

    # 在通道维 (dim=1) 归约，得到每像素的标量系数
    a_coef = (d * d * (inv1 + inv2)).sum(dim=1, keepdim=True)                          # [B,1,H,W]
    b_coef = 2.0 * (d * ((a - muA) * inv1 + (a - muB) * inv2)).sum(dim=1, keepdim=True)# [B,1,H,W]

    t_opt = (-b_coef / (2.0 * a_coef + eps))                                           # [B,1,H,W]
    if clip:
        t_opt = t_opt.clamp(0.0, 1.0)

    content_point = a + t_opt * d                                                      # [B,C,H,W]
    return content_point, t_opt


def style_sampler_softargmax(
    muA, muB, sigmaA, sigmaB, sampA, sampB,
    G: int = 2049,           # 网格点数
    T_factor: float = 3.0,  # 搜索半径系数：最终区间为 [-T, T]，T 会随方差尺度自适应
    tau: float = 0.15,      # soft-argmax 温度
    eps: float = 1e-8):
    
    
    """
    输入:  muA, muB, sigmaA, sigmaB, sampA, sampB 皆为 [B, C, H, W]（C=特征维）
    输出:  style_A, style_B : [B, C, H, W]
          tA, tB            : [B, 1, H, W]（每像素一个标量 t）

    与 test35 的软版一致的关键点：
      1) 直线参数化采用 x(t) = m + t·v，其中 v 是单位方向，m 是中点；
      2) 搜索区间为对称的 [-T, T]，T 随“沿 v 方向”的标准差自适应（两域取 max 后乘以 T_factor）；
      3) 打分采用“相对偏好” pref(t) = pA(t) - pB(t) 的数值稳定形式（完整使用常数项），
         然后用 softmax(pref/τ) 与 softmax(-pref/τ) 计算两侧的 soft-argmax / soft-argmin。
    """
    assert muA.shape == muB.shape == sigmaA.shape == sigmaB.shape == sampA.shape == sampB.shape, \
        f"shape mismatch: {muA.shape}, {muB.shape}, {sigmaA.shape}, {sigmaB.shape}, {sampA.shape}, {sampB.shape}"
    assert muA.dim() == 4, f"expect [B,C,H,W], got {muA.shape}"

    # 1) 直线 x(t) = m + t·v（v 为单位向量）
    d = sampB - sampA                                            # [B,C,H,W]
    d_norm = torch.sqrt((d * d).sum(dim=1, keepdim=True) + eps)  # [B,1,H,W]
    v = d / d_norm                                               # [B,C,H,W]

    # 退化兜底（d≈0）：给固定单位方向，仅在退化像素生效
    deg_mask = (d_norm < 1e-8)
    if deg_mask.any().item():
        v_fallback = torch.zeros_like(v)
        v_fallback[:, :1, :, :] = 1.0
        v = torch.where(deg_mask, v_fallback, v)

    m = 0.5 * (sampA + sampB)                                    # [B,C,H,W]

    # 2) 二次型系数（对角协方差 Λ=diag(1/σ^2)），在通道维归约得到标量场
    invA = 1.0 / (sigmaA * sigmaA + eps)                         # [B,C,H,W]
    invB = 1.0 / (sigmaB * sigmaB + eps)
    diffA = (m - muA)                                            # [B,C,H,W]
    diffB = (m - muB)

    a2_A = (v * v * invA).sum(dim=1, keepdim=True)               # [B,1,H,W]
    a1_A = 2.0 * (v * diffA * invA).sum(dim=1, keepdim=True)
    a0_A = (diffA * diffA * invA).sum(dim=1, keepdim=True)

    a2_B = (v * v * invB).sum(dim=1, keepdim=True)
    a1_B = 2.0 * (v * diffB * invB).sum(dim=1, keepdim=True)
    a0_B = (diffB * diffB * invB).sum(dim=1, keepdim=True)

    # 3) 自适应搜索半径：T = T_factor * max(std_A_along_v, std_B_along_v)
    #    其中 std_along_v = sqrt( sum_c v_c^2 * sigma_c^2 )
    stdA_dir = torch.sqrt((v * v * (sigmaA * sigmaA)).sum(dim=1, keepdim=True) + eps)  # [B,1,H,W]
    stdB_dir = torch.sqrt((v * v * (sigmaB * sigmaB)).sum(dim=1, keepdim=True) + eps)  # [B,1,H,W]
    T = T_factor * torch.maximum(stdA_dir, stdB_dir)                                   # [B,1,H,W]

    t_lin  = torch.linspace(-1.0, 1.0, steps=G, device=muA.device, dtype=muA.dtype)    # [G]
    t_grid = T.unsqueeze(-1) * t_lin.view(1, 1, 1, 1, G)                               # [B,1,H,W,G]

    # 4) 稳定的相对偏好：pref(t) = pA(t) - pB(t)，保留常数项
    #    q(t) = a2 t^2 + a1 t + a0, l = -0.5 q
    a2_Ae, a1_Ae, a0_Ae = a2_A.unsqueeze(-1), a1_A.unsqueeze(-1), a0_A.unsqueeze(-1)   # [B,1,H,W,1]
    a2_Be, a1_Be, a0_Be = a2_B.unsqueeze(-1), a1_B.unsqueeze(-1), a0_B.unsqueeze(-1)

    t2 = t_grid * t_grid
    qA = a2_Ae * t2 + a1_Ae * t_grid + a0_Ae                                           # [B,1,H,W,G]
    qB = a2_Be * t2 + a1_Be * t_grid + a0_Be

    lA = -0.5 * qA
    lB = -0.5 * qB
    lmax = torch.maximum(lA, lB)
    pA_ = torch.exp(lA - lmax)
    pB_ = torch.exp(lB - lmax)
    pref = pA_ - pB_                                                                    # [B,1,H,W,G]

    # 5) Soft-argmax / Soft-argmin 得到期望 tA / tB（与 test35 一致）
    tau = max(tau, 1e-6)
    wA = F.softmax(pref / tau, dim=-1)                                                  # [B,1,H,W,G]
    wB = F.softmax((-pref) / tau, dim=-1)

    # tA = (wA * t_grid).sum(dim=-1)                                                      # [B,1,H,W]
    # tB = (wB * t_grid).sum(dim=-1)                                                      # [B,1,H,W]

    # # 6) 复原风格点：x = m + t·v
    # style_A = m + v * tA                                                                # [B,C,H,W]
    # style_B = m + v * tB                                                                # [B,C,H,W]

    # return style_A, style_B, tA, tB
    tA = (wA * t_grid).sum(dim=-1)  # 更偏 A 的极值（pref 最大侧）
    tB = (wB * t_grid).sum(dim=-1)  # 更偏 B 的极值（pref 最小侧）

    style_plus  = m + v * tA        # 候选点1
    style_minus = m + v * tB        # 候选点2

    # Keep the original RDFD relative-preference assignment.
    style_A = style_plus
    style_B = style_minus

    # The following A-only likelihood reassignment was evaluated but is not
    # used because it can reverse otherwise consistent relative preferences.
    # llA_plus = -0.5 * ((style_plus - muA)**2 / (sigmaA**2 + eps)).sum(dim=1, keepdim=True)
    # llA_minus = -0.5 * ((style_minus - muA)**2 / (sigmaA**2 + eps)).sum(dim=1, keepdim=True)
    # mask_A = (llA_plus >= llA_minus)
    # style_A = torch.where(mask_A, style_plus, style_minus)
    # style_B = torch.where(mask_A, style_minus, style_plus)
    
    return style_A, style_B, tA, tB




class cont_sample(nn.Module):
    def __init__(self, clip: bool = False, eps: float = 1e-8, in_ch: int = 256, out_ch: int = 256, norm_num_groups: int = 16):
        super().__init__()
        self.clip = clip
        self.eps = eps
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.cont_proj = Convolution(
            spatial_dims=2, in_channels=in_ch, out_channels=out_ch,
            strides=1, kernel_size=1, padding=0, conv_only=True,
        )
        self.cont_norm = nn.GroupNorm(norm_num_groups, out_ch, eps=1e-6, affine=True)   # 也可 IN2d
        self.cont_act  = nn.SiLU()

    def forward(self, muA, muB, sigmaA, sigmaB, sampA, sampB):
        continuet_point, t_opt = content_sampler(
            muA, muB, sigmaA, sigmaB, sampA, sampB, clip=self.clip, eps=self.eps
        )
        continuet_point = self.cont_proj(continuet_point)
        continuet_point = self.cont_norm(continuet_point)
        continuet_point = self.cont_act(continuet_point)
        return continuet_point, t_opt
    
class style_sample(nn.Module):
    def __init__(self, in_ch: int = 256, out_ch: int = 256, norm_num_groups: int = 16):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        # Conv -> Norm -> Act
        self.styA_norm = nn.GroupNorm(norm_num_groups, out_ch, eps=1e-6, affine=True)
        self.styB_norm = nn.GroupNorm(norm_num_groups, out_ch, eps=1e-6, affine=True)
        self.sty_A_proj = Convolution(
            spatial_dims=2, in_channels=in_ch, out_channels=out_ch,
            strides=1, kernel_size=1, padding=0, conv_only=True,
        )
        self.sty_B_proj = Convolution(
            spatial_dims=2, in_channels=in_ch, out_channels=out_ch,
            strides=1, kernel_size=1, padding=0, conv_only=True,
        )

    def forward(self, muA, muB, sigmaA, sigmaB, sampA, sampB):
        style_A, style_B, tA, tB = style_sampler_softargmax(
            muA, muB, sigmaA, sigmaB, sampA, sampB)

        style_A = self.sty_A_proj(style_A)
        style_A = self.styA_norm(style_A)
        style_A = F.silu(style_A)  # 非原地

        style_B = self.sty_B_proj(style_B)
        style_B = self.styB_norm(style_B)
        style_B = F.silu(style_B)

        # 可选：对通道维做 L2 归一化，有利于相似度检索/记忆匹配的稳定性
        # style_A = F.normalize(style_A, p=2, dim=1)
        # style_B = F.normalize(style_B, p=2, dim=1)

        return style_A, style_B, tA, tB


def _sync(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    
    
if __name__ == '__main__':
    device = 'mps'
    B, C, H, W = 4, 32, 64, 64
    muA = torch.randn(B, C, H, W, device=device)
    muB = torch.randn(B, C, H, W, device=device)
    sigmaA= torch.randn(B, C, H, W, device=device)
    sigmaB = torch.randn(B, C, H, W, device=device)
    epsA = torch.randn_like(muA)
    epsB = torch.randn_like(muB)
    sampleA = muA + sigmaA * epsA
    sampleB = muB + sigmaB * epsB

    cont_sampler = cont_sample(in_ch=C, out_ch=64).to(device)
    style_sampler = style_sample(in_ch=C, out_ch=48).to(device)

    # 预热（去除首次 kernel / 内存分配开销）
    for _ in range(2):
        _ = cont_sampler(muA, muB, sigmaA, sigmaB, sampleA, sampleB)
        _ = style_sampler(muA, muB, sigmaA, sigmaB, sampleA, sampleB)
    # 单次测时
    _sync(device)
    t0 = time.perf_counter()
    cont, t_opt = cont_sampler(muA, muB, sigmaA, sigmaB, sampleA, sampleB)
    _sync(device)
    t1 = time.perf_counter()
    style_A, style_B, tA, tB = style_sampler(muA, muB, sigmaA, sigmaB, sampleA, sampleB)
    _sync(device)
    t2 = time.perf_counter()
    print(f'[single] content={ (t1-t0)*1000:.2f} ms  style={ (t2-t1)*1000:.2f} ms  total={ (t2-t0)*1000:.2f} ms')

    # 多次平均
    repeats = 10
    c_times, s_times = [], []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        _ = cont_sampler(muA, muB, sigmaA, sigmaB, sampleA, sampleB)
        _sync(device)
        t1 = time.perf_counter()
        _ = style_sampler(muA, muB, sigmaA, sigmaB, sampleA, sampleB)
        _sync(device)
        t2 = time.perf_counter()
        c_times.append((t1 - t0) * 1000)
        s_times.append((t2 - t1) * 1000)
    print(f'[avg {repeats}] content={sum(c_times)/repeats:.2f} ms  style={sum(s_times)/repeats:.2f} ms  total={(sum(c_times)+sum(s_times))/repeats:.2f} ms')

    print('shapes:', cont.shape, t_opt.shape, style_A.shape, style_B.shape)
