import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# major organs + lesions + abdomen-unlabeled + black-border
# ----------------------------

# 说明：
# - 51..57: lesion classes (highest priority)
# - organs: subset you care about (next priority)
# - 60: abdomen-unlabeled background INSIDE body but not in selected organs and not lesion
# - 0: black-border (outside body, abdomen==0)  <-- 新增单独一类
#
# 互斥规则：lesion > organ > abdomen-unlabeled(60) > black-border(0)

LESION_IDS = [51, 52, 53, 54, 55, 56, 57]
ABD_UNLABELED_ID = 60
BLACK_BG_ID = 0

# 你的 major organs (canonical ids)
# 注意：这些 canonical id 会通过 SEG_GROUPS_MAJOR 映射到 raw organ label ids
ORG_CANON_IDS = [1, 2, 5, 6, 10, 19, 22]

SEG_GROUPS_MAJOR = {
    1:  [1],          # spleen
    2:  [2, 3],       # kidneys (L/R)
    5:  [5],          # liver (parenchyma = liver minus lesion)
    6:  [6],          # stomach
    10: [10, 11],     # lungs (L/R)
    19: [19, 20, 21], # spine + cord
    22: [22],         # heart
}

# seg_idx 的顺序决定 memory_list 的顺序：必须和 init_memory_list 一致
# 这里给出 16 类：black-bg(0) + organs(7) + lesions(7) + abdomen-unlabeled(60) 共 1+7+7+1=16
SEG_IDX_V7 = [BLACK_BG_ID] + [ABD_UNLABELED_ID] + ORG_CANON_IDS + LESION_IDS 


class MemorySharedPartsV7(nn.Module):
    """
：
      - lesion (mask[:,1]) 优先级最高
      - organs (mask[:,0]) 次之（可选做 liver subtract lesion）
      - abdomen-unlabeled (60): body内且不属于上述任何已选 organ/lesion 的区域
      - black-border (0): body外(abdomen==0) 的黑边区域，单独一类

    Input mask convention:
      - mask[:,0]: organ labels (multi-class, raw ids)
      - mask[:,1]: lesion labels (51..57 or 0)
      - mask[:,2]: abdomen/body (1 = inside body, 0 = outside/black)
    """

    def __init__(
        self,
        memory_size,
        kdim: int,
        vdim: int,
        seg_idx=None,
        seg_groups=None,
        lesion_ids=None,
        abdomen_unlabeled_id: int = ABD_UNLABELED_ID,
        black_bg_id: int = BLACK_BG_ID,
        subtract_lesion_from_liver: bool = True,
        momentum: float = 0.5,
        loss_scale: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.memory_size = list(memory_size)
        self.kdim = int(kdim)
        self.vdim = int(vdim)

        self.seg_idx = list(seg_idx) if seg_idx is not None else list(SEG_IDX_V8)
        self.seg_groups = dict(seg_groups) if seg_groups is not None else dict(SEG_GROUPS_MAJOR)
        self.lesion_ids = list(lesion_ids) if lesion_ids is not None else list(LESION_IDS)

        self.abdomen_unlabeled_id = int(abdomen_unlabeled_id)
        self.black_bg_id = int(black_bg_id)

        self.subtract_lesion_from_liver = bool(subtract_lesion_from_liver)
        self.momentum = float(momentum)
        self.loss_scale = float(loss_scale)
        self.eps = float(eps)

        assert len(self.memory_size) == len(self.seg_idx), "memory_size 长度必须与 seg_idx 一致"
        assert 0.0 <= self.momentum <= 1.0

        # seg_id -> memory index
        self.seg2mem = {int(s): i for i, s in enumerate(self.seg_idx)}

        self._lesion_set = set(int(x) for x in self.lesion_ids)

        # canonical organ ids are those in seg_idx that are NOT lesion, NOT abdomen, NOT black-bg
        self._organ_canon_ids = [
            int(x) for x in self.seg_idx
            if int(x) not in self._lesion_set
            and int(x) != self.abdomen_unlabeled_id
            and int(x) != self.black_bg_id
        ]

        self.val_norm = nn.LayerNorm(self.vdim, elementwise_affine=False)

        # debug stats
        self.last_stats = None

    # -------- mask building (disjoint) --------
    def build_active_masks(self, mask: torch.Tensor):
        """
        Returns:
          active_indices: sorted list of memory indices that appear in this batch
          mask_dict: {mem_idx: (B,1,H,W) float mask}, disjoint by construction
        """
        B, C, H, W = mask.shape
        assert C == 3, "期望 mask 为 Bx3xHxW: [organ, lesion, abdomen]"

        organ = mask[:, 0].long()
        lesion = mask[:, 1].long()
        abdomen = mask[:, 2].long()

        mask_dict = {}
        active = []

        # (A) lesions (highest priority)
        uniq_les = torch.unique(lesion)
        for seg_id in uniq_les.tolist():
            seg_id = int(seg_id)
            if seg_id == 0 or seg_id not in self._lesion_set:
                continue
            mem_idx = self.seg2mem.get(seg_id, None)
            if mem_idx is None:
                continue
            m = (lesion == seg_id).float().unsqueeze(1)
            if m.sum() > 0:
                mask_dict[mem_idx] = m
                active.append(mem_idx)

        # (B) union of selected organ raw labels (for abdomen-unlabeled subtraction)
        organ_selected_union = torch.zeros((B, H, W), dtype=torch.bool, device=mask.device)
        for canon_id in self._organ_canon_ids:
            raw_list = self.seg_groups.get(canon_id, [canon_id])
            for raw_id in raw_list:
                organ_selected_union |= (organ == int(raw_id))

        # organ pixels allowed (exclude lesion pixels)
        organ_allowed = (lesion == 0)

        # (C) organs (second priority)
        for canon_id in self._organ_canon_ids:
            mem_idx = self.seg2mem.get(canon_id, None)
            if mem_idx is None:
                continue

            raw_list = self.seg_groups.get(canon_id, [canon_id])
            m_bool = torch.zeros((B, H, W), dtype=torch.bool, device=mask.device)
            for raw_id in raw_list:
                m_bool |= (organ == int(raw_id))

            # remove lesion overlap
            m_bool = m_bool & organ_allowed

            # (optional) liver subtract lesion (already removed by organ_allowed, keep for clarity)
            if canon_id == 5 and self.subtract_lesion_from_liver:
                m_bool = m_bool & organ_allowed

            if m_bool.any():
                mask_dict[mem_idx] = m_bool.float().unsqueeze(1)
                active.append(mem_idx)

        # (D) abdomen-unlabeled (third priority): inside-body & not selected organ & not lesion
        if self.abdomen_unlabeled_id in self.seg2mem:
            mem_idx = self.seg2mem[self.abdomen_unlabeled_id]
            m_bool = (abdomen == 1) & (~organ_selected_union) & (lesion == 0)
            if m_bool.any():
                mask_dict[mem_idx] = m_bool.float().unsqueeze(1)
                active.append(mem_idx)

        # (E) black-border bg (lowest priority): outside-body
        if self.black_bg_id in self.seg2mem:
            mem_idx = self.seg2mem[self.black_bg_id]
            m_bool = (abdomen == 0)
            if m_bool.any():
                mask_dict[mem_idx] = m_bool.float().unsqueeze(1)
                active.append(mem_idx)

        active = sorted(set(active))
        return active, mask_dict

    # -------- core helpers --------
    def _flat_select(self, x_4d: torch.Tensor, flat_idx: torch.Tensor) -> torch.Tensor:
        return x_4d.view(-1, x_4d.size(-1)).index_select(0, flat_idx)

    def _logits(self, mem: torch.Tensor, query_4d: torch.Tensor, flat_idx: torch.Tensor) -> torch.Tensor:
        q = self._flat_select(query_4d, flat_idx)
        return q @ mem.t()

    def _scatter_sum(self, index: torch.Tensor, src: torch.Tensor, m: int) -> torch.Tensor:
        out = torch.zeros((m, src.size(1)), device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        return out

    def _scatter_sum_1d(self, index: torch.Tensor, src: torch.Tensor, m: int) -> torch.Tensor:
        out = torch.zeros((m,), device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        return out

    # -------- update --------
    def update_shared(self, cont_4d, styA_4d, styB_4d, flat_idx, memory):
        key, val_a, val_b = memory
        m = key.size(0)

        logits = self._logits(key, cont_4d, flat_idx)              # (N,M)
        prob = F.softmax(logits, dim=1)                            # (N,M)
        gather_idx = prob.argmax(dim=1)                            # (N,)

        cont_flat = self._flat_select(cont_4d, flat_idx)           # (N,kdim)
        styA_flat = self._flat_select(styA_4d, flat_idx)           # (N,vdim)
        styB_flat = self._flat_select(styB_4d, flat_idx)

        w = prob.gather(1, gather_idx.view(-1, 1)).squeeze(1).clamp_min(self.eps)  # (N,)

        slot_w = self._scatter_sum_1d(gather_idx, w, m)            # (m,)
        hit = slot_w > 0

        cont_sum = self._scatter_sum(gather_idx, w.unsqueeze(1) * cont_flat, m)
        valA_sum = self._scatter_sum(gather_idx, w.unsqueeze(1) * styA_flat, m)
        valB_sum = self._scatter_sum(gather_idx, w.unsqueeze(1) * styB_flat, m)

        denom = slot_w.unsqueeze(1) + self.eps
        cont_mean = cont_sum / denom
        valA_mean = valA_sum / denom
        valB_mean = valB_sum / denom

        new_key = key.clone()
        new_a = val_a.clone()
        new_b = val_b.clone()

        if hit.any():
            key_up = F.normalize(cont_mean[hit], dim=1)
            a_up = self.val_norm(valA_mean[hit])
            b_up = self.val_norm(valB_mean[hit])

            mom = self.momentum
            new_key[hit] = F.normalize((1.0 - mom) * key[hit] + mom * key_up, dim=1)
            new_a[hit] = (1.0 - mom) * val_a[hit] + mom * a_up
            new_b[hit] = (1.0 - mom) * val_b[hit] + mom * b_up

        return new_key.detach(), new_a.detach(), new_b.detach()

    # -------- read --------
    def read(self, query_4d, flat_idx, mem):
        key, val_a, val_b = mem
        logits = self._logits(key, query_4d, flat_idx)             # (N,M)
        prob = F.softmax(logits, dim=1)                            # (N,M)
        out_a = prob.detach() @ val_a
        out_b = prob.detach() @ val_b
        return out_a.detach(), out_b.detach()

    # -------- losses --------
    def gather_loss_shared(self, cont_4d, styA_4d, styB_4d, flat_idx, mem):
        key, val_a, val_b = mem

        logits_key = self._logits(key, cont_4d, flat_idx)          # (N,M)
        gather_idx = logits_key.argmax(dim=1).detach()             # (N,)

        logits_a = self._logits(val_a, styA_4d, flat_idx)          # (N,M)
        logits_b = self._logits(val_b, styB_4d, flat_idx)          # (N,M)

        key_loss = self.loss_scale * F.cross_entropy(logits_key, gather_idx)
        val_loss = self.loss_scale * (F.cross_entropy(logits_a, gather_idx) + F.cross_entropy(logits_b, gather_idx))
        return key_loss, val_loss, gather_idx

    def gather_total_loss_shared(self, cont_4d, gathers, memorys, valid_idx):
        mem0 = memorys[0][0]
        for i in range(1, len(memorys)):
            mem0 = torch.cat((mem0, memorys[i][0]), dim=0)

        bs, h, w, d = cont_4d.size()
        logits_all = (cont_4d @ mem0.t()).view(bs * h * w, -1)      # (flat, M_total)
        logits = logits_all.index_select(0, valid_idx)
        tgt = gathers.index_select(0, valid_idx)[:, 0].to(torch.long)

        total_loss = self.loss_scale * F.cross_entropy(logits, tgt)
        return total_loss

    # -------- forward --------
    def forward(self, cont_shared, sty_a, sty_b, mask, memorys):
        B, Cc, H, W = cont_shared.size()
        assert Cc == self.kdim, f"cont_shared channel({Cc}) must == kdim({self.kdim})"
        assert sty_a.size(1) == self.vdim and sty_b.size(1) == self.vdim, "style channel must == vdim"

        device = cont_shared.device
        flat = B * H * W

        num_classes = len(self.seg_idx)
        total_slots = int(sum(self.memory_size))
        pix_per_class = torch.zeros((num_classes,), device=device, dtype=torch.float32)
        hit_global = torch.zeros((total_slots,), device=device, dtype=torch.float32)

        sty_A = torch.zeros((flat, self.vdim), device=device)
        sty_B = torch.zeros((flat, self.vdim), device=device)
        gathers = torch.zeros((flat, 1), device=device)
        assigned = torch.zeros((flat,), dtype=torch.bool, device=device)

        key_loss = torch.tensor(0.0, device=device)
        value_loss = torch.tensor(0.0, device=device)

        updated_memorys = list(memorys)
        active_indices, mask_dict = self.build_active_masks(mask)

        cont_4d = cont_shared.permute(0, 2, 3, 1).contiguous()      # (B,H,W,kdim)
        styA_4d = sty_a.permute(0, 2, 3, 1).contiguous()            # (B,H,W,vdim)
        styB_4d = sty_b.permute(0, 2, 3, 1).contiguous()

        for i in active_indices:
            mask_i = mask_dict.get(i, None)
            if mask_i is None or mask_i.sum().item() == 0:
                continue

            flat_idx = mask_i.view(-1).nonzero(as_tuple=False).squeeze(1)
            if flat_idx.numel() == 0:
                continue

            pix_per_class[i] = pix_per_class[i] + float(flat_idx.numel())
            assigned[flat_idx] = True

            mem = updated_memorys[i]

            # update
            updated_memorys[i] = self.update_shared(cont_4d, styA_4d, styB_4d, flat_idx, mem)

            # read for output
            valA, valB = self.read(cont_4d, flat_idx, updated_memorys[i])
            sty_A[flat_idx] = valA
            sty_B[flat_idx] = valB

            # per-class gather loss
            k_loss_i, v_loss_i, gather_idx = self.gather_loss_shared(cont_4d, styA_4d, styB_4d, flat_idx, updated_memorys[i])

            # hit histogram pack
            offset = int(sum(self.memory_size[:i]))
            ones = torch.ones_like(gather_idx, dtype=torch.float32)
            hit_global.index_add_(0, gather_idx.to(torch.long) + offset, ones)

            gathers[flat_idx, 0] = (gather_idx + offset).to(torch.float)

            key_loss = key_loss + k_loss_i
            value_loss = value_loss + v_loss_i

        valid_idx = assigned.nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() > 0:
            key_loss = key_loss + self.gather_total_loss_shared(cont_4d, gathers, updated_memorys, valid_idx)

        sty_A = sty_A.view(B, H, W, self.vdim).permute(0, 3, 1, 2).contiguous()
        sty_B = sty_B.view(B, H, W, self.vdim).permute(0, 3, 1, 2).contiguous()

        self.last_stats = {
            'pix_per_class': pix_per_class.detach(),
            'hit_global': hit_global.detach(),
        }

        # rand_A/rand_B 你目前训练没用到，我这里不返回（保持与你 train.py 接口一致的话需要返回占位）
        # 为了不动训练代码，这里仍返回 rand_A/rand_B 两个占位张量
        rand_A = torch.zeros_like(sty_A)
        rand_B = torch.zeros_like(sty_B)

        return updated_memorys, sty_A, sty_B, rand_A, rand_B, key_loss, value_loss

    @torch.no_grad()
    def forward_second(self, cont_shared, mask, memorys):
        B, Cc, H, W = cont_shared.size()
        assert Cc == self.kdim

        device = cont_shared.device
        flat = B * H * W
        sty_A = torch.zeros((flat, self.vdim), device=device)
        sty_B = torch.zeros((flat, self.vdim), device=device)

        cont_4d = cont_shared.permute(0, 2, 3, 1).contiguous()

        active_indices, mask_dict = self.build_active_masks(mask)
        for i in active_indices:
            mask_i = mask_dict.get(i, None)
            if mask_i is None or mask_i.sum().item() == 0:
                continue
            flat_idx = mask_i.view(-1).nonzero(as_tuple=False).squeeze(1)
            if flat_idx.numel() == 0:
                continue
            valA, valB = self.read(cont_4d, flat_idx, memorys[i])
            sty_A[flat_idx] = valA
            sty_B[flat_idx] = valB

        sty_A = sty_A.view(B, H, W, self.vdim).permute(0, 3, 1, 2).contiguous()
        sty_B = sty_B.view(B, H, W, self.vdim).permute(0, 3, 1, 2).contiguous()
        return sty_A, sty_B
