
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.Tri_Plane_help import normalize_3d_coordinate


class Decoders(nn.Module):
    """Tri-plane decoders for Gaussian attributes (RGB + Semantic)."""

    def __init__(
        self,
        bound,
        c_dim=32,
        hidden_size=64,  # 16
        n_blocks=2,
        sem_dim=52,     # 语义维度（默认 52，可从配置覆盖）
        sem_feat_dim=None,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.sem_dim = sem_dim
        self.sem_feat_dim = int(sem_feat_dim) if sem_feat_dim is not None else 2 * c_dim
        self.register_buffer("bound", bound.clone())

        # tri-plane 特征拼接后的输入维度，保持与你原逻辑一致

        # 颜色分支 MLP
        self.c_linears = nn.ModuleList(
            [nn.Linear(2*c_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) for _ in range(n_blocks - 1)]
        )
        self.c_output_linear = nn.Linear(hidden_size, 3)

        # 语义分支 MLP（结构同颜色分支，但参数独立）
        self.s_linears = nn.ModuleList(
            [nn.Linear(self.sem_feat_dim, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size) for _ in range(n_blocks - 1)]
        )
        self.s_output_linear = nn.Linear(hidden_size, sem_dim)

        # FiLM 调制（仅用于语义分支，可选使用颜色先验）
        # 输出维度与语义特征输入一致（默认与 tri-plane 查询输出相同）
        self.film_scale = 0.1
        self.s_film_gamma = nn.Linear(3, self.sem_feat_dim)
        self.s_film_beta = nn.Linear(3, self.sem_feat_dim)
        with torch.no_grad():
            self.s_film_gamma.weight.zero_()
            self.s_film_gamma.bias.zero_()
            self.s_film_beta.weight.zero_()
            self.s_film_beta.bias.zero_()

    def sample_plane_feature(self, p_nor, planes_xy, planes_xz, planes_yz):
        """
        Sample feature from planes
        Args:
            p_nor (tensor): normalized 3D coordinates
            planes_xy (list): xy planes
            planes_xz (list): xz planes
            planes_yz (list): yz planes
        Returns:
            feat (tensor): sampled features
        """
        vgrid = p_nor[None, :, None]

        feat = []
        for i in range(len(planes_xy)):
        # for i in [1]:
            xy = F.grid_sample(
                planes_xy[i], vgrid[..., [0, 1]],
                padding_mode='border', align_corners=True, mode='bilinear'
            ).squeeze().transpose(0, 1)
            xz = F.grid_sample(
                planes_xz[i], vgrid[..., [0, 2]],
                padding_mode='border', align_corners=True, mode='bilinear'
            ).squeeze().transpose(0, 1)
            yz = F.grid_sample(
                planes_yz[i], vgrid[..., [1, 2]],
                padding_mode='border', align_corners=True, mode='bilinear'
            ).squeeze().transpose(0, 1)
            feat.append(xy + xz + yz)
        feat = torch.cat(feat, dim=-1)  # (N, c_in_dim)
        return feat

    def get_raw_rgb(self, p_nor, all_planes):
        """
        Get raw RGB
        Args:
            p_nor (tensor): normalized 3D coordinates
            all_planes (Tuple): all feature planes
        Returns:
            rgb (tensor): raw RGB in [0,1]
        """
        c_planes_xy, c_planes_xz, c_planes_yz = all_planes[3:6]
        c_feat = self.sample_plane_feature(p_nor, c_planes_xy, c_planes_xz, c_planes_yz)

        h = c_feat
        for l in self.c_linears:
            h = F.relu(l(h), inplace=True)
        rgb = torch.sigmoid(self.c_output_linear(h))
        return rgb

    def get_semantic(self, p_nor, all_planes, rgb=None):
        """
        Get raw semantic logits (no activation)
        Returns:
            sem (tensor): (N, sem_dim)
        """
        c_planes_xy, c_planes_xz, c_planes_yz = all_planes[3:6]
        c_feat = self.sample_plane_feature(p_nor, c_planes_xy, c_planes_xz, c_planes_yz)  # [N, 2*c_dim]
        return self.get_semantic_from_feature(c_feat, rgb=rgb)

    def get_semantic_from_feature(self, sem_feat, rgb=None):
        """Decode semantic logits directly from per-Gaussian semantic features."""
        if sem_feat.shape[-1] != self.sem_feat_dim:
            raise ValueError(
                f"Semantic feature dim mismatch: expected {self.sem_feat_dim}, got {sem_feat.shape[-1]}"
            )

        # FiLM 调制：颜色先验仅单向作为条件（不回传梯度）
        if rgb is not None:
            rgb_prior = torch.clamp(rgb.detach(), 0.0, 1.0)
            gamma = 1.0 + self.film_scale * torch.tanh(self.s_film_gamma(rgb_prior))
            beta = self.film_scale * torch.tanh(self.s_film_beta(rgb_prior))
            sem_feat = gamma * sem_feat + beta

        h = sem_feat
        for l in self.s_linears:
            h = F.relu(l(h), inplace=True)
        sem = self.s_output_linear(h)  # 不做激活，按需在 loss 里用 CE / BCE / softmax
        return sem

    def forward(self, p, all_planes, decode_rgb: bool = True, decode_sem: bool = True, rgb=None, sem_feat=None):
        """Decode requested attributes from tri-planes.

        Returns a dict mapping attribute names to tensors with shape matching ``p``.
        """
        p_shape = p.shape
        p_nor = normalize_3d_coordinate(p.clone(), self.bound) if (decode_rgb or (decode_sem and sem_feat is None)) else None

        out = {}
        if decode_rgb:

            rgb = self.get_raw_rgb(p_nor, all_planes)
            out['rgb_colors'] = rgb.contiguous().view(*p_shape[:-1], 3)
        if decode_sem:
            if sem_feat is not None:
                sem = self.get_semantic_from_feature(sem_feat, rgb=rgb.detach() if rgb is not None else None)
            elif rgb is not None:
                rgb_prior = rgb.detach()
                sem = self.get_semantic(p_nor, all_planes, rgb=rgb_prior)
            else:
                sem = self.get_semantic(p_nor, all_planes)
            out['semantic_id'] = sem.contiguous().view(*p_shape[:-1], self.sem_dim)
        return out
