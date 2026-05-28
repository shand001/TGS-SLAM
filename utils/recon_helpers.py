import torch
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
except ImportError:
    class Camera:
        """Fallback camera settings container for gsplat-only code paths."""

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

def setup_camera(w, h, k, w2c, near=0.01, far=100, device="cuda"):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).to(device).float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).to(device).float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device=device),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
    return cam

def to_gsplat_camera(w, h, k, w2c, device="cuda"):
    # k: 3x3 内参；w2c: 4x4（世界→相机）外参
    K = torch.as_tensor(k, device=device, dtype=torch.float32)[None]      # [1,3,3]
    view = torch.as_tensor(w2c, device=device, dtype=torch.float32)[None] # [1,4,4]
    H, W = int(h), int(w)
    return H, W, K, view
