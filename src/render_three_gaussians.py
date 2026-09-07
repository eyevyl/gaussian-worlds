from pathlib import Path
import torch
from gsplat import rasterization
from torchvision.utils import save_image

device = torch.device("cuda")

width = 320
height = 240

# 3D Gaussian centers in world coordinates: [x, y, z].
means = torch.tensor(
    [
        [-0.55, 0.00, 3.0],   # red: left
        [0.55, 0.00, 3.0],    # green: right
        [0.00, -0.45, 2.5],   # blue: higher and closer
    ],
    dtype=torch.float32,
    device=device,
    requires_grad=True,
)

# Identity rotations in gsplat's [w, x, y, z] quaternion convention.
quats = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ],
    device=device,
)

# Extent of each Gaussian along its local x, y, and z axes.
scales = torch.tensor(
    [
        [0.22, 0.22, 0.22],
        [0.22, 0.22, 0.22],
        [0.25, 0.25, 0.25],
    ],
    device=device,
)

opacities = torch.tensor([0.9, 0.9, 0.9], device=device)

colors = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    device=device,
)

# Identity world-to-camera transform.
viewmats = torch.eye(4, device=device).unsqueeze(0)

fx = 250.0
fy = 250.0
cx = width / 2
cy = height / 2

intrinsics = torch.tensor(
    [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ],
    device=device,
).unsqueeze(0)

rendered_colors, rendered_alphas, metadata = rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,
    colors=colors,
    viewmats=viewmats,
    Ks=intrinsics,
    width=width,
    height=height,
    packed=False,
)

rgb = rendered_colors[0].permute(2, 0, 1).clamp(0.0, 1.0)

output_path = Path("outputs/three_gaussians.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
save_image(rgb, output_path)

# Confirm that gradients can travel from pixels back to 3D positions.
loss = rendered_colors[..., 0].mean()
loss.backward()

print(f"Saved render to: {output_path}")
print(f"Rendered image shape: {tuple(rendered_colors.shape)}")
print(f"Projected radii: {metadata['radii']}")
print(f"Gradient on 3D means:\n{means.grad}")