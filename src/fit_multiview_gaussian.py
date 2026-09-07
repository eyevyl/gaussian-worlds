from pathlib import Path

import torch
import torch.nn.functional as F
from gsplat import rasterization
from torchvision.utils import save_image

device = torch.device("cuda")
width = 320
height = 240

intrinsics = torch.tensor(
    [
        [250.0, 0.0, width / 2],
        [0.0, 250.0, height / 2],
        [0.0, 0.0, 1.0],
    ],
    device=device,
)

# Camera 1 is at the world origin.
camera_1 = torch.eye(4, device=device)

# Camera 2 has its center at x = 0.5 in world coordinates.
# Because this is a world-to-camera matrix, its translation is -0.5.
camera_2 = torch.eye(4, device=device)
camera_2[0, 3] = -0.5

viewmats = torch.stack([camera_1, camera_2])

quats = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0]],
    device=device,
)

opacities = torch.tensor([0.9], device=device)
colors = torch.tensor([[1.0, 0.0, 0.0]], device=device)


def render(mean, log_scale, cameras):
    """Render one Gaussian from one or more cameras."""

    # Exponentiation guarantees that scale remains positive.
    scale = torch.exp(log_scale)
    scales = scale.expand(1, 3)

    camera_intrinsics = intrinsics.unsqueeze(0).repeat(
        cameras.shape[0], 1, 1
    )

    rendered_colors, _, _ = rasterization(
        means=mean.unsqueeze(0),
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=cameras,
        Ks=camera_intrinsics,
        width=width,
        height=height,
        packed=False,
    )

    return rendered_colors


target_mean = torch.tensor(
    [0.2, -0.1, 3.0],
    device=device,
)

target_log_scale = torch.log(
    torch.tensor(0.25, device=device)
)

initial_mean = torch.tensor(
    [0.1, -0.05, 1.5],
    device=device,
)

initial_log_scale = torch.log(
    torch.tensor(0.125, device=device)
)

with torch.no_grad():
    target_images = render(
        target_mean,
        target_log_scale,
        viewmats,
    )


def fit_gaussian(cameras, targets, steps=300):
    mean = torch.nn.Parameter(initial_mean.clone())
    log_scale = torch.nn.Parameter(initial_log_scale.clone())

    optimizer = torch.optim.Adam(
        [mean, log_scale],
        lr=0.02,
    )

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)

        predictions = render(mean, log_scale, cameras)
        loss = F.mse_loss(predictions, targets)

        loss.backward()

        if step % 50 == 0 or step == steps - 1:
            print(
                f"step={step:03d} "
                f"loss={loss.item():.8f} "
                f"mean={mean.detach().cpu().tolist()} "
                f"scale={torch.exp(log_scale).item():.4f}"
            )

        optimizer.step()

    return mean.detach(), log_scale.detach()


print("\n--- Fitting with one camera ---")
single_mean, single_log_scale = fit_gaussian(
    viewmats[:1],
    target_images[:1],
)

print("\n--- Fitting with two cameras ---")
multi_mean, multi_log_scale = fit_gaussian(
    viewmats,
    target_images,
)

with torch.no_grad():
    single_images = render(
        single_mean,
        single_log_scale,
        viewmats,
    )

    multi_images = render(
        multi_mean,
        multi_log_scale,
        viewmats,
    )

comparison = torch.cat(
    [
        target_images,
        single_images,
        multi_images,
    ],
    dim=0,
).permute(0, 3, 1, 2)

output_path = Path("outputs/multiview_optimization.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
save_image(comparison, output_path, nrow=2)

print("\nTarget:")
print(" mean:", target_mean.cpu().tolist())
print(" scale:", torch.exp(target_log_scale).item())

print("\nSingle-camera result:")
print(" mean:", single_mean.cpu().tolist())
print(" scale:", torch.exp(single_log_scale).item())

print("\nTwo-camera result:")
print(" mean:", multi_mean.cpu().tolist())
print(" scale:", torch.exp(multi_log_scale).item())

print(f"\nSaved comparison to: {output_path}")
