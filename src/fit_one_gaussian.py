from pathlib import Path
import torch
import torch.nn.functional as F
from gsplat import rasterization
from torchvision.utils import save_image

device = torch.device("cuda")

width = 320
height = 240

viewmats = torch.eye(4, device=device).unsqueeze(0)

intrinsics = torch.tensor(
    [
        [250.0, 0.0, width / 2],
        [0.0, 250.0, height / 2],
        [0.0, 0.0, 1.0],
    ],
    device=device,
).unsqueeze(0)

quats = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0]],
    device=device,
)

scales = torch.tensor(
    [[0.22, 0.22, 0.22]],
    device=device,
)

opacities = torch.tensor([0.9], device=device)
colors = torch.tensor([[1.0, 0.0, 0.0]], device=device)

def render_gaussian(x: torch.Tensor) -> torch.Tensor:
    """Render one Gaussian whose x-coordinate may be learnable."""

    zero = x.new_tensor(0.0)
    depth = x.new_tensor(3.0)

    # torch.stack preserves the autograd connection to x.
    means = torch.stack([x, zero, depth]).unsqueeze(0)

    rendered_colors, _, _ = rasterization(
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

    # Remove the one-camera batch dimension: [1, H, W, 3] -> [H, W, 3].
    return rendered_colors[0]

output_directory = Path("outputs")
output_directory.mkdir(parents=True, exist_ok=True)

# The target acts like the training photograph.
with torch.no_grad():
    target_x = torch.tensor(-0.20, device=device)
    target_image = render_gaussian(target_x)

# This is the scene parameter that gradient descent will estimate.
learned_x = torch.nn.Parameter(torch.tensor(-0.55, device=device))
optimizer = torch.optim.Adam([learned_x], lr=0.03)

with torch.no_grad():
    initial_image = render_gaussian(learned_x)

for step in range(100):
    optimizer.zero_grad(set_to_none=True)

    prediction = render_gaussian(learned_x)
    loss = F.l1_loss(prediction, target_image)

    loss.backward()
    gradient = learned_x.grad.item()
    optimizer.step()

    if step % 10 == 0 or step == 99:
        print(
            f"step={step:03d} "
            f"loss={loss.item():.6f} "
            f"x={learned_x.item():.4f} "
            f"gradient={gradient:+.6f}"
        )

with torch.no_grad():
    final_image = render_gaussian(learned_x)

comparison = torch.stack(
    [initial_image, target_image, final_image]
).permute(0, 3, 1, 2)

output_path = output_directory / "one_gaussian_optimization.png"
save_image(comparison, output_path, nrow=3)

print(f"\nTarget x:  {target_x.item():.4f}")
print(f"Learned x: {learned_x.item():.4f}")
print(f"Saved initial/target/final comparison to: {output_path}")
