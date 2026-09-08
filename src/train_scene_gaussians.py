from pathlib import Path
import numpy as np
import pycolmap
import torch
from PIL import Image
import torch.nn.functional as F
from gsplat.rendering import rasterization

from render_initial_scene import (
    estimate_isotropic_scales,
    load_filtered_points,
)


DATA_DIR = Path("data/poster")
IMAGE_DIR = DATA_DIR / "images_8"
SPARSE_DIR = DATA_DIR / "colmap/sparse/0"
OUTPUT_DIR = Path("outputs/scene_training")

HOLDOUT_STRIDE = 5

NUM_TRAINING_STEPS = 2000
PRINT_EVERY = 100
RANDOM_SEED = 42

MINIMUM_SCALE = 0.001
MAXIMUM_SCALE = 0.5


def split_training_and_heldout_images(
    reconstruction,
    image_directory,
    holdout_stride,
):
    """
    Create an interleaved train/held-out split.

    Every holdout_stride-th matching image is reserved for evaluation.
    """

    downloaded_names = {
        path.name
        for path in image_directory.iterdir()
        if path.is_file()
    }

    matching_images = [
        image
        for image in reconstruction.images.values()
        if image.name in downloaded_names
    ]

    matching_images.sort(key=lambda image: image.name)

    training_images = []
    heldout_images = []

    for index, image in enumerate(matching_images):
        if index % holdout_stride == 0:
            heldout_images.append(image)
        else:
            training_images.append(image)

    return training_images, heldout_images, len(matching_images)


def load_training_view(
    reconstruction,
    colmap_image,
    image_directory,
    device,
):
    """
    Convert a given COLMAP image and its photograph into tensors that
    can be passed to gsplat.
    """

    camera = reconstruction.cameras[colmap_image.camera_id]
    image_path = image_directory / colmap_image.name

    # Load RGB pixels and normalize integers in [0, 255] to
    # floating-point values in [0, 1].
    with Image.open(image_path) as pil_image:
        rgb_array = np.asarray(
            pil_image.convert("RGB"),
            dtype=np.float32,
        ).copy()

    target_image = torch.tensor(
        rgb_array,
        device=device,
    ) / 255.0

    height, width = target_image.shape[:2]

    # ---------------------------------------------------------
    # World-to-camera transformation
    # ---------------------------------------------------------

    cam_from_world = colmap_image.cam_from_world()

    view_matrix_array = np.eye(4, dtype=np.float32)

    view_matrix_array[:3, :3] = np.asarray(
        cam_from_world.rotation.matrix(),
        dtype=np.float32,
    )

    view_matrix_array[:3, 3] = np.asarray(
        cam_from_world.translation,
        dtype=np.float32,
    )

    # Render one camera at a time.
    view_matrix = torch.tensor(
        view_matrix_array,
        device=device,
    )[None]

    # ---------------------------------------------------------
    # Camera intrinsics
    # ---------------------------------------------------------

    scale_x = width / camera.width
    scale_y = height / camera.height

    fx, fy, cx, cy = camera.params[:4]

    intrinsic_matrix = torch.tensor(
        [
            [fx * scale_x, 0.0, cx * scale_x],
            [0.0, fy * scale_y, cy * scale_y],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )[None]

    # ---------------------------------------------------------
    # OPENCV lens-distortion parameters
    # ---------------------------------------------------------

    if len(camera.params) != 8:
        raise RuntimeError(
            f"Expected an 8-parameter OPENCV camera, "
            f"but found {len(camera.params)} parameters."
        )

    k1, k2, p1, p2 = camera.params[4:8]

    radial_coefficients = torch.zeros(
        (1, 6),
        device=device,
        dtype=torch.float32,
    )

    radial_coefficients[0, 0] = float(k1)
    radial_coefficients[0, 1] = float(k2)

    tangential_coefficients = torch.tensor(
        [[p1, p2]],
        device=device,
        dtype=torch.float32,
    )

    return {
        "name": colmap_image.name,
        "target": target_image,
        "view_matrix": view_matrix,
        "intrinsic_matrix": intrinsic_matrix,
        "radial_coefficients": radial_coefficients,
        "tangential_coefficients": tangential_coefficients,
        "width": width,
        "height": height,
    }


def inverse_sigmoid(value):
    """
    Convert a value from (0, 1) into an unconstrained logit.

    sigmoid(inverse_sigmoid(x)) is approximately x.
    """

    return torch.log(value / (1.0 - value))


def initialize_gaussian_parameters(
    reconstruction,
    device,
):
    """
    Initialize trainable Gaussians from the filtered COLMAP points.
    """

    means_array, colors_array = load_filtered_points(reconstruction)

    initial_means = torch.tensor(
        means_array,
        device=device,
        dtype=torch.float32,
    )

    initial_colors = torch.tensor(
        colors_array,
        device=device,
        dtype=torch.float32,
    )

    print("\nEstimating initial Gaussian scales...")

    initial_scales = estimate_isotropic_scales(
        initial_means,
        num_neighbours=3,
    )

    number_of_gaussians = initial_means.shape[0]

    # Identity quaternion in gsplat's [w, x, y, z] convention.
    initial_quaternions = torch.zeros(
        (number_of_gaussians, 4),
        device=device,
        dtype=torch.float32,
    )
    initial_quaternions[:, 0] = 1.0

    # Start with relatively transparent Gaussians.
    initial_opacities = torch.full(
        (number_of_gaussians,),
        0.1,
        device=device,
        dtype=torch.float32,
    )

    # Avoid taking the logit of exactly 0 or 1.
    safe_colors = initial_colors.clamp(
        min=1e-4,
        max=1.0 - 1e-4,
    )

    gaussian_parameters = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(
                initial_means.clone()
            ),
            "quaternions": torch.nn.Parameter(
                initial_quaternions
            ),
            "log_scales": torch.nn.Parameter(
                torch.log(initial_scales)
            ),
            "opacity_logits": torch.nn.Parameter(
                inverse_sigmoid(initial_opacities)
            ),
            "color_logits": torch.nn.Parameter(
                inverse_sigmoid(safe_colors)
            ),
        }
    )

    return gaussian_parameters


def render_gaussians(
    gaussians,
    view,
):
    """
    Decode the trainable parameters and render them from one camera.
    """

    # Convert unconstrained stored parameters into valid Gaussian values.
    scales = torch.exp(
        gaussians["log_scales"]
    )

    opacities = torch.sigmoid(
        gaussians["opacity_logits"]
    )

    colors = torch.sigmoid(
        gaussians["color_logits"]
    )

    # Normalize each quaternion so that it represents a valid rotation.
    quaternions = F.normalize(
        gaussians["quaternions"],
        dim=-1,
    )

    background = torch.ones(
        (1, 3),
        device=gaussians["means"].device,
        dtype=torch.float32,
    )

    rendered_images, rendered_alphas, metadata = rasterization(
        means=gaussians["means"],
        quats=quaternions,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=view["view_matrix"],
        Ks=view["intrinsic_matrix"],
        width=view["width"],
        height=view["height"],
        backgrounds=background,
        camera_model="pinhole",
        # radial_coeffs=view["radial_coefficients"],
        # tangential_coeffs=view["tangential_coefficients"],
        render_mode="RGB",
        packed=False,
        with_ut=False,
    )

    rendered_image = rendered_images[0, :, :, :3]
    rendered_alpha = rendered_alphas[0, :, :, 0]

    return rendered_image, rendered_alpha, metadata


def run_gradient_check(
    gaussians,
    view,
    output_directory,
):
    """
    Perform one render and one backward pass without updating anything.
    """

    gaussians.zero_grad(set_to_none=True)

    rendered_image, rendered_alpha, _ = render_gaussians(
        gaussians=gaussians,
        view=view,
    )

    target_image = view["target"]

    loss = F.l1_loss(
        rendered_image,
        target_image,
    )

    mse = F.mse_loss(
        rendered_image,
        target_image,
    )

    psnr = -10.0 * torch.log10(mse)

    # Calculate derivatives of the loss with respect to every
    # trainable Gaussian parameter.
    loss.backward()

    print(f"\nGradient check using: {view['name']}")
    print(f"Initial L1 loss: {loss.item():.6f}")
    print(f"Initial PSNR: {psnr.item():.2f} dB")

    print("\nGradient statistics:")

    for name, parameter in gaussians.items():
        gradient = parameter.grad

        if gradient is None:
            print(f"  {name:16s} gradient=None")
            continue

        finite = torch.isfinite(gradient).all().item()
        mean_absolute = gradient.abs().mean().item()
        maximum_absolute = gradient.abs().max().item()

        nonzero_fraction = (
            (gradient != 0).float().mean().item()
        )

        print(
            f"  {name:16s} "
            f"mean_abs={mean_absolute:.8e} "
            f"max_abs={maximum_absolute:.8e} "
            f"nonzero={nonzero_fraction:.2%} "
            f"finite={finite}"
        )

    # Save target, render, and opacity beside one another.
    target_array = (
        target_image.detach()
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
    )

    render_array = (
        rendered_image.detach()
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
    )

    alpha_array = (
        rendered_alpha.detach()
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
    )

    alpha_rgb = np.repeat(
        alpha_array[:, :, None],
        repeats=3,
        axis=2,
    )

    comparison = np.concatenate(
        [target_array, render_array, alpha_rgb],
        axis=1,
    )

    comparison_uint8 = (
        comparison * 255.0
    ).round().astype(np.uint8)

    output_path = output_directory / "gradient_check.png"

    Image.fromarray(comparison_uint8).save(output_path)

    print(
        "\nSaved target/render/opacity comparison to: "
        f"{output_path}"
    )


def create_optimizer(gaussians):
    """
    Create an Adam optimizer with a separate learning rate for each
    type of Gaussian parameter.
    """

    optimizer = torch.optim.Adam(
        [
            {
                "params": [gaussians["means"]],
                "lr": 0.001,
                "name": "means",
            },
            {
                "params": [gaussians["log_scales"]],
                "lr": 0.005,
                "name": "log_scales",
            },
            {
                "params": [gaussians["quaternions"]],
                "lr": 0.001,
                "name": "quaternions",
            },
            {
                "params": [gaussians["opacity_logits"]],
                "lr": 0.01,
                "name": "opacity_logits",
            },
            {
                "params": [gaussians["color_logits"]],
                "lr": 0.025,
                "name": "color_logits",
            },
        ]
    )

    return optimizer


def run_one_optimization_step(
    gaussians,
    optimizer,
    view,
):
    """
    Perform one complete gradient-descent update and report what changed.
    """

    parameter_values_before = {
        name: parameter.detach().clone()
        for name, parameter in gaussians.items()
    }

    # Remove gradients left over from the earlier gradient check.
    optimizer.zero_grad(set_to_none=True)

    rendered_before, _, _ = render_gaussians(
        gaussians=gaussians,
        view=view,
    )

    loss_before = F.l1_loss(
        rendered_before,
        view["target"],
    )

    # Calculate gradients.
    loss_before.backward()

    # Use those gradients to update the Gaussian parameters.
    optimizer.step()

    # Render again after the update.
    with torch.no_grad():
        rendered_after, _, _ = render_gaussians(
            gaussians=gaussians,
            view=view,
        )

        loss_after = F.l1_loss(
            rendered_after,
            view["target"],
        )

    print(f"\nOne-step optimization using: {view['name']}")
    print(f"Loss before update: {loss_before.item():.6f}")
    print(f"Loss after update:  {loss_after.item():.6f}")

    print("\nMean absolute parameter changes:")

    for name, parameter in gaussians.items():
        change = (
            parameter.detach() - parameter_values_before[name]
        ).abs().mean()

        print(f"  {name:16s} {change.item():.8e}")


def evaluate_training_views(
    gaussians,
    training_views,
):
    l1_values = []
    psnr_values = []

    with torch.no_grad():
        for view in training_views:
            rendered_image, _, _ = render_gaussians(
                gaussians=gaussians,
                view=view,
            )

            l1 = F.l1_loss(
                rendered_image,
                view["target"],
            )

            mse = F.mse_loss(
                rendered_image,
                view["target"],
            )

            psnr = -10.0 * torch.log10(
                mse.clamp_min(1e-10)
            )

            l1_values.append(l1.item())
            psnr_values.append(psnr.item())

    return {
        "mean_l1": float(np.mean(l1_values)),
        "mean_psnr": float(np.mean(psnr_values)),
    }


def save_render_comparison(
    gaussians,
    view,
    output_path,
):
    with torch.no_grad():
        rendered_image, rendered_alpha, _ = render_gaussians(
            gaussians=gaussians,
            view=view,
        )

    target_array = (view["target"].detach().clamp(0.0, 1.0).cpu().numpy())
    render_array = (rendered_image.detach().clamp(0.0, 1.0).cpu().numpy())
    alpha_array = (rendered_alpha.detach().clamp(0.0, 1.0).cpu().numpy())

    alpha_rgb = np.repeat(
        alpha_array[:, :, None],
        repeats=3,
        axis=2,
    )

    comparison = np.concatenate(
        [target_array, render_array, alpha_rgb],
        axis=1,
    )

    comparison_uint8 = (
        comparison * 255.0
    ).round().astype(np.uint8)

    Image.fromarray(comparison_uint8).save(output_path)


def train_gaussians(
    gaussians,
    optimizer,
    training_views,
    number_of_steps,
):
    recent_losses = []

    print("\nStarting multi-view optimization...")

    for step in range(number_of_steps):
        # Randomly select one of the eight training cameras.
        view_index = torch.randint(
            low=0,
            high=len(training_views),
            size=(1,),
        ).item()

        view = training_views[view_index]

        optimizer.zero_grad(set_to_none=True)

        rendered_image, _, _ = render_gaussians(
            gaussians=gaussians,
            view=view,
        )

        loss = F.l1_loss(
            rendered_image,
            view["target"],
        )

        loss.backward()
        optimizer.step()

        # Keep transformed parameter values inside safe ranges.
        with torch.no_grad():
            gaussians["log_scales"].clamp_(
                min=float(np.log(MINIMUM_SCALE)),
                max=float(np.log(MAXIMUM_SCALE)),
            )

            gaussians["opacity_logits"].clamp_(
                min=-8.0,
                max=8.0,
            )

            gaussians["color_logits"].clamp_(
                min=-8.0,
                max=8.0,
            )

            # Keep stored quaternions at unit length.
            normalized_quaternions = F.normalize(
                gaussians["quaternions"],
                dim=-1,
            )

            gaussians["quaternions"].copy_(
                normalized_quaternions
            )

        recent_losses.append(loss.item())

        if step % PRINT_EVERY == 0 or step == number_of_steps - 1:
            recent_mean = float(
                np.mean(recent_losses[-PRINT_EVERY:])
            )

            print(
                f"step={step:04d} "
                f"view={view_index} "
                f"name={view['name']} "
                f"loss={loss.item():.6f} "
                f"recent_mean={recent_mean:.6f}"
            )


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for scene optimization.")

    device = torch.device("cuda")
    reconstruction = pycolmap.Reconstruction(SPARSE_DIR)

    training_images, heldout_images, matching_count = split_training_and_heldout_images(
        reconstruction=reconstruction,
        image_directory=IMAGE_DIR,
        holdout_stride=HOLDOUT_STRIDE,
    )

    print(f"Device: {device}")
    print(f"Registered/downloaded image matches: {matching_count}")
    print(f"Selected training images: {len(training_images)}")
    print(f"Reserved held-out images: {len(heldout_images)}")

    print("\nTraining views:")

    for index, colmap_image in enumerate(training_images):
        image_path = IMAGE_DIR / colmap_image.name

        with Image.open(image_path) as pil_image:
            width, height = pil_image.size

        print(
            f"  [{index}] "
            f"image_id={colmap_image.image_id:3d} "
            f"name={colmap_image.name} "
            f"resolution={width}x{height}"
        )

    training_views = [
        load_training_view(
            reconstruction=reconstruction,
            colmap_image=colmap_image,
            image_directory=IMAGE_DIR,
            device=device,
        )
        for colmap_image in training_images
    ]

    print("\nLoaded training tensors:")

    for index, view in enumerate(training_views):
        print(
            f"  [{index}] {view['name']} "
            f"target={tuple(view['target'].shape)} "
            f"view_matrix={tuple(view['view_matrix'].shape)} "
            f"K={tuple(view['intrinsic_matrix'].shape)}"
        )

    gaussians = initialize_gaussian_parameters(
        reconstruction=reconstruction,
        device=device,
    )

    print("\nTrainable Gaussian tensors:")

    for name, parameter in gaussians.items():
        print(
            f"  {name:16s} "
            f"shape={tuple(parameter.shape)} "
            f"requires_grad={parameter.requires_grad}"
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in gaussians.parameters()
    )

    print(f"\nTotal trainable scalar values: {parameter_count:,}")

    actual_scales = torch.exp(gaussians["log_scales"])
    actual_opacities = torch.sigmoid(
        gaussians["opacity_logits"]
    )
    actual_colors = torch.sigmoid(
        gaussians["color_logits"]
    )

    print(
        "Initial median scale: "
        f"{actual_scales[:, 0].median().item():.6f}"
    )
    print(
        "Initial mean opacity: "
        f"{actual_opacities.mean().item():.4f}"
    )
    print(
        "Initial color range: "
        f"[{actual_colors.min().item():.4f}, "
        f"{actual_colors.max().item():.4f}]"
    )

    gradient_check_view = training_views[
        len(training_views) // 2
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # run_gradient_check(
    #     gaussians=gaussians,
    #     view=gradient_check_view,
    #     output_directory=OUTPUT_DIR,
    # )

    optimizer = create_optimizer(gaussians)

    # run_one_optimization_step(
    #     gaussians=gaussians,
    #     optimizer=optimizer,
    #     view=gradient_check_view,
    # )

    monitor_view = training_views[
        len(training_views) // 2
    ]

    save_render_comparison(
        gaussians=gaussians,
        view=monitor_view,
        output_path=OUTPUT_DIR / "before_training.png",
    )

    initial_metrics = evaluate_training_views(
        gaussians=gaussians,
        training_views=training_views,
    )

    print(
        "\nBefore training: "
        f"mean L1={initial_metrics['mean_l1']:.6f}, "
        f"mean PSNR={initial_metrics['mean_psnr']:.2f} dB"
    )

    train_gaussians(
        gaussians=gaussians,
        optimizer=optimizer,
        training_views=training_views,
        number_of_steps=NUM_TRAINING_STEPS,
    )

    final_metrics = evaluate_training_views(
        gaussians=gaussians,
        training_views=training_views,
    )

    print(
        "\nAfter training: "
        f"mean L1={final_metrics['mean_l1']:.6f}, "
        f"mean PSNR={final_metrics['mean_psnr']:.2f} dB"
    )

    save_render_comparison(
        gaussians=gaussians,
        view=monitor_view,
        output_path=OUTPUT_DIR / "after_training.png",
    )

    checkpoint_path = OUTPUT_DIR / "fixed_gaussians.pt"

    torch.save(
        {
            "gaussians": gaussians.state_dict(),
            "training_image_names": [
                view["name"]
                for view in training_views
            ],
            "training_steps": NUM_TRAINING_STEPS,
        },
        checkpoint_path,
    )

    print(f"Saved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
