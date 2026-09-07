from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
import torch
from gsplat.rendering import rasterization
from PIL import Image

DATA_DIR = Path("data/poster")
IMAGE_DIR = DATA_DIR / "images_8"
SPARSE_DIR = DATA_DIR / "colmap/sparse/0"
OUTPUT_PATH = Path("outputs/initial_scene_render.png")

IMAGE_NAME = "frame_00205.png"

MIN_TRACK_LENGTH = 3
MAX_REPROJECTION_ERROR = 2.0

NUM_NEIGHBOURS = 3
SCALE_MULTIPLIER = 0.6
INITIAL_OPACITY = 0.7


def load_filtered_points(reconstruction):
    """Load reliable COLMAP points and their RGB colors."""

    means = []
    colors = []

    for point in reconstruction.points3D.values():
        if point.track.length() < MIN_TRACK_LENGTH:
            continue

        if point.error > MAX_REPROJECTION_ERROR:
            continue

        means.append(np.asarray(point.xyz, dtype=np.float32))
        colors.append(np.asarray(point.color, dtype=np.float32) / 255.0)

    means = np.asarray(means, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)

    return means, colors


def estimate_isotropic_scales(
    means,
    num_neighbours=3,
    chunk_size=1024,
):
    """
    Estimate each Gaussian's radius using nearby COLMAP points.

    The calculation is performed in chunks so that we do not create
    one enormous N-by-N distance matrix.
    """

    number_of_points = means.shape[0]
    neighbour_scales = []

    for start in range(0, number_of_points, chunk_size):
        end = min(start + chunk_size, number_of_points)

        query_points = means[start:end]

        # Shape:
        #     [number of query points, total scene points]
        distances = torch.cdist(query_points, means)

        # A point has distance zero from itself. Replace those diagonal
        # entries with infinity so it cannot count as its own neighbour.
        local_rows = torch.arange(end - start, device=means.device)
        global_columns = torch.arange(start, end, device=means.device)
        distances[local_rows, global_columns] = float("inf")

        nearest_distances = torch.topk(
            distances,
            k=num_neighbours,
            dim=1,
            largest=False,
        ).values

        local_scales = nearest_distances.mean(dim=1)
        neighbour_scales.append(local_scales)

    scalar_scales = torch.cat(neighbour_scales)

    # Isolated outliers can have very large neighbour distances.
    # Clamp the most extreme 5% at either end.
    lower_bound = torch.quantile(scalar_scales, 0.05)
    upper_bound = torch.quantile(scalar_scales, 0.95)

    scalar_scales = torch.clamp(
        scalar_scales,
        min=lower_bound,
        max=upper_bound,
    )

    scalar_scales = SCALE_MULTIPLIER * scalar_scales

    # Convert one radius per Gaussian into [sx, sy, sz].
    isotropic_scales = scalar_scales[:, None].repeat(1, 3)

    return isotropic_scales


def find_image(reconstruction, image_name):
    for image in reconstruction.images.values():
        if image.name == image_name:
            return image

    raise RuntimeError(
        f"{image_name} was not found in the COLMAP reconstruction."
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this render.")

    device = torch.device("cuda")

    reconstruction = pycolmap.Reconstruction(SPARSE_DIR)
    colmap_image = find_image(reconstruction, IMAGE_NAME)
    camera = reconstruction.cameras[colmap_image.camera_id]

    means_np, colors_np = load_filtered_points(reconstruction)

    means = torch.tensor(means_np, device=device)
    colors = torch.tensor(colors_np, device=device)

    print(f"Initialized Gaussian count: {len(means)}")
    print("Estimating scales from neighbouring points...")

    scales = estimate_isotropic_scales(
        means,
        num_neighbours=NUM_NEIGHBOURS,
    )

    # gsplat uses quaternion ordering [w, x, y, z].
    # [1, 0, 0, 0] is the identity rotation.
    quaternions = torch.zeros(
        (len(means), 4),
        device=device,
        dtype=torch.float32,
    )
    quaternions[:, 0] = 1.0

    opacities = torch.full(
        (len(means),),
        INITIAL_OPACITY,
        device=device,
        dtype=torch.float32,
    )

    image_path = IMAGE_DIR / IMAGE_NAME

    with Image.open(image_path) as pil_image:
        target_image = np.asarray(pil_image.convert("RGB"))

    height, width = target_image.shape[:2]

    # Build the world-to-camera matrix expected by gsplat.
    cam_from_world = colmap_image.cam_from_world()

    view_matrix_np = np.eye(4, dtype=np.float32)
    view_matrix_np[:3, :3] = np.asarray(
        cam_from_world.rotation.matrix(),
        dtype=np.float32,
    )
    view_matrix_np[:3, 3] = np.asarray(
        cam_from_world.translation,
        dtype=np.float32,
    )

    view_matrix = torch.tensor(
        view_matrix_np,
        device=device,
    )[None]

    # Scale the original intrinsics to match images_8.
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

    # COLMAP OPENCV parameter order:
    # fx, fy, cx, cy, k1, k2, p1, p2
    k1, k2, p1, p2 = camera.params[4:8]

    # gsplat accepts six radial coefficients. This camera only has k1
    # and k2, so the remaining coefficients are zero.
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

    background = torch.ones(
        (1, 3),
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        rendered_images, rendered_alphas, _ = rasterization(
            means=means,
            quats=quaternions,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=view_matrix,
            Ks=intrinsic_matrix,
            width=width,
            height=height,
            backgrounds=background,
            packed=False,#packed=True,
            camera_model="pinhole",
            radial_coeffs=radial_coefficients,
            tangential_coeffs=tangential_coefficients,
            render_mode="RGB",
            with_ut=True
        )

    rendered_image = (
        rendered_images[0]
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
    )

    alpha_image = (
        rendered_alphas[0, :, :, 0]
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
    )

    scale_values = scales[:, 0].cpu()

    print(
        "Scale statistics: "
        f"min={scale_values.min():.6f}, "
        f"median={scale_values.median():.6f}, "
        f"max={scale_values.max():.6f}"
    )

    coverage = float((alpha_image > 0.01).mean())

    print(f"Image resolution: {width} x {height}")
    print(f"Pixel coverage above 0.01 opacity: {coverage:.2%}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(12, 7))

    axes[0].imshow(target_image)
    axes[0].set_title("Real image")

    axes[1].imshow(rendered_image)
    axes[1].set_title("Initialized Gaussians")

    axes[2].imshow(alpha_image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("Accumulated opacity")

    for axis in axes:
        axis.set_axis_off()

    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=180)
    plt.close(figure)

    print(f"Saved comparison to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
