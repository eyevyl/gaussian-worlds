from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
from PIL import Image

DATA_DIR = Path("data/poster")
IMAGE_DIR = DATA_DIR / "images_8"
SPARSE_DIR = DATA_DIR / "colmap/sparse/0"
OUTPUT_PATH = Path("outputs/reprojection_check.png")

def main():
    reconstruction = pycolmap.Reconstruction(SPARSE_DIR)

    downloaded_names = {path.name for path in IMAGE_DIR.iterdir() if path.is_file()}

    available_images = [
        image
        for image in reconstruction.images.values()
        if image.name in downloaded_names
    ]

    if not available_images:
        raise RuntimeError("No downloaded images match the COLMAP reconstruction.")

    # Choose the downloaded image containing the most reconstructed observations.
    selected_image = max(
        available_images,
        key=lambda image: sum(point.has_point3D() for point in image.points2D),
    )

    camera = reconstruction.cameras[selected_image.camera_id]

    observed_pixels = []
    world_points = []

    for point2D in selected_image.points2D:
        if not point2D.has_point3D():
            continue

        point3D = reconstruction.points3D[point2D.point3D_id]

        observed_pixels.append(np.asarray(point2D.xy))
        world_points.append(np.asarray(point3D.xyz))

    observed_pixels = np.asarray(observed_pixels)
    world_points = np.asarray(world_points)

    # Convert points from world coordinates into this camera's coordinates:
    #
    #     X_camera = R X_world + t
    #
    cam_from_world = selected_image.cam_from_world()
    rotation = np.asarray(cam_from_world.rotation.matrix())
    translation = np.asarray(cam_from_world.translation)

    camera_points = (rotation @ world_points.T).T + translation

    if np.any(camera_points[:, 2] <= 0):
        raise RuntimeError("Some observed points are behind the selected camera.")

    # This applies the complete OPENCV camera model, including distortion.
    projected_pixels = np.asarray(camera.img_from_cam(camera_points))

    distortion_aware_errors = np.linalg.norm(
        projected_pixels - observed_pixels,
        axis=1,
    )

    # Compare against projection using K alone, without lens distortion.
    normalized_x = camera_points[:, 0] / camera_points[:, 2]
    normalized_y = camera_points[:, 1] / camera_points[:, 2]

    fx, fy, cx, cy = camera.params[:4]

    pinhole_pixels = np.column_stack(
        [
            fx * normalized_x + cx,
            fy * normalized_y + cy,
        ]
    )

    pinhole_errors = np.linalg.norm(
        pinhole_pixels - observed_pixels,
        axis=1,
    )

    image_path = IMAGE_DIR / selected_image.name

    with Image.open(image_path) as pil_image:
        display_image = np.asarray(pil_image.convert("RGB"))

    display_height, display_width = display_image.shape[:2]

    scale = np.array(
        [
            display_width / camera.width,
            display_height / camera.height,
        ]
    )

    projected_display = projected_pixels * scale

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(display_image)

    scatter = axes[0].scatter(
        projected_display[:, 0],
        projected_display[:, 1],
        c=distortion_aware_errors,
        cmap="turbo",
        vmin=0.0,
        vmax=3.0,
        s=10,
        alpha=0.8,
    )

    axes[0].set_title(
        f"COLMAP points reprojected onto {selected_image.name}"
    )
    axes[0].set_axis_off()

    colorbar = figure.colorbar(scatter, ax=axes[0], fraction=0.046)
    colorbar.set_label("Reprojection error at original resolution (pixels)")

    maximum_histogram_error = max(
        5.0,
        float(np.percentile(pinhole_errors, 95)),
    )

    axes[1].hist(
        distortion_aware_errors,
        bins=50,
        range=(0.0, maximum_histogram_error),
        alpha=0.75,
        label="OPENCV model",
    )

    axes[1].hist(
        pinhole_errors,
        bins=50,
        range=(0.0, maximum_histogram_error),
        alpha=0.55,
        label="K only; distortion ignored",
    )

    axes[1].set_title("Reprojection-error distribution")
    axes[1].set_xlabel("Error at original resolution (pixels)")
    axes[1].set_ylabel("Number of observations")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=180)
    plt.close(figure)

    print(f"Selected image: {selected_image.name}")
    print(f"Matched observations: {len(world_points)}")
    print(
        "Distortion-aware error: "
        f"mean={distortion_aware_errors.mean():.4f}, "
        f"median={np.median(distortion_aware_errors):.4f}, "
        f"max={distortion_aware_errors.max():.4f} pixels"
    )
    print(
        "Pinhole-only error: "
        f"mean={pinhole_errors.mean():.4f}, "
        f"median={np.median(pinhole_errors):.4f}, "
        f"max={pinhole_errors.max():.4f} pixels"
    )
    print(f"Saved visualization to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
