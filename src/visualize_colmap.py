from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycolmap

reconstruction_path = Path("data/poster/colmap/sparse/0")
image_directory = Path("data/poster/images_8")
output_path = Path("outputs/colmap_scene.png")

reconstruction = pycolmap.Reconstruction(
    str(reconstruction_path)
)

available_names = {
    path.name
    for path in image_directory.iterdir()
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
}

available_images = sorted(
    [
        image
        for image in reconstruction.images.values()
        if image.name in available_names
    ],
    key=lambda image: image.name,
)

points = list(reconstruction.points3D.values())

xyz = np.stack([point.xyz for point in points])
rgb = np.stack([point.color for point in points]) / 255.0
errors = np.array([point.error for point in points])
track_lengths = np.array(
    [point.track.length() for point in points]
)

# Keep points with reasonable reprojection error and observations
# from at least three cameras.
quality_mask = (errors <= 2.0) & (track_lengths >= 3)

xyz_filtered = xyz[quality_mask]
rgb_filtered = rgb[quality_mask]

camera_centers = []
view_directions = []

for image in available_images:
    center = np.asarray(image.projection_center())
    rotation = np.asarray(
        image.cam_from_world().rotation.matrix()
    )

    # COLMAP's camera looks along +z in camera coordinates.
    forward_camera = np.array([0.0, 0.0, 1.0])

    # Convert the viewing direction from camera to world coordinates.
    forward_world = rotation.T @ forward_camera

    camera_centers.append(center)
    view_directions.append(forward_world)

camera_centers = np.stack(camera_centers)
view_directions = np.stack(view_directions)

# Robust limits prevent a few extreme points from shrinking the scene.
lower = np.percentile(xyz_filtered, 1, axis=0)
upper = np.percentile(xyz_filtered, 99, axis=0)
center = (lower + upper) / 2
radius = np.max(upper - lower) / 2


def draw_scene(axis, elevation, azimuth, title):
    axis.scatter(
        xyz_filtered[:, 0],
        xyz_filtered[:, 1],
        xyz_filtered[:, 2],
        c=rgb_filtered,
        s=1,
        alpha=0.55,
        linewidths=0,
    )

    axis.scatter(
        camera_centers[:, 0],
        camera_centers[:, 1],
        camera_centers[:, 2],
        c="orange",
        marker="^",
        s=16,
        label="Available cameras",
    )

    # Draw every fifth viewing direction to reduce clutter.
    axis.quiver(
        camera_centers[::5, 0],
        camera_centers[::5, 1],
        camera_centers[::5, 2],
        view_directions[::5, 0],
        view_directions[::5, 1],
        view_directions[::5, 2],
        color="black",
        length=radius * 0.12,
        normalize=True,
        linewidth=0.8,
    )

    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))

    axis.set_xlabel("World x")
    axis.set_ylabel("World y")
    axis.set_zlabel("World z")
    axis.set_title(title)
    axis.view_init(elev=elevation, azim=azimuth)


figure = plt.figure(figsize=(14, 6))

perspective_axis = figure.add_subplot(1, 2, 1, projection="3d")
draw_scene(
    perspective_axis,
    elevation=20,
    azimuth=-60,
    title="Perspective view",
)

top_axis = figure.add_subplot(1, 2, 2, projection="3d")
draw_scene(
    top_axis,
    elevation=90,
    azimuth=-90,
    title="Top view",
)

figure.tight_layout()
output_path.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output_path, dpi=180)
plt.close(figure)

print("Registered COLMAP images:", reconstruction.num_reg_images())
print("Downloaded matching images:", len(available_images))
print("Original sparse points:", len(points))
print("Filtered sparse points:", len(xyz_filtered))
print("Saved visualization to:", output_path)
