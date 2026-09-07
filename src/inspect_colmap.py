from pathlib import Path
import numpy as np
import pycolmap
from PIL import Image

reconstruction_path = Path("data/poster/colmap/sparse/0")
image_directory = Path("data/poster/images_8")

reconstruction = pycolmap.Reconstruction(
    str(reconstruction_path)
)

print("--- Reconstruction summary ---")
print(reconstruction.summary())
print("Internally valid:", reconstruction.is_valid())

# Sort by filename so the selected image is deterministic.
registered_images = sorted(
    reconstruction.images.values(),
    key=lambda image: image.name,
)

first_image = registered_images[0]
camera = reconstruction.cameras[first_image.camera_id]

print("\n--- Camera intrinsics ---")
print("Camera ID:", camera.camera_id)
print("Model:", camera.model_name)
print("Original resolution:", camera.width, "x", camera.height)
print("Parameter ordering:", camera.params_info)
print("Parameters:", camera.params)

K_original = np.asarray(camera.calibration_matrix())
print("Original intrinsic matrix K:")
print(K_original)

# Examine the corresponding downsampled RGB image.
rgb_path = image_directory / first_image.name

with Image.open(rgb_path) as rgb:
    small_width, small_height = rgb.size

scale_x = small_width / camera.width
scale_y = small_height / camera.height

K_small = K_original.copy()
K_small[0, 0] *= scale_x  # fx
K_small[0, 2] *= scale_x  # cx
K_small[1, 1] *= scale_y  # fy
K_small[1, 2] *= scale_y  # cy

print("\n--- Selected registered image ---")
print("Image ID:", first_image.image_id)
print("Filename:", first_image.name)
print("Downsampled resolution:", small_width, "x", small_height)
print("Scale factors:", scale_x, scale_y)
print("Downsampled intrinsic matrix K:")
print(K_small)

# COLMAP stores a world-to-camera rigid transformation:
#
#     p_camera = R @ p_world + t
#
cam_from_world = first_image.cam_from_world()

R = np.asarray(cam_from_world.rotation.matrix())
t = np.asarray(cam_from_world.translation)

world_to_camera = np.eye(4)
world_to_camera[:3, :3] = R
world_to_camera[:3, 3] = t

# Solve 0 = R @ C + t for the world-space camera center C.
camera_center_manual = -R.T @ t
camera_center_pycolmap = np.asarray(
    first_image.projection_center()
)

print("\n--- Camera extrinsics ---")
print("World-to-camera matrix:")
print(world_to_camera)
print("Translation t:", t)
print("Camera center from -R.T @ t:", camera_center_manual)
print("Camera center from PyCOLMAP:", camera_center_pycolmap)
print(
    "Center difference:",
    np.linalg.norm(
        camera_center_manual - camera_center_pycolmap
    ),
)

points = list(reconstruction.points3D.values())

xyz = np.stack([point.xyz for point in points])
colors = np.stack([point.color for point in points])
errors = np.array([point.error for point in points])
track_lengths = np.array(
    [point.track.length() for point in points]
)

print("\n--- Sparse 3D points ---")
print("Number of points:", len(points))
print("XYZ minimum:", xyz.min(axis=0))
print("XYZ maximum:", xyz.max(axis=0))
print("Median reprojection error:", np.median(errors))
print("Mean track length:", track_lengths.mean())
print("Maximum track length:", track_lengths.max())
print("First point XYZ:", xyz[0])
print("First point RGB:", colors[0])
