# Gaussian Worlds

An implementation of 3D Gaussian Splatting for reconstructing a 3D scene from multi-view 2D photographs.

<p align="center">
  <img width="810" height="480" alt="image" src="https://github.com/user-attachments/assets/4d8d6081-c552-4e6b-9c02-d0b2f71edb1e" />
</p>

*Post-training comparison using the combined L1 and SSIM loss. From left to right: the target photograph, the optimized Gaussian-splat render, and the accumulated opacity map showing the scene regions covered by overlapping Gaussians.*

## Future Directions

The current system initializes a fixed set of Gaussians from a sparse COLMAP reconstruction and optimizes their positions, scales, rotations, colors, and opacities using multi-view supervision. Planned extensions include:

- **Densification:** Identify under-reconstructed and over-reconstructed regions at each training iteration.
- **Pruning:** Remove nearly transparent or excessively large Gaussians.
- **Lens-undistortion:** Undistort the input photographs before optimization.
- **Spherical-harmonics:** Replace constant RGB colors with spherical-harmonic coefficients.
- **Interactive scene inspection:** Add a viewer for navigating the reconstructed 3D scene.
- **Generative editing:** Explore inserting fantasy elements into the 3D scene.
