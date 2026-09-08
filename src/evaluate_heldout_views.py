from pathlib import Path
import numpy as np
import pycolmap
import torch

from train_scene_gaussians import (
    IMAGE_DIR,
    SPARSE_DIR,
    evaluate_training_views,
    load_training_view,
    save_render_comparison,
)


CHECKPOINT_PATH = Path(
    "outputs/scene_training/fixed_gaussians.pt"
)

OUTPUT_DIR = Path(
    "outputs/scene_training/heldout"
)


def get_matching_images(
    reconstruction,
    image_directory,
):
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

    return matching_images


def select_heldout_images(
    matching_images,
    training_names,
):
    """
    Return every available image that was not stored as a training image.
    """

    return [
        image
        for image in matching_images
        if image.name not in training_names
    ]


def load_saved_gaussians(
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    gaussians = {
        name: tensor.to(device)
        for name, tensor
        in checkpoint["gaussians"].items()
    }

    return gaussians, checkpoint["training_image_names"]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for rendering.")

    device = torch.device("cuda")
    reconstruction = pycolmap.Reconstruction(SPARSE_DIR)

    gaussians, training_names = load_saved_gaussians(
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )

    matching_images = get_matching_images(
        reconstruction=reconstruction,
        image_directory=IMAGE_DIR,
    )

    heldout_images = select_heldout_images(
        matching_images=matching_images,
        training_names=set(training_names),
    )

    heldout_views = [
        load_training_view(
            reconstruction=reconstruction,
            colmap_image=colmap_image,
            image_directory=IMAGE_DIR,
            device=device,
        )
        for colmap_image in heldout_images
    ]

    print(f"Training images in checkpoint: {len(training_names)}")
    print(f"Selected held-out images: {len(heldout_views)}")

    for index, view in enumerate(heldout_views):
        print(f"  [{index}] {view['name']}")

    metrics = evaluate_training_views(
        gaussians=gaussians,
        training_views=heldout_views,
    )

    print(
        "\nHeld-out results: "
        f"mean L1={metrics['mean_l1']:.6f}, "
        f"mean PSNR={metrics['mean_psnr']:.2f} dB"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for index, view in enumerate(heldout_views):
        save_render_comparison(
            gaussians=gaussians,
            view=view,
            output_path=(
                OUTPUT_DIR
                / f"heldout_{index:02d}_{view['name']}"
            ),
        )

    print(f"Saved held-out comparisons to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
