import torch

from typing import Tuple

from src.model.model import Model
from src.model.renderer import FootRenderer


def optimize_latent_vectors(
    segmented_image: torch.tensor,
    model: Model,
    renderer: FootRenderer,
    R: torch.tensor,
    T: torch.tensor,
    device: torch.device,
    my_optimizer: torch.optim.Optimizer,
    learning_rate: float,
    loss_function: torch.nn.Module,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Optimize the latent vectors of the model to fit the segmented image.
    Here, the camera extrinsic parameters (R, T) are fixed.

    :param segmented_image: The segmented image to fit.
    :param model: The model to optimize.
    :param renderer: The renderer to render the mesh.
    :param R: The rotation matrix of the camera.
    :param T: The translation vector of the camera.
    :param device: The device to use.
    :param my_optimizer: The optimizer to use.
    :param learning_rate: The learning rate to use.
    :param loss_function: The loss function to use.
    :return: The optimized latent vectors.
    """
    # Initialize latent vectors
    shapevec = torch.randn(1, 100, requires_grad=True, device=device)
    posevec = torch.randn(1, 100, requires_grad=True, device=device)
    textvec = torch.randn(1, 100, requires_grad=True, device=device)

    # Set up optimizer
    optimizer = my_optimizer([shapevec, posevec, textvec], lr=learning_rate)

    # Optimization loop
    num_iterations = 1000
    for i in range(num_iterations):
        optimizer.zero_grad()

        # Generate mesh
        prediction = model.get_meshes(
            shapevec=shapevec, posevec=posevec, texvec=textvec
        )
        mesh = prediction["meshes"]

        # Render mask
        out = renderer(
            mesh, R, T, return_images=False, return_mask=True, mask_with_grad=False
        )
        rendered_mask = out["mask"]

        # Calculate loss
        loss = loss_function(rendered_mask, segmented_image)

        # Backward pass
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Iteration {i}: Loss: {loss.item()}")

    return shapevec.detach(), posevec.detach(), textvec.detach()


if __name__ == "__main__":
    pass
