import torch

from typing import Tuple, Union

from src.model.model import Model, process_opts, model_from_opts
from src.model.renderer import FootRenderer
from src.train.opts import Opts


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


class FootLatentVectorOptimizer:
    """
    Class to optimize the latent vectors of the model to fit the segmented image.
    """

    def __init__(
        self,
        model_options: Opts,
        segmented_image: torch.tensor,
        renderer_function: FootRenderer,
        optimizer_function: torch.optim.Optimizer,
        learning_rate: float,
        loss_function: torch.nn.Module,
    ) -> None:
        """
        Initialize the optimizer.
        """

        # Device, model and optimizer
        self.device = model_options.device
        self.model = model_from_opts(model_options)
        self.loss = loss_function

        # Using the model to evaluate the segmented image
        self.model = self.model.eval().to(self.device)

        # Images, rendering and mesh
        self.gt_segm_image = segmented_image
        self.image_size = self.gt_segm_image.shape[0]
        self.render_function = renderer_function(
            image_size=self.image_size, device=self.device
        )
        self.predicted_segm_image = None
        self.predicted_mesh = None

        # Camera extrinsic parameters
        # TODO transform as learnable parameters
        self.camera_rotation = None
        self.camera_translation = None

        # Latent vectors
        self.shapevec = torch.randn(
            1, self.model.shapevec_size, requires_grad=True, device=self.device
        )
        self.posevec = torch.randn(
            1, self.model.posevec_size, requires_grad=True, device=self.device
        )
        self.textvec = torch.randn(
            1, self.model.textvec_size, requires_grad=True, device=self.device
        )

        # Optimizer
        self.optimizer = optimizer_function(
            [self.shapevec, self.posevec, self.textvec], lr=learning_rate
        )

    def set_camera_extrinsic_parameters(
        self, param: Union[str, Tuple[torch.tensor, torch.tensor]]
    ) -> None:
        """
        Set the camera extrinsic parameters.

        :param param: Either a string to load the parameters from a file, or a tuple
            (R, T) containing the rotation matrix and the translation vector.
        """
        if isinstance(param, str):
            (
                self.camera_rotation,
                self.camera_translation,
            ) = self.render_function.view_from(param)
        elif isinstance(param, tuple):
            self.camera_rotation, self.camera_translation = param
        else:
            raise TypeError(
                f"Parameter must be either a string or a tuple, not {type(param)} or str should be in ['topdown', 'side1', 'side2', 'toes', '45', '60']."
            )

    def optimize(
        self, num_iterations: int
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        pass


if __name__ == "__main__":
    pass
