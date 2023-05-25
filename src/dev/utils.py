import init_paths
import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Union
from tqdm import tqdm

from src.model.model import Model, process_opts, model_from_opts
from src.model.renderer import FootRenderer
from src.train.opts import Opts
from src.utils.logger import Logger

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance


class FootLatentVectorOptimizer:
    """
    Class to optimize the latent vectors of the model to fit the segmented image.
    """

    def __init__(
        self,
        model_options: Opts,
        logger: Logger,
        segmented_image: torch.tensor,
        renderer_function: FootRenderer,
        optimizer_function: torch.optim.Optimizer,
        learning_rate: float,
        loss_function: torch.nn.Module,
        gt_mesh: torch.tensor = None,
    ) -> None:
        """
        Initialize the optimizer.
        """
        # Logger
        self.logger = logger

        # Device, model and optimizer
        self.device = model_options.device
        self.model = model_from_opts(model_options)
        self.loss_function = loss_function
        self.data_history = []
        self.loss_history = []
        self.chamfer_history = []

        # Using the model to evaluate the segmented image
        self.model = self.model.eval().to(self.device)

        # Images, rendering and mesh
        self.gt_segm_image = segmented_image

        self.image_size = self.gt_segm_image.shape[-1]
        self.render_function = renderer_function(
            image_size=self.image_size, device=self.device
        )
        self.predicted_segm_image = None
        self.predicted_mesh = None

        # Camera extrinsic parameters
        # TODO transform as learnable parameters
        self.camera_rotation = None
        self.camera_translation = None

        # Latent vectors, random initialization
        self.shapevec = torch.randn(
            1, self.model.shapevec_size, requires_grad=True, device=self.device
        )
        self.posevec = torch.randn(
            1, self.model.posevec_size, requires_grad=True, device=self.device
        )
        self.texvec = torch.randn(
            1, self.model.texvec_size, requires_grad=True, device=self.device
        )

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optimizer_function(
            [self.shapevec, self.posevec, self.texvec], lr=learning_rate
        )

        # Ground truth mesh
        self.gt_mesh = gt_mesh

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
        self, num_iterations: int, save_every: int = 10
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Optimize the latent vectors of the model to fit the segmented image.

        :param num_iterations: The number of iterations to run.
        :return: The optimized latent vectors.
        """
        if self.camera_rotation is None or self.camera_translation is None:
            raise ValueError(
                "Camera extrinsic parameters must be set before optimizing."
            )

        # Optimization loop
        for i in range(num_iterations):
            self.optimizer.zero_grad()

            # Generate the mesh
            temp_prediction = self.model.get_meshes(
                shapevec=self.shapevec, posevec=self.posevec, texvec=self.texvec
            )

            temp_pred_mesh = temp_prediction["meshes"]

            # Render segmented prediction
            temp_out_render = self.render_function(
                temp_pred_mesh,
                self.camera_rotation,
                self.camera_translation,
                return_mask=True,
                mask_with_grad=True,
            )
            temp_pred_segm_image = temp_out_render["mask"]

            # Compute loss
            temp_loss = self.loss_function(temp_pred_segm_image, self.gt_segm_image)
            self.loss_history.append(temp_loss.item())

            # Backpropagate
            temp_loss.backward()
            self.optimizer.step()

            if i % save_every == 0:
                print("")
                data = {
                    "step": i,
                    "shapevec": self.shapevec.detach().cpu().numpy(),
                    "posevec": self.posevec.detach().cpu().numpy(),
                    "texvec": self.texvec.detach().cpu().numpy(),
                    "camera_rotation": self.camera_rotation,
                    "camera_translation": self.camera_translation,
                    "mesh": temp_pred_mesh.detach().cpu(),
                    "segm_image": temp_pred_segm_image.detach().cpu().numpy(),
                    "loss": temp_loss.item(),
                }

                # Compute chamfer distance
                if self.gt_mesh is not None:
                    samples = 10000
                    gt_pts = sample_points_from_meshes(
                        self.gt_mesh, num_samples=samples
                    )
                    pred_pts = sample_points_from_meshes(
                        temp_pred_mesh, num_samples=samples
                    )
                    chamf, _ = chamfer_distance(gt_pts, pred_pts)
                    self.logger.write(
                        f"Step {i} - Chamfer distance (μm): {chamf.cpu().detach().numpy() * 1e6}"
                    )

                    data["chamfer_distance"] = chamf.cpu().detach().numpy()
                    self.chamfer_history.append(chamf.cpu().detach().numpy())

                self.logger.write(f"Step {i} - Loss: {temp_loss.item()}")
                self.data_history.append(data)

            else:
                self.logger.add_to_log(f"Step {i} - Loss: {temp_loss.item()}")

        # Update prediction
        self.predicted_segm_image = temp_pred_segm_image
        self.predicted_mesh = temp_pred_mesh

        return self.shapevec, self.posevec, self.texvec

    def save_history(self, path: str) -> None:
        """
        Save the history of the optimization.

        :param path: The path to save the history.
        """
        with open(path, "wb") as f:
            torch.save(self.data_history, f)

    def plot_loss(self, path: str) -> None:
        """
        Plot the loss history.
        """

        plt.plot(np.arange(len(self.loss_history)), self.loss_history)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss (MSE)")
        plt.savefig(path)
        plt.close()


if __name__ == "__main__":
    pass
