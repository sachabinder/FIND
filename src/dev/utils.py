import init_paths
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from typing import Tuple, Union
from tqdm import tqdm

from src.model.model import model_from_opts
from src.model.renderer import FootRenderer
from src.train.opts import Opts
from src.utils.logger import Logger
from src.dev.viz_tools import (
    add_image_title,
    create_combined_image,
    create_video_from_images,
    overlay_images,
)


from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance


class FootLatentVectorOptimizer:
    """
    Class to optimize the latent vectors of the model to fit the segmented image.
    """

    def __init__(
        self,
        name: str,
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
        self.name = name
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
            [self.posevec, self.shapevec], lr=learning_rate
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
        self,
        num_iterations: int,
        chamfer_supervision: bool = False,
        save_every: int = 10,
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

            if chamfer_supervision and self.gt_mesh is not None:
                temp_chamfer = chamfer_distance(
                    sample_points_from_meshes(
                        temp_pred_mesh, num_samples=1000, return_normals=False
                    ),
                    sample_points_from_meshes(
                        self.gt_mesh, num_samples=1000, return_normals=False
                    ),
                )
                temp_loss = temp_chamfer[0]

            # Backpropagate
            temp_loss.backward()
            self.optimizer.step()

            if i % save_every == 0:
                print("")
                data = {
                    "name": self.name,
                    "step": i,
                    "shapevec": self.shapevec.detach().cpu().numpy(),
                    "posevec": self.posevec.detach().cpu().numpy(),
                    "texvec": self.texvec.detach().cpu().numpy(),
                    "camera_rotation": self.camera_rotation,
                    "camera_translation": self.camera_translation,
                    "mesh": temp_pred_mesh.detach().cpu(),
                    "segm_image": temp_pred_segm_image.detach().cpu().numpy(),
                    "loss": temp_loss.item(),
                    "gt_segm_image": self.gt_segm_image.detach().cpu().numpy(),
                    "gt_mesh": self.gt_mesh if self.gt_mesh is not None else None,
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
                        f"Step {i} - Chamfer distance (μm): {chamf.cpu().detach().numpy() * 1e6} \n"
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

    def load_from_file(self, path: str) -> None:
        """
        Load an instance of the class from a pth file.
        """
        with open(path, "rb") as f:
            data = torch.load(f)

        # Load data
        self.data_history = data

        best_data = data[-1]

        # Load last optimization step data
        self.camera_rotation = best_data["camera_rotation"]
        self.camera_translation = best_data["camera_translation"]

        self.shapevec = torch.tensor(best_data["shapevec"], requires_grad=True)
        self.posevec = torch.tensor(best_data["posevec"], requires_grad=True)
        self.texvec = torch.tensor(best_data["texvec"], requires_grad=True)

        self.predicted_mesh = best_data["mesh"]
        self.predicted_segm_image = best_data["segm_image"]

        self.gt_segm_image = best_data["gt_segm_image"]
        self.gt_mesh = best_data["gt_mesh"]

    def plot_losses(self, path: str) -> None:
        """
        Plot the loss history.
        """

        file_path = path.split("/")
        file_path[-1] = "MSE-loss_" + file_path[-1]

        plt.plot(np.arange(len(self.loss_history)), self.loss_history)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss (MSE)")
        plt.savefig("/".join(file_path))
        plt.close()

        if len(self.chamfer_history) > 0:
            file_path = path.split("/")
            file_path[-1] = "Chamfer-loss_" + file_path[-1]

            plt.plot(np.arange(len(self.chamfer_history)), self.chamfer_history)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("Chamfer distance")
            plt.savefig("/".join(file_path))
            plt.close()

    def generate_optimized_silhouette_video(
        self, export_path: str, extrinsic_params: Tuple[torch.tensor] = None
    ) -> None:
        """
        Generate a video of the fitting process of the optimized silhouette,
        next to the ground truth silhouette.
        The video is by default from the initial point of view of the camera
        but it can be changed by setting the camera extrinsic parameters by modifying args.

        :param export_path: The path to export the video.
        :param extrinsic_params: The camera extrinsic parameters.
        """
        keys = self.data_history[0].keys()
        dict_of_lists = {key: [d[key] for d in self.data_history] for key in keys}

        # Generate the video
        if extrinsic_params is None:
            images = dict_of_lists["segm_image"]
            images = [
                create_combined_image(
                    add_image_title(
                        cv2.cvtColor(image[0, 0], cv2.COLOR_GRAY2BGR),
                        f"Prediction (step {i})",
                    ),
                    add_image_title(
                        cv2.cvtColor(
                            self.gt_segm_image[0, 0].detach().cpu().numpy(),
                            cv2.COLOR_GRAY2BGR,
                        ),
                        "Ground truth",
                    ),
                )
                for i, image in enumerate(images)
            ]

        else:
            meshes = dict_of_lists["mesh"]
            images = [
                create_combined_image(
                    add_image_title(
                        cv2.cvtColor(
                            self.render_function(
                                mesh,
                                extrinsic_params[0],
                                extrinsic_params[1],
                                return_mask=True,
                                mask_with_grad=True,
                            )["mask"][0, 0],
                            cv2.COLOR_GRAY2BGR,
                        ),
                        "Prediction",
                    ),
                    add_image_title(
                        cv2.cvtColor(self.gt_segm_image[0, 0], cv2.COLOR_GRAY2BGR),
                        "Ground truth",
                    ),
                )
                for mesh in meshes
            ]

        create_video_from_images(images, export_path, fps=2)

    def generate_optimized_overlay_video(self, export_path: str) -> None:
        """
        Generate a video of the fitting process of an overlay of the optimized
        silhouette and the ground truth silhouette.

        :param export_path: The path to export the video.
        """
        keys = self.data_history[0].keys()
        dict_of_lists = {key: [d[key] for d in self.data_history] for key in keys}

        # Generate the video
        images = dict_of_lists["segm_image"]
        images = [
            add_image_title(
                overlay_images(
                    self.gt_segm_image[0, 0].detach().cpu().numpy(), image[0, 0]
                ),
                f"Prediction (step {i})",
            )
            for i, image in enumerate(images)
        ]

        create_video_from_images(images, export_path, fps=2)


if __name__ == "__main__":
    pass
