"""
File of experiments to run.
"""
import init_paths
import torch
import os

from torch.utils.data import DataLoader
from src.dev.utils import FootLatentVectorOptimizer
from src.utils.logger import Logger
from src.model.renderer import FootRenderer
from src.model.model import process_opts
from src.model.renderer import FootRenderer
from src.data.dataset import Foot3DDataset, BatchCollator

# =============== Device config ===============
gpu = 1
if torch.cuda.is_available():
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
else:
    device = "cpu"

# =============== Model config ===============
opts = "exp/3D_only/FIND/opts.yaml"
opts = process_opts(opts, eval=True)
opts.load_model = "exp/3D_only/FIND/model_best.pth"
opts.device = device


def optimize_on_dataset(dataloader: DataLoader, exp_name: str):
    """
    Function taking a dataloader as input and optimizing the latent vectors
    over the entire dataset of foot 3d scans.

    input segmented image are artificially generated from the foot 3d scans
    for a given fixed camera position.

    :param dataloader: dataloader of the dataset
    """
    out_dir = f"exp/{exp_name}"
    logger = Logger(logfile=os.path.join(out_dir, "log.txt"))
    imsize = 256
    renderer = FootRenderer(image_size=imsize, device=device)
    R, T = renderer.view_from("topdown")

    for template in dataloader:
        logger.write(f"Optimizing latent vector for foot {template['name']} \n")

        gt_mesh = template["mesh"]
        gt_rendered = renderer(gt_mesh, R, T, return_mask=True, mask_with_grad=False)
        gt_segmented = gt_rendered["mask"]

        optimizer = FootLatentVectorOptimizer(
            model_options=opts,
            logger=logger,
            segmented_image=gt_segmented,
            renderer_function=FootRenderer,
            optimizer_function=torch.optim.Adam,
            learning_rate=0.001,
            loss_function=torch.nn.functional.mse_loss,
            gt_mesh=gt_mesh,
        )

        optimizer.set_camera_extrinsic_parameters((R, T))

        optimizer.optimize(num_iterations=1000)

        logger.save()

        # export the data
        optimizer.save_history(f"exp/{exp_name}/history/{template['name']}.pth")
        optimizer.plot_loss(f"exp/{exp_name}/plots/{template['name']}.png")


if __name__ == "__main__":
    # =============== Dataset config ===============
    """
    Here, we will work with only one foot for testing.
    This foot is the same as the one used for the tamplate.
    """
    collate_fn = BatchCollator(device=device).collate_batches
    template_foot = "0003"
    template_dset = Foot3DDataset(
        left_only=True, tpose_only=True, specific_feet=[template_foot], device=device
    )
    template_loader = DataLoader(template_dset, shuffle=False, collate_fn=collate_fn)

    optimize_on_dataset(template_loader, "latent_optimization")
