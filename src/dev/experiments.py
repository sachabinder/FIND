"""
File of experiments to run.
"""
import init_paths
import torch
import os
import datetime

from torch.utils.data import DataLoader
from adabelief_pytorch import AdaBelief
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm

from src.dev.utils import FootLatentVectorOptimizer, ExtrinsicParamsOptimizer
from src.utils.logger import Logger
from src.model.renderer import FootRenderer
from src.model.model import process_opts
from src.model.renderer import FootRenderer
from src.data.dataset import Foot3DDataset, BatchCollator
from src.eval.eval_metrics import IOU


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
    logger.write(
        f"\n\n\n ################ Experiment {exp_name} ran at {datetime.datetime.now()} ################ \n\n\n"
    )
    imsize = 256
    renderer = FootRenderer(image_size=imsize, device=device)
    R, T = renderer.view_from("45")

    for template in dataloader:
        logger.write(f"Optimizing latent vector for foot {template['name'][0]} \n")

        gt_mesh = template["mesh"]
        gt_rendered = renderer(gt_mesh, R, T, return_mask=True, mask_with_grad=True)
        gt_segmented = gt_rendered["mask"]

        optimizer = FootLatentVectorOptimizer(
            name=template["name"][0],
            model_options=opts,
            logger=logger,
            segmented_image=gt_segmented,
            renderer_function=FootRenderer,
            optimizer_function=torch.optim.SGD,
            learning_rate=0.1,
            loss_function=torch.nn.functional.l1_loss,
            gt_mesh=gt_mesh,
        )

        optimizer.set_camera_extrinsic_parameters((R, T))

        optimizer.optimize(num_iterations=1000, chamfer_supervision=False)

        logger.save()

        # export the data
        optimizer.save_history(f"exp/{exp_name}/history/{template['name'][0]}.pth")
        optimizer.plot_losses(f"exp/{exp_name}/plots/{template['name'][0]}.png")
        optimizer.generate_optimized_silhouette_video(
            f"exp/{exp_name}/plots/{template['name'][0]}.mp4"
        )
        optimizer.generate_optimized_overlay_video(
            f"exp/{exp_name}/plots/{template['name'][0]}_overlay.mp4"
        )

        del optimizer


def generate_renders(file_path: str, exp_name: str):
    """
    Function generating the renders for a given latent vector.

    :param file_path: path of pth file to load
    """
    out_dir = f"exp/{exp_name}"
    imsize = 256
    optimizer = FootLatentVectorOptimizer(
        name="0003-A",
        model_options=opts,
        logger=None,
        segmented_image=torch.zeros((1, 1, imsize, imsize), device=device),
        renderer_function=FootRenderer,
        optimizer_function=torch.optim.Adam,
        learning_rate=0.01,
        loss_function=torch.nn.functional.mse_loss,
        gt_mesh=None,
    )
    optimizer.load_from_file(file_path)
    optimizer.generate_optimized_overlay_video(
        f"exp/{exp_name}/plots/{optimizer.name}_overlay.mp4"
    )


def optimize_extrinsic(tamplate_mesh_dataloader: DataLoader, exp_name):
    out_dir = f"exp/{exp_name}"
    logger = Logger(logfile=os.path.join(out_dir, "log.txt"))
    logger.write(
        f"\n\n\n ################ Experiment {exp_name} ran at {datetime.datetime.now()} ################ \n\n\n"
    )
    imsize = 256
    renderer = FootRenderer(image_size=imsize, device=device)

    # ground truth camera parameters
    R, T = renderer.view_from("topdown")

    for template in tamplate_mesh_dataloader:
        logger.write(f"Optimizing camera params with {template['name'][0]} \n")

        template_mesh = template["mesh"]
        gt_rendered = renderer(
            template_mesh, R, T, return_mask=True, mask_with_grad=True
        )
        gt_segmented = gt_rendered["mask"]

        optimizer = ExtrinsicParamsOptimizer(
            template_mesh=template_mesh,
            device=device,
            logger=logger,
            segmented_image=gt_segmented,
            renderer_function=FootRenderer,
            optimizer_function=torch.optim.SGD,
            learning_rate=0.1,
            loss_function=torch.nn.functional.mse_loss,
        )
        optimizer.optimize(num_iterations=1000)


# Experiment 1: find the best loss L1 vs L2
def test_loss_l1(dataloader: DataLoader, exp_name: str = "loss_l1"):
    """
    Function taking a dataloader as input and finding the best loss function
    by extrapolating losses form the dataset.
    """
    out_dir = f"exp/{exp_name}"
    logger = Logger(logfile=os.path.join(out_dir, "log.txt"))

    logger.add_to_log(
        f"\n\n\n ################ Experiment {exp_name} ran at {datetime.datetime.now()} ################ \n\n\n"
    )
    print(
        f"\n\n\n ################ Experiment {exp_name} ran at {datetime.datetime.now()} ################ \n\n\n"
    )

    imsize = 256
    renderer = FootRenderer(image_size=imsize, device=device)
    R, T = renderer.view_from("45")
    NUM_ITERATIONS = 2000

    logger.add_to_log(f"Tamplate name, L1 loss, L2 loss, chamfer distance (µm), IoU")
    print(f"Tamplate name, L1 loss, L2 loss, chamfer distance (µm), IoU")

    for template in dataloader:
        gt_mesh = template["mesh"]
        gt_rendered = renderer(gt_mesh, R, T, return_mask=True, mask_with_grad=True)
        gt_segmented = gt_rendered["mask"]

        optimizer = FootLatentVectorOptimizer(
            name=template["name"][0],
            model_options=opts,
            logger=logger,
            segmented_image=gt_segmented,
            renderer_function=FootRenderer,
            optimizer_function=torch.optim.SGD,
            learning_rate=0.01,
            loss_function=torch.nn.functional.l1_loss,
            gt_mesh=gt_mesh,
        )

        optimizer.set_camera_extrinsic_parameters((R, T))
        final_loss = optimizer.optimize(
            num_iterations=NUM_ITERATIONS, chamfer_supervision=False
        )

        # coputation of interesing metrics
        predicted_mesh = optimizer.predicted_mesh
        predicted_segm_image = optimizer.predicted_segm_image

        l2_loss = torch.nn.functional.mse_loss(predicted_segm_image, gt_segmented)
        l1_loss = torch.nn.functional.l1_loss(predicted_segm_image, gt_segmented)
        iou_metric = IOU(predicted_segm_image, gt_segmented)

        # compute the chamfer distance
        samples = 10000
        gt_pts = sample_points_from_meshes(gt_mesh, num_samples=samples)
        pred_pts = sample_points_from_meshes(predicted_mesh, num_samples=samples)
        chamf, _ = chamfer_distance(gt_pts, pred_pts)

        # log the results
        logger.add_to_log(
            f"{template['name'][0]}, {l1_loss}, {l2_loss}, {chamf*1e6}, {iou_metric}"
        )
        print(f"{template['name'][0]}, {l1_loss}, {l2_loss}, {chamf*1e6}, {iou_metric}")

        logger.save()


# Experiment 1: find the best learning rate
def find_best_learning_rate(dataloader: DataLoader, exp_name: str = "best_lr"):
    """
    Function taking a dataloader as input and finding the best learning rate
    by extrapolating losses form the dataset.
    """
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    out_dir = f"exp/{exp_name}"
    logger = Logger(logfile=os.path.join(out_dir, "log.txt"))
    logger.write(
        f"\n\n\n ################ Experiment {exp_name} ran at {datetime.datetime.now()} ################ \n\n\n"
    )


if __name__ == "__main__":
    # =============== Dataset config ===============
    """
    Here, we will work with only one foot for testing.
    This foot is the same as the one used for the tamplate.
    """
    collate_fn = BatchCollator(device=device).collate_batches
    template_foot = [
        "0008",
    ]
    template_dset = Foot3DDataset(left_only=True, device=device)
    print(f"Dataset size: {len(template_dset)}")
    template_loader = DataLoader(template_dset, shuffle=False, collate_fn=collate_fn)

    # optimize_on_dataset(template_loader, "latent_optimization")
    # generate_renders(
    #     f"exp/latent_optimization/history/{template_foot}-A.pth", "latent_optimization"
    # )
    test_loss_l1(template_loader, "loss_l1")
