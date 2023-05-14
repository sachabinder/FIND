import init_paths
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

from src.dev.utils import optimize_latent_vectors
from src.model.model import process_opts, model_from_opts
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

# Load model
model = model_from_opts(opts)
model = model.eval().to(device)


# =============== Renderer config ===============
"""
We first fix extrinsic parameters (R, T) for the camera.
We will then use the same parameters for all the feet.
"""
imsize = 256
renderer = FootRenderer(image_size=imsize, device=device)
R, T = renderer.linspace_views(nviews=1, dist=0.5, elev_min=50, elev_max=90)

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

meshes = []
for _, template in enumerate(template_loader):
    meshes.append(template["mesh"])

if __name__ == "__main__":
    for _, template in enumerate(template_loader):
        gt_mesh = template["mesh"]

        # get the segmented image from the dataset
        gt_out = renderer(
            gt_mesh,
            R,
            T,
            return_images=False,
            return_mask=True,
            mask_with_grad=False,
        )
        # compute the optimal latent vectors
        shape, pose, tex = optimize_latent_vectors(
            segmented_image=gt_out["mask"],
            model=model,
            renderer=renderer,
            R=R,
            T=T,
            device=device,
            my_optimizer=torch.optim.Adam,
            learning_rate=0.01,
            loss_function=torch.nn.functional.mse_loss,
        )

        with torch.no_grad():
            # passforward the model
            prediction = model.get_meshes(shapevec=shape, posevec=pose, texvec=tex)

            # Out the mesh and vizualize it
            prediction_mesh = prediction["meshes"]
            prediction_out = renderer(
                prediction_mesh,
                R,
                T,
                return_images=True,
                return_mask=True,
                mask_with_grad=False,
            )
            image = prediction_out["image"][0, 0].cpu().numpy()
            mask = prediction_out["mask"][0, 0].cpu().numpy()

            # Calculate chamfer losses
            samples = 10000
            gt_pts = sample_points_from_meshes(gt_mesh, num_samples=samples)
            pred_pts = sample_points_from_meshes(prediction_mesh, num_samples=samples)
            chamf, _ = chamfer_distance(gt_pts, pred_pts)
            print(f"Chamfer distance (μm): {chamf.cpu().detach().numpy() * 1e6}")

            # save the image
            plt.imsave("tamplate_image.png", image)
            plt.imsave("optim_mask.png", mask, cmap="gray")
