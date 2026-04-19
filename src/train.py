import torch
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from torchvision import transforms
import subprocess

from src.map_model import MapModel
from src.bayes_model import BayesModel
from src.base_model import BaseModel
from src.helper_funcs import generate_Z_x, create_lrs, crop_y_obs_to_patch
from src.grid_funcs import build_grid_params

# Initialize Parameters
hr_shape = torch.tensor([128, 128])
K = 16
beta = 400
downsample_ratio = 4
shift_range = [-2, 2]
rot_range = [-4, 4]
gamma = 2.0

A = 0.04
r = 1.0

v_params = build_grid_params(
    hr_shape=hr_shape,
    downsample_ratio=downsample_ratio,
    lr_patch=None,
    hr_margin=0,
)

Z_x = generate_Z_x(v_params, A, r)
Z_x_inv = torch.linalg.inv(Z_x)

v_params_patch = build_grid_params(
    hr_shape=hr_shape,
    downsample_ratio=downsample_ratio,
    lr_patch=(12, 12, 8, 8),
    hr_margin=6,
)

Z_x_patch = generate_Z_x(v_params_patch, A, r)
Z_x_patch_inv = torch.linalg.inv(Z_x_patch)


# FOLDERS
results_dir = Path("imgs/results")
if results_dir.exists():
    shutil.rmtree(results_dir)
results_dir.mkdir(parents=True, exist_ok=True)
(results_dir / "lr").mkdir(exist_ok=False)

# High Resolution Image
hr_img = Image.open("imgs/hr.jpg")
hr_img = hr_img.resize(hr_shape.tolist()[::-1])
hr_img.save("imgs/hr_small.png")
hr_img = transforms.ToTensor()(hr_img)

print(f"Creating {K} Observations")
y_obs = create_lrs(
    hr_img=hr_img,
    downsample_ratio=downsample_ratio,
    shift_range=shift_range,
    rot_range=rot_range,
    gamma=gamma,
    beta=beta,
    K=K,
    save_file=results_dir / "lr",
)
y_obs_patch = crop_y_obs_to_patch(y_obs, v_params_patch.lr_bounds)

# MODEL
print("Building Model (Inverting Z_x takes a minute)")
# Z_x_inv = torch.linalg.inv(Z_x)

map_model = MapModel(
    v_params=v_params,
    K=K,
    beta=beta,
    Z_x=Z_x,
    Z_x_inv=Z_x_inv,
)

bayes_model = BayesModel(
    v_params=v_params_patch,
    K=K,
    beta=beta,
    Z_x=Z_x_patch,
    Z_x_inv=Z_x_patch_inv,
)

def train(model: BaseModel, y_obs: torch.Tensor, steps: int = 1000, name: str = "empty"):
    recons_dir = results_dir / name / "recons"
    losses_dir = results_dir / name / "loss"
    recons_dir.mkdir(exist_ok=False, parents=True)
    losses_dir.mkdir(exist_ok=False, parents=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_obs = y_obs.to(device, dtype=torch.float32)
    model.to(device)

    losses = []

    print("Beginning Training")
    for step in tqdm(range(steps), desc=f"Optimizing {name}"):
        optimizer.zero_grad()
        loss = model(y_obs)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 50 == 0:
            with torch.no_grad():
                out_img = model.get_HR(y_obs).detach().cpu().numpy()
                out_display = ((out_img + 0.5) * 255).clip(0, 255).astype("uint8")
                Image.fromarray(out_display).save(recons_dir / f"recon_{step:05d}.png")

            plt.figure()
            plt.plot(losses)
            plt.xlabel("# Steps")
            plt.ylabel("Loss")
            plt.savefig(losses_dir / "loss_plot.png")
            plt.close()

    subprocess.run(
        f"ffmpeg -y -framerate 10 -pattern_type glob -i '{recons_dir}/*.png' {recons_dir}/output.gif",
        shell=True,
        check=True,
        text=True,
        capture_output=True,
    )

train(bayes_model, y_obs_patch, steps=10_000, name="bayes")
train(map_model, y_obs, steps=1000, name="map")