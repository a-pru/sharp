import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

from src.datamodules.av2_dataset import Av2Dataset, collate_fn
from src.utils.vis import visualize_scenario
from hydra.utils import instantiate
from omegaconf import OmegaConf


def ptsToGlobal(origin, theta, pts):
    bs = pts.shape[0]
    origin = origin.view(bs, 1, 1, 2)
    theta = theta.view(bs, 1)
    rotate_mat = torch.stack(
        [
            torch.cos(theta),
            torch.sin(theta),
            -torch.sin(theta),
            torch.cos(theta),
        ],
        dim=1,
    ).reshape(bs, 2, 2)

    if len(pts.shape) < 4: pts = pts.unsqueeze(-2)
    with torch.no_grad():
        global_trajectory = (
            torch.matmul(pts[..., :2].double(), rotate_mat.unsqueeze(1).double())
            + origin
        )
    return global_trajectory.squeeze(-2)


def main():
    plt.rcParams["pdf.fonttype"] = 42
    torch.set_printoptions(sci_mode=False)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    split = "val"
    data_root = Path("TODO_UPDATE")
    data_root = Path("/data/datasets/av2/DeMo_processed/")
    chkpt_dir = "exps/av2_single_agent/checkpoints/" # Use pretrained model checkpoint for visualization (TODO: update with your own checkpoint path)
    av2_raw_data_dir = Path("TODO_UPDATE") / split   
    av2_raw_data_dir = Path("/data/datasets/av2/motion-forecasting/") / split  


    chkpt = "{}/{}".format(chkpt_dir, sorted(os.listdir(chkpt_dir))[-1])
    cfg = OmegaConf.load(chkpt_dir + "../.hydra/config.yaml")
    print("LOAD", chkpt)

    model = instantiate(cfg.model.pl_module)
    model.load_chkpt(chkpt)
    model = model.eval().cuda()
    model.on_validation_start()

    B = 32
    dataset = Av2Dataset(data_root=data_root, split=split, split_points=[10, 20, 30, 40, 50], num_historical_steps=10) #TODO Update to align with model config
    dataloader = TorchDataLoader(
        dataset,
        batch_size=B,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    ###################################################################################################################################################################################################

    for seq_data in tqdm(dataloader):
        for d in range(len(seq_data)):
            for k in seq_data[d].keys():
                if torch.is_tensor(seq_data[d][k]): seq_data[d][k] = seq_data[d][k].cuda()      
    
        with torch.no_grad():
            all_out = model.validation_step(seq_data, 0)

        for i, ao in enumerate(all_out): 
            data = seq_data[i]
            origin = data['origin'].view(-1, 1, 1, 2).double()
            theta = data['theta'].double()

            rotate_mat = torch.stack(
                [
                    torch.cos(theta),
                    torch.sin(theta),
                    -torch.sin(theta),
                    torch.cos(theta),
                ],
                dim=1,
            ).reshape(-1, 2, 2)

            with torch.no_grad():
                global_trajectory = (
                    torch.matmul(ao["y_hat"][..., :2].double(), rotate_mat.unsqueeze(1))
                    + origin
                )
                all_out[i]["y_hat_global"] = global_trajectory

        data = seq_data[-1]
        for b in range(len(data["scenario_id"])):
            y = data["target"][b]
            plt.figure(figsize=(6, 6))  

            scene_id = data["scenario_id"][b]

            scene_file = av2_raw_data_dir / scene_id / ("scenario_" + scene_id + ".parquet")
            map_file = av2_raw_data_dir / scene_id / ("log_map_archive_" + scene_id + ".json")
            scenario = scenario_serialization.load_argoverse_scenario_parquet(scene_file)
            static_map = ArgoverseStaticMap.from_json(map_file)

            out = all_out[-1]
            prob = out["pi"][b]
            prediction_local = out["y_hat"][b]
            prediction = out["y_hat_global"][b].cpu().numpy()

            ep_local = prediction_local[:, -1, :2]
            error = torch.min(torch.norm(y[0, -1].unsqueeze(0).repeat(6, 1) - ep_local, dim=-1)).item()
            title = "t={}s (minFDE-{}: {:5.3f})".format(5, 6, error) 
            visualize_scenario(scenario, static_map, title=title, prediction=prediction, tight=True, timestep=int(data["timestamp"][0]*10), create_fig=False)
            plt.show()
            plt.close()
    return


if __name__ == "__main__":
    main()
