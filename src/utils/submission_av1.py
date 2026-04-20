import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch import Tensor

TYPE_LIST = Union[List[np.ndarray], np.ndarray]


def generate_forecasting_h5(
    data: Dict[int, TYPE_LIST],
    output_path: str,
    filename: str = "argoverse_forecasting_baseline",
    probabilities: Optional[Dict[int, List[float]]] = None,
) -> None:
    """
    Helper function to generate the result h5 file for argoverse forecasting challenge

    Args:
        data: a dictionary of trajectories, with the key being the sequence ID, and value being
              predicted trajectories for the sequence, stored in a (n,30,2) np.ndarray.
              "n" can be any number >=1. If probabilities are provided, the evaluation server
              will use the top-K most likely forecasts for any top-K metric. If probabilities
              are unavailable, the first-K trajectories will be evaluated instead. Each
              predicted trajectory should consist of 30 waypoints.
        output_path: path to the output directory to store the output h5 file
        filename: to be used as the name of the file
        probabilities (optional) : normalized probability for each trajectory
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    hf = h5py.File(os.path.join(output_path, filename + ".h5"), "w")
    future_frames = 30
    d_all: List[np.ndarray] = []
    counter = 0
    for key, value in data.items():
        print("\r" + str(counter + 1) + "/" + str(len(data)), end="")

        if isinstance(value, List):
            value = np.array(value)
        assert value.shape[1:3] == (
            future_frames,
            2,
        ), f"ERROR: the data should be of shape (n,30,2), currently getting {value.shape}"

        n = value.shape[0]
        len_val = len(value)
        value = value.reshape(n * future_frames, 2)
        if probabilities is not None:
            assert key in probabilities.keys(), f"missing probabilities for sequence {key}"
            assert (
                len(probabilities[key]) == len_val
            ), f"mismatch sequence and probabilities len for {key}: {len(probabilities[key])} !== {len_val}"
            # assert np.isclose(np.sum(probabilities[key]), 1), "probabilities are not normalized"

            d = np.array(
                [
                    [
                        int(key),
                        np.float32(x),
                        np.float32(y),
                        probabilities[key][int(np.floor(i / future_frames))],
                    ]
                    for i, (x, y) in enumerate(value)
                ]
            )
        else:
            d = np.array([[key, np.float32(x), np.float32(y)] for x, y in value])

        d_all.append(d)
        counter += 1

    d_all = np.concatenate(d_all, 0)
    hf.create_dataset("argoverse_forecasting", data=d_all, compression="gzip", compression_opts=9)
    hf.close()


class SubmissionAv1:
    def __init__(self, save_dir: str = "") -> None:
        stamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.output_path = save_dir
        self.submission_fname = f"single_agent_{stamp}"
        self.challenge_submission_data = {"traj": {}, "prob": {}}

    def format_data(
        self,
        data: dict,
        trajectory: Tensor,
        probability: Tensor,
        normalized_probability=False,
        inference=False,
    ) -> None:
        """
        trajectory: (B, M, 60, 2)
        probability: (B, M)
        normalized_probability: if the input probability is normalized,
        """
        scenario_ids = data["scenario_id"]
        batch = len(scenario_ids)

        origin = data["origin"].view(batch, 1, 1, 2).double()
        theta = data["theta"].double()

        rotate_mat = torch.stack(
            [
                torch.cos(theta),
                torch.sin(theta),
                -torch.sin(theta),
                torch.cos(theta),
            ],
            dim=1,
        ).reshape(batch, 2, 2)

        trajectory = trajectory[:, :, :30]

        with torch.no_grad():
            global_trajectory = (
                torch.matmul(trajectory[..., :2].double(), rotate_mat.unsqueeze(1))
                + origin
            )
            if not normalized_probability:
                probability = torch.softmax(probability.double(), dim=-1)

        global_trajectory = global_trajectory.detach().cpu().float().numpy()
        probability = probability.detach().cpu().float().numpy()

        """
        def transform_coord(coords, angle, org):
            x = coords[..., 0]
            y = coords[..., 1]
            x_transform = np.cos(angle)*x-np.sin(angle)*y
            y_transform = np.cos(angle)*y+np.sin(angle)*x
            output_coords = np.stack((x_transform, y_transform), axis=-1)
            print(output_coords.shape, org.shape)
            output_coords = output_coords + org
            return output_coords
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(global_trajectory[0, 0, :, 0], global_trajectory[0, 0, :, 1])
        tmp = transform_coord(trajectory[0, 0, :, :2].cpu(), theta[0].cpu().numpy(), origin[0].cpu().numpy())
        print(tmp.shape, global_trajectory.shape)
        plt.plot(tmp[0, :, 0], tmp[0, :, 1])
        print(global_trajectory[0, 0])
        print(tmp)
        plt.show()
        """

        if inference: return global_trajectory, probability

        for i, scene_id in enumerate(scenario_ids):
            self.challenge_submission_data["traj"][scene_id] = global_trajectory[i]
            self.challenge_submission_data["prob"][scene_id] = probability[i]

    def generate_submission_file(self):
        print("generating submission file for argoverse 1 motion forecasting challenge")
        #np.save("traj.npy", self.challenge_submission_data["traj"])
        #np.save("prob.npy", self.challenge_submission_data["prob"])
        generate_forecasting_h5(self.challenge_submission_data["traj"],
                                self.output_path,
                                self.submission_fname,
                                self.challenge_submission_data["prob"])
        print(f"file saved to {self.output_path}/{self.submission_fname}")
