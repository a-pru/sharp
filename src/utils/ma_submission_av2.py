import time
from pathlib import Path

import torch
from torch import Tensor


class SubmissionAv2:
    def __init__(self, save_dir: str = '') -> None:
        stamp = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
        self.submission_file = Path(save_dir) / f'multi_agent_{stamp}.parquet'
        from .ma_submission_protocol import ChallengeSubmission
        self.challenge_submission = ChallengeSubmission(predictions={})

        self.probs = {}

    def format_data(
        self,
        data: dict,
        trajectory: Tensor,
        probability: Tensor,
        normalized_probability=False,
        inference=False,
    ) -> None:
        '''
        trajectory: (B, K, N, 60, 2)
        probability: (B, M)
        normalized_probability: if the input probability is normalized,
        '''

        scenario_ids = data['scenario_id']
        track_ids = data['track_id']
        B, N = data['x_key_valid_mask'].shape

        origin = data['origins'].view(B*N, 1, 1, 2).double()[data['x_key_valid_mask'].view(-1)]
        theta = data['thetas'].double().view(B*N)[data['x_key_valid_mask'].view(-1)]

        trajectory = trajectory.permute(0, 2, 1, 3, 4)[data['x_key_valid_mask']]

        trajectory = trajectory[:, :, :60]

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
                torch.matmul(trajectory[..., :2].double(), rotate_mat.unsqueeze(1))
                + origin
            )
            if not normalized_probability:
                probability = torch.softmax(probability.double(), dim=-1)
  
        global_trajectory = global_trajectory.detach().cpu().numpy()
        probability = probability.detach().cpu().numpy()

        if inference:
            return global_trajectory, probability
        
        out_idx = 0
        for i, scene_id in enumerate(scenario_ids):
            self.challenge_submission.predictions[scene_id] = [probability[i], {}]
            for track_id in track_ids[i]:
                self.challenge_submission.predictions[scene_id][1][track_id] = global_trajectory[out_idx]
                out_idx += 1
            #print(self.challenge_submission.predictions[scene_id])

    def generate_submission_file(self):
        print('generating submission file for argoverse 2 motion forecasting challenge')
        self.challenge_submission.to_parquet(self.submission_file)
        print(f'file saved to {self.submission_file}')
        #torch.save(self.probs, "tmp.pt")
