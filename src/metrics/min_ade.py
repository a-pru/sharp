from typing import Any, Callable, Dict, Optional

import torch
from torchmetrics import Metric

from .utils import sort_predictions


class minADE(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        k=6,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(minADE, self).__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.k = k
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> None:
        with torch.no_grad():
            pred, _ = sort_predictions(outputs['y_hat'], outputs['pi'], k=self.k)
            ade = torch.norm(
                pred[..., :2] - target.unsqueeze(1)[..., :2], p=2, dim=-1
            ).mean(-1)
            min_ade = ade.min(-1)[0]
            self.sum += min_ade.sum()
            self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
