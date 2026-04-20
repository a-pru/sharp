import os
import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate


DIR = "./conf/" 

@hydra.main(version_base=None, config_path=DIR, config_name="config")
def main(cfg):
    pl.seed_everything(cfg.seed)
    assert os.path.exists(cfg.checkpoint), f"Checkpoint {cfg.checkpoint} does not exist"

    datamodule = instantiate(cfg.datamodule.pl_module, test=cfg.submit)
    model = instantiate(cfg.model.pl_module)

    trainer = pl.Trainer(
        logger=False,
        accelerator="gpu",
        devices=cfg.gpus if not cfg.submit else 1,
        max_epochs=1,
    )

    if not cfg.submit:
        trainer.validate(model, datamodule, ckpt_path=cfg.checkpoint)
    else:
        trainer.test(model, datamodule, ckpt_path=cfg.checkpoint)


if __name__ == "__main__":
    main()
