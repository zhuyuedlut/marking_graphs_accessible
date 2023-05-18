import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from config import CFG

if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(CFG.output_path)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=CFG.epochs,
        val_check_interval=CFG.val_check_interval,
        check_val_every_n_epoch=CFG.check_val_every_n_epoch,
        gradient_clip_val=CFG.gradient_clip_val,
        precision=16,  # if you have tensor cores (t4, v100, a100, etc.) training will be 2x faster
        num_sanity_val_steps=5,
        callbacks=[checkpoint_callback],
        logger=[]
    )

    trainer.fit()
