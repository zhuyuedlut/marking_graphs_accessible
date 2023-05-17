import pytorch_lightning as pl

from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from config import CONFIG


data_dir = Path("./data/train")
images_path = data_dir / "images"
train_json_files = list((data_dir / "annotations").glob("*.json"))




checkpoint_callback = ModelCheckpoint(CONFIG.output_path)
loggers = []

trainer = pl.Trainer(
    accelerator='gpu',
    devices=CONFIG.gpus,
    max_epochs=CONFIG.epochs,
    val_check_interval=CONFIG.val_check_interval,
    check_val_every_n_epoch=CONFIG.check_val_every_n_epoch,
    gradient_clip_val=CONFIG.gradient_clip_val,
    precision=16,  # if you have tensor cores (t4, v100, a100, etc.) training will be 2x faster
    num_sanity_val_steps=5,
    callbacks=[checkpoint_callback],
    logger=loggers
)
