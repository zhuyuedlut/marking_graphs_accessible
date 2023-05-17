import pytorch_lightning as pl
import torch

from typing import Dict, List, Any
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from torch import Tensor

from config import CONFIG as CFG
from utils import validation_metrics


class DonutModelPLModule(pl.LightningModule):
    def __init__(self, processor: PreTrainedTokenizerBase, model: PreTrainedModel, gt_df: DataFrame, num_training_steps: int):
        super().__init__()
        self.processor = processor
        self.model = model
        self.gt_df = gt_df
        self.num_training_steps = num_training_steps

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int, dataset_idx: int = 0) -> None:
        pixel_values = batch["pixel_values"]
        batch_size = pixel_values.shape[0]
        # we feed the prompt to the model
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.model.config.decoder_start_token_id,
            device=self.device,
        )

        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=CFG.max_length,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            top_k=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        self.val_outputs.extend(
            self.processor.tokenizer.batch_decode(outputs.sequences)
        )
        self.val_ids.extend(batch["id"])

    def on_validation_start(self) -> None:
        self.val_outputs, self.val_ids = [], []

    def validation_epoch_end(self, outputs: List[Any]) -> None:

        metrics = validation_metrics(self.val_outputs, self.val_ids, self.gt_df)
        print("\n", metrics)

        self.log_dict(metrics)

        self.val_outputs, self.val_ids = [], []

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=CFG.lr)

        return optimizer