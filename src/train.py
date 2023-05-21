import json
from functools import partial
from itertools import chain
from typing import Dict, List, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import Dataset
from datasets import Image as ds_img
from datasets import concatenate_datasets
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderConfig, DonutProcessor, VisionEncoderDecoderModel

from config import CFG
from src.data import gen_data, add_image_sizes, preprocess
from src.models.dotnut import DonutModelLPModule
from src.tokens import new_tokens, PROMPT_TOKEN


def check_for_unk(examples: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Check for unknown tokens in the given examples.

    This function takes a dictionary containing a list of ground truth texts and
    tokenizes them using the processor's tokenizer. It then checks for any unknown
    tokens in the tokenized text and returns a dictionary containing a list of the
    unknown tokens for each example.

    Args:
        examples (dict): A dictionary containing a list of ground truth texts.
            Example: {"ground_truth": ["text1", "text2", ...]}

    Returns:
        dict: A dictionary containing a list of unknown tokens for each example.
            Example: {"unk_tokens": [["unk1", "unk2"], [], ["unk3"], ...]}
    """

    texts = examples["ground_truth"]

    ids = processor.tokenizer(texts).input_ids
    tokens = [processor.tokenizer.tokenize(x, add_special_tokens=True) for x in texts]

    unk_tokens = []
    for example_ids, example_tokens in zip(ids, tokens):
        example_unk_tokens = []
        for i in range(len(example_ids)):
            if example_ids[i] == processor.tokenizer.unk_token_id:
                example_unk_tokens.append(example_tokens[i])

        unk_tokens.append(example_unk_tokens)

    return {"unk_tokens": unk_tokens}


def collate_fn(samples: List[Dict[str, Union[torch.Tensor, List[int], str]]]) -> Dict[
    str, Union[torch.Tensor, List[str]]]:
    """
    Custom collate function for DataLoader.

    This function takes a list of samples and combines them into a batch with
    properly padded input_ids.

    Args:
        samples (List[Dict[str, Union[torch.Tensor, List[int], str]]]):
            A list of samples, where each sample is a dictionary containing
            "pixel_values" (torch.Tensor), "input_ids" (List[int]), and "id" (str).

    Returns:
        Dict[str, Union[torch.Tensor, List[str]]]:
            A dictionary containing the combined pixel values, padded input_ids, and ids.
    """

    batch = {}

    batch["pixel_values"] = torch.stack([x["pixel_values"] for x in samples])

    max_length = max([len(x["input_ids"]) for x in samples])

    # Make a multiple of 8 to efficiently use the tensor cores
    if max_length % 8 != 0:
        max_length = (max_length // 8 + 1) * 8

    input_ids = [
        x["input_ids"] + [pad_token_id] * (max_length - len(x["input_ids"]))
        for x in samples
    ]

    labels = torch.tensor(input_ids)
    labels[labels == pad_token_id] = -100  # ignore loss on padding tokens
    batch["labels"] = labels

    batch["id"] = [x["id"] for x in samples]

    return batch


if __name__ == "__main__":
    train_json_files = list((CFG.data_dir / "annotations").glob("*.json"))
    ds = Dataset.from_generator(
        gen_data, gen_kwargs={"files": train_json_files}, num_proc=CFG.num_proc
    )
    ds = ds.map(add_image_sizes, batched=True, num_proc=CFG.num_proc)

    config = VisionEncoderDecoderConfig.from_pretrained(CFG.pretrained_model_dir)
    config.encoder.image_size = (CFG.image_height, CFG.image_width)
    config.decoder.max_length = CFG.max_length

    processor = DonutProcessor.from_pretrained(CFG.pretrained_model_dir)
    processor.image_processor.size = {
        "height": CFG.image_height,
        "width": CFG.image_width,
    }
    unk = ds.map(check_for_unk, batched=True, num_proc=CFG.num_proc)
    unk = unk.filter(lambda x: len(x["unk_tokens"]) > 0, num_proc=CFG.num_proc)
    all_unk_tokens = [x for y in unk["unk_tokens"] for x in y]

    example_str = "0.1 1 1990"

    temp_ids = processor.tokenizer(example_str).input_ids
    print("ids:", temp_ids)
    print("tokenized:", processor.tokenizer.tokenize(example_str))
    print("decoded:", processor.tokenizer.decode(temp_ids))
    print("unk id:", processor.tokenizer.unk_token_id)

    # Adding these tokens should mean that there should be very few unknown tokens
    num_added = processor.tokenizer.add_tokens(["<one>"] + new_tokens)
    print(num_added, "tokens added")
    config.pad_token_id = processor.tokenizer.pad_token_id
    config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([PROMPT_TOKEN])[0]

    one_token_id = processor.tokenizer("<one>", add_special_tokens=False).input_ids[0]
    unk_token_id = processor.tokenizer.unk_token_id

    image_ds = ds.cast_column("image_path", ds_img())
    image_ds.set_transform(partial(preprocess, processor=processor, CFG=CFG))

    extracted_ds = ds.filter(lambda x: x["source"] == "extracted", num_proc=CFG.num_proc)
    generated_ds = ds.filter(lambda x: x["source"] == "generated", num_proc=CFG.num_proc)
    chart_types = extracted_ds["chart-type"]

    skf = StratifiedKFold(n_splits=4)
    fold_idxs = []
    for _, val_idxs in skf.split(chart_types, y=chart_types):
        fold_idxs.append(val_idxs)

    fold = 0
    train_extracted = extracted_ds.select(
        list(chain(*[x for i, x in enumerate(fold_idxs) if i != fold]))
    )
    train_ds = concatenate_datasets([train_extracted, generated_ds])
    train_ds = train_ds.cast_column("image_path", ds_img())
    train_ds.set_transform(partial(preprocess, processor=processor, CFG=CFG))

    val_gt_ds = extracted_ds.select(fold_idxs[fold])
    val_ds = val_gt_ds.cast_column("image_path", ds_img())
    val_ds.set_transform(partial(preprocess, processor=processor, CFG=CFG))
    gt_chart_type = val_gt_ds["chart-type"]
    gt_x = [json.loads(_) for _ in val_gt_ds["x"]]
    gt_y = [json.loads(_) for _ in val_gt_ds["y"]]
    gt_ids = val_gt_ds["id"]

    pad_token_id = processor.tokenizer.pad_token_id

    train_dataloader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=CFG.num_workers,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=CFG.num_workers,
    )

    num_training_steps = len(train_dataloader) * CFG.epochs // CFG.gpus

    batch = next(iter(train_dataloader))

    batch.keys(), [(k, v.shape) for k, v in batch.items() if k != "id"]

    gt_chart_type = val_gt_ds["chart-type"]
    gt_x = [json.loads(_) for _ in val_gt_ds["x"]]
    gt_y = [json.loads(_) for _ in val_gt_ds["y"]]
    gt_ids = val_gt_ds["id"]

    index = [f"{id_}_x" for id_ in gt_ids] + [f"{id_}_y" for id_ in gt_ids]
    gt_df = pd.DataFrame(
        index=index,
        data={
            "data_series": gt_x + gt_y,
            "chart_type": gt_chart_type * 2,
        },
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        "/home/zhuyuedlut/Pretrained_Model/donut-base", config=config, ignore_mismatched_sizes=True
    )
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model_module = DonutModelLPModule(processor, model, gt_df, num_training_steps)

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

    trainer.fit(model_module, train_dataloaders=train_dataloader)
