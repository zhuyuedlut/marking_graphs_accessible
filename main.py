import pytorch_lightning as pl
import torch

from typing import Dict, List, Any, Union
from functools import partial

from datasets import Dataset
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import VisionEncoderDecoderConfig, DonutProcessor

from config import CONFIG
from utils import gen_data, add_image_sizes, new_tokens, PROMPT_TOKEN

def check_for_unk(examples: Dict[str, str]) -> dict[str, list[list[Any]]]:
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


def replace_unk_tokens_with_one(example_ids: List[int], example_tokens: List[str], one_token_id: int,
                                unk_token_id: int) -> List[int]:
    """
    Replace unknown tokens that represent "1" with the correct token id.

    Args:
        example_ids (list): List of token ids for a given example
        example_tokens (list): List of tokens for the same given example
        one_token_id (int): Token id for the "<one>" token
        unk_token_id (int): Token id for the unknown token

    Returns:
        list: The updated list of token ids with the correct token id for "1"
    """

    temp_ids = []
    for id_, token in zip(example_ids, example_tokens):
        if id_ == unk_token_id and token == "1":
            id_ = one_token_id
        temp_ids.append(id_)
    return temp_ids


def preprocess(examples: Dict[str, str], processor: DonutProcessor, CFG: CONFIG) -> Dict[
    str, Union[torch.Tensor, List[int], List[str]]]:
    """
    Preprocess the given examples.

    This function processes the input examples by tokenizing the texts, replacing
    any unknown tokens that represent "1" with the correct token id, and loading
    the images.

    Args:
        examples (dict): A dictionary containing ground truth texts, image paths, and ids
        processor: An object responsible for tokenizing texts and processing images
        CFG: A configuration object containing settings and hyperparameters

    Returns:
        dict: A dictionary containing preprocessed images, token ids, and ids
    """

    pixel_values = []

    texts = examples["ground_truth"]

    ids = processor.tokenizer(
        texts,
        add_special_tokens=False,
        max_length=CFG.max_length,
        padding=True,
        truncation=True,
    ).input_ids

    if isinstance(texts, str):
        texts = [texts]

    tokens = [processor.tokenizer.tokenize(text, add_special_tokens=False) for text in texts]

    one_token_id = processor.tokenizer("<one>", add_special_tokens=False).input_ids[0]
    unk_token_id = processor.tokenizer.unk_token_id

    final_ids = [
        replace_unk_tokens_with_one(example_ids, example_tokens, one_token_id, unk_token_id)
        for example_ids, example_tokens in zip(ids, tokens)
    ]

    for sample in examples["image_path"]:
        pixel_values.append(processor(sample, random_padding=True).pixel_values)

    return {
        "pixel_values": torch.tensor(np.vstack(pixel_values)),
        "input_ids": final_ids,
        "id": examples["id"],
    }


data_dir = Path("./data/train")
images_path = data_dir / "images"
train_json_files = list((data_dir / "annotations").glob("*.json"))

ds = Dataset.from_generator(
    gen_data,
    gen_kwargs={"files": train_json_files}, num_proc=CONFIG.num_proc
)
ds = ds.map(add_image_sizes, batched=True, num_proc=CONFIG.num_proc)


config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
config.encoder.image_size = (CONFIG.image_height, CONFIG.image_width)
config.decoder.max_length = CONFIG.max_length
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
processor.image_processor.size = {
    "height": CONFIG.image_height,
    "width": CONFIG.image_width,
}
unk = ds.map(check_for_unk, batched=True, num_proc=CONFIG.num_proc)
unk = unk.filter(lambda x: len(x["unk_tokens"]) > 0, num_proc=CONFIG.num_proc)
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

config.pad_token_id = processor.tokenizer.pad_token_id
config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([PROMPT_TOKEN])[0]

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
