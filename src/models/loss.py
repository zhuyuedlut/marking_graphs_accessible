from typing import List, Dict

import pandas as pd


def validation_metrics(val_outputs: List[str], val_ids: List[str], gt_df: pd.DataFrame) -> Dict[str, float]:
    pred_triplets = []

    for example_output in val_outputs:
        pass
