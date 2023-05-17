import re
import pandas as pd
import numpy as np
import json
import os


from typing import List, Dict, Union, Tuple
from polyleven import levenshtein

PROMPT_TOKEN = "<|PROMPT|>"
X_START = "<x_start>"
X_END = "<x_end>"
Y_START = "<y_start>"
Y_END = "<y_end>"

SEPARATOR_TOKENS = [
    PROMPT_TOKEN,
    X_START,
    X_END,
    Y_START,
    Y_END,
]

LINE_TOKEN =  "<line>"
VERTICAL_BAR_TOKEN = "<vertical_bar>"
HORIZONTAL_BAR_TOKEN = "<horizontal_bar>"
SCATTER_TOKEN = "<scatter>"
DOT_TOKEN = "<dot>"

CHART_TYPE_TOKENS = [
    LINE_TOKEN,
    VERTICAL_BAR_TOKEN,
    HORIZONTAL_BAR_TOKEN,
    SCATTER_TOKEN,
    DOT_TOKEN,
]

new_tokens = SEPARATOR_TOKENS + CHART_TYPE_TOKENS


def add_image_sizes(examples: Dict[str, Union[str, os.PathLike]]) -> Dict[str, List[int]]:
    """
    This function takes a dictionary of examples and adds the width and height of the
    image to the dictionary. This is to be used with the `Dataset.map` function.

    Args:
        examples (dict): A dictionary of examples (from `map` function)

    Returns:
        dict: The dictionary with the width and height of the image added
    """

    sizes = [Image.open(x).size for x in examples["image_path"]]

    width, height = list(zip(*sizes))

    return {
        "width": list(width),
        "height": list(height),
    }

def gen_data(files: List[Union[str, os.PathLike]]) -> Dict[str, str]:
    """
    This function takes a list of json files and returns a generator that yields a
    dictionary with the ground truth string and the path to the image.

    Args:
        files (list): A list of json files

    Returns:
        generator: A generator that yields a dictionary with the ground truth string and
            the path to the corresponding image.
    """

    for f in files:

        yield {
            **get_gt_string_and_xy(f),
            "image_path": str(images_path / f"{f.stem}.jpg"),
        }


def round_float(value: Union[int, float, str]) -> Union[str, float]:
    """
    Convert a float value to a string with the specified number of decimal places.
    If there is more than 1 digit in the integer, then we will truncate to 1 decimal.
    Otherwise, will truncate to 4 decimals.

    Args:
        value (int, float, str): The float value to convert

    Returns:
        str: The rounded float value as a string
    """
    if isinstance(value, float):
        value = str(value)

        if "." in value:
            integer, decimal = value.split(".")
            if abs(float(integer)) > 1:
                decimal = decimal[:1]
            else:
                decimal = decimal[:4]

            value = integer + "." + decimal
    return value


def is_nan(value: Union[int, float, str]) -> bool:
    """
    Check if a value is NaN (not a number).

    Args:
        value (int, float, str): The value to check

    Returns:
        bool: True if the value is NaN, False otherwise
    """
    return isinstance(value, float) and str(value) == "nan"


def get_gt_string_and_xy(filepath: Union[str, os.PathLike]) -> Dict[str, str]:
    """
    Get the ground truth string and x-y data from the given JSON file.

    Args:
        filepath (str): The path to the JSON file

    Returns:
        dict: A dictionary containing the ground truth string, x-y data, chart type, id, and source
    """
    filepath = Path(filepath)

    with open(filepath) as fp:
        data = json.load(fp)

    data_series = data["data-series"]

    all_x, all_y = [], []

    for d in data_series:
        x = d["x"]
        y = d["y"]

        x = round_float(x)
        y = round_float(y)

        # Ignore nan values
        if is_nan(x) or is_nan(y):
            continue

        all_x.append(x)
        all_y.append(y)

    chart_type = f"<{data['chart-type']}>"
    x_str = X_START + ";".join(list(map(str, all_x))) + X_END
    y_str = Y_START + ";".join(list(map(str, all_y))) + Y_END

    gt_string = PROMPT_TOKEN + chart_type + x_str + y_str

    return {
        "ground_truth": gt_string,
        "x": json.dumps(all_x),
        "y": json.dumps(all_y),
        "chart-type": data["chart-type"],
        "id": filepath.stem,
        "source": data["source"],
    }

def rmse(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between the true and predicted values.

    Args:
        y_true (List[float]): The true values.
        y_pred (List[float]): The predicted values.

    Returns:
        float: The Root Mean Square Error.
    """
    return np.sqrt(np.mean(np.square(np.subtract(y_true, y_pred))))


def sigmoid(x: float) -> float:
    """
    Calculate the sigmoid function for the given value.

    Args:
        x (float): The input value.

    Returns:
        float: The result of the sigmoid function.
    """
    return 2 - 2 / (1 + np.exp(-x))


def normalized_rmse(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate the normalized Root Mean Square Error (RMSE) between the true and predicted values.

    Args:
        y_true (List[float]): The true values.
        y_pred (List[float]): The predicted values.

    Returns:
        float: The normalized Root Mean Square Error.
    """
    numerator = rmse(y_true, y_pred)
    denominator = rmse(y_true, np.mean(y_true))

    # https://www.kaggle.com/competitions/benetech-making-graphs-accessible/discussion/396947
    if denominator == 0:
        if numerator == 0:
            return 1.0
        return 0.0

    return sigmoid(numerator / denominator)

def normalized_levenshtein_score(y_true: List[str], y_pred: List[str]) -> float:
    """
    Calculate the normalized Levenshtein distance between two lists of strings.

    Args:
        y_true (List[str]): The true values.
        y_pred (List[str]): The predicted values.

    Returns:
        float: The normalized Levenshtein distance.
    """
    total_distance = np.sum([levenshtein(yt, yp) for yt, yp in zip(y_true, y_pred)])
    length_sum = np.sum([len(yt) for yt in y_true])
    return sigmoid(total_distance / length_sum)

def score_series(
    y_true: List[Union[float, str]], y_pred: List[Union[float, str]]
) -> float:
    """
    Calculate the score for a series of true and predicted values.

    Args:
        y_true (List[Union[float, str]]): The true values.
        y_pred (List[Union[float, str]]): The predicted values.

    Returns:
        float: The score for the series.
    """
    if len(y_true) != len(y_pred):
        return 0.0
    if isinstance(y_true[0], str):
        return normalized_levenshtein_score(y_true, y_pred)
    else:
        # Since this is a generative model, there is a chance it doesn't produce a float.
        # In that case, we return 0.0.
        try:
            return normalized_rmse(y_true, list(map(float, y_pred)))
        except:
            return 0.0

def benetech_score(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """Evaluate predictions using the metric from the Benetech - Making Graphs Accessible.

    Parameters
    ----------
    ground_truth: pd.DataFrame
        Has columns `[data_series, chart_type]` and an index `id`. Values in `data_series`
        should be either arrays of floats or arrays of strings.

    predictions: pd.DataFrame
    """
    if not ground_truth.index.equals(predictions.index):
        raise ValueError(
            "Must have exactly one prediction for each ground-truth instance."
        )
    if not ground_truth.columns.equals(predictions.columns):
        raise ValueError(f"Predictions must have columns: {ground_truth.columns}.")
    pairs = zip(
        ground_truth.itertuples(index=False), predictions.itertuples(index=False)
    )
    scores = []
    for (gt_series, gt_type), (pred_series, pred_type) in pairs:
        if gt_type != pred_type:  # Check chart_type condition
            scores.append(0.0)
        else:  # Score with RMSE or Levenshtein as appropriate
            scores.append(score_series(gt_series, pred_series))

    ground_truth["score"] = scores

    grouped = ground_truth.groupby("chart_type", as_index=False)["score"].mean()

    chart_type2score = {
        chart_type: score
        for chart_type, score in zip(grouped["chart_type"], grouped["score"])
    }

    return np.mean(scores), chart_type2score

def string2triplet(pred_string: str) -> Tuple[str, List[str], List[str]]:
    """
    Convert a prediction string to a triplet of chart type, x values, and y values.

    Args:
        pred_string (str): The prediction string.

    Returns:
        Tuple[str, List[str], List[str]]: A triplet of chart type, x values, and y values.
    """

    chart_type = "line"
    for tok in CHART_TYPE_TOKENS:
        if tok in pred_string:
            chart_type = tok.strip("<>")

    pred_string = re.sub(r"<one>", "1", pred_string)

    x = pred_string.split(X_START)[1].split(X_END)[0].split(";")
    y = pred_string.split(Y_START)[1].split(Y_END)[0].split(";")

    if len(x) == 0 or len(y) == 0:
        return chart_type, [], []

    min_length = min(len(x), len(y))

    x = x[:min_length]
    y = y[:min_length]

    return chart_type, x, y

def validation_metrics(val_outputs: List[str], val_ids: List[str], gt_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate validation metrics for a set of outputs, ids, and ground truth dataframe.

    Args:
        val_outputs (List[str]): A list of validation outputs.
        val_ids (List[str]): A list of validation ids.
        gt_df (pd.DataFrame): The ground truth dataframe.

    Returns:
        Dict[str, float]: A dictionary containing the validation scores.
    """
    pred_triplets = []

    for example_output in val_outputs:

        if not all([x in example_output for x in [X_START, X_END, Y_START, Y_END]]):
            pred_triplets.append(("line", [], []))
        else:
            pred_triplets.append(string2triplet(example_output))

    pred_df = pd.DataFrame(
        index=[f"{id_}_x" for id_ in val_ids] + [f"{id_}_y" for id_ in val_ids],
        data={
            "data_series": [x[1] for x in pred_triplets]
            + [x[2] for x in pred_triplets],
            "chart_type": [x[0] for x in pred_triplets] * 2,
        },
    )

    overall_score, chart_type2score = benetech_score(
        gt_df.loc[pred_df.index.values], pred_df
    )

    return {
        "val_score": overall_score,
        **{f"{k}_score": v for k, v in chart_type2score.items()},
    }

if __name__ == '__main__':
    from pathlib import Path

    data_dir = Path("./data/train")
    images_path = data_dir / "images"
    train_json_files = list((data_dir / "annotations").glob("*.json"))

    print(get_gt_string_and_xy(data_dir / "annotations" / "000d269c8e26.json"))