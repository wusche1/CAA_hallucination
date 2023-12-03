import sys

sys.path.append("../..")
from lib import automated_evaluation, chat_helper
from lib.automated_evaluation import caesar_decrypt
from lib.hallucination_lib import get_letter_probabilities_batched
import pickle
import pandas as pd
from tqdm import tqdm
import torch

token = "redacted_for_publishing"
import os
import re
import json
import numpy as np

system_prompt = (
    "You are a helpful, honest and concise assistant."  # Needed for chat model
)
model = chat_helper.Llama7BChatHelper(
    token, system_prompt, master_device=1, threshold=0.2
)
model.tokenizer.pad_token = model.tokenizer.eos_token


def average_random_lines(data, N=100):
    """
    This function selects 100 random rows from the input dataset and returns their average.

    Parameters:
    - data (torch.Tensor): The input dataset.

    Returns:
    - torch.Tensor: A tensor containing the average of the randomly selected 100 rows.
    """

    # Randomly select 100 indices
    indices = torch.randperm(data.size(0))[:N]

    # Select the rows corresponding to these indices
    selected_data = data[indices]

    # Return the average of the selected rows
    return torch.mean(selected_data, dim=0)


question_path = "../steering_vectors/"
question_types = [
    "direct_questions",
    "questioning_assuming_statement",
    "conversation",
    "alluding_questions",
]
layer_list = [14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30]
coeff_list = [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]


gridsearch_data_path = "./gridsearch_data/"
batch_size = 10
if not os.path.exists(gridsearch_data_path):
    os.makedirs(gridsearch_data_path)
total_updates = len(question_types) * len(layer_list) * len(coeff_list) * 4
pbar = tqdm(total=total_updates, desc="Processing")

for question_type in question_types:
    output_filepath = f"{gridsearch_data_path}{question_type}.csv"
    if os.path.exists(output_filepath):
        gridesearch_data = pd.read_csv(output_filepath)
    else:
        question_data = pd.read_csv(f"{question_path}{question_type}/_test.csv")
        gridesearch_data = question_data.iloc[::20, :].reset_index(drop=True)
    for layer in layer_list:
        steering_data_fiction = torch.load(
            f"{question_path}{question_type}/fiction/all_diffs_layer_{layer}.pt"
        )
        steering_vector_fiction = average_random_lines(steering_data_fiction)
        steering_data_truth = torch.load(
            f"{question_path}{question_type}/truth/all_diffs_layer_{layer}.pt"
        )
        steering_vector_truth = average_random_lines(steering_data_truth)
        steering_data_mixed = torch.load(
            f"{question_path}{question_type}/mixed/all_diffs_layer_{layer}.pt"
        )
        steering_vector_mixed = average_random_lines(steering_data_mixed)

        steering_vector_combined = (steering_vector_fiction + steering_vector_truth) / 2
        vector_list = [
            steering_vector_truth,
            steering_vector_fiction,
            steering_vector_combined,
            steering_vector_mixed,
        ]
        for coeff in coeff_list:
            name = f"layer_{layer}_coeff_{coeff}"
            name_list = [
                name + "_truth",
                name + "_fiction",
                name + "_combined",
                name + "_mixed",
            ]
            for name, vector in zip(name_list, vector_list):
                if name + "_truth_chosen" in gridesearch_data.columns:
                    pbar.update(1)
                    continue
                model.reset_all()
                model.set_add_activations(
                    layer, vector / np.linalg.norm(vector) * coeff
                )
                fiction_question = gridesearch_data["fiction_question"].tolist()
                truth_question = gridesearch_data["truth_question"].tolist()
                fiction_accepted = gridesearch_data["fiction_accepetance"].tolist()
                truth_accepted = gridesearch_data["truth_accepetance"].tolist()
                fiction_rejected = gridesearch_data["fiction_rejection"].tolist()
                truth_rejected = gridesearch_data["truth_rejection"].tolist()
                p_chosen_truth, p_rejected_truth = get_letter_probabilities_batched(
                    truth_question,
                    truth_accepted,
                    truth_rejected,
                    model,
                    batch_size=batch_size,
                )
                p_chosen_fiction, p_rejected_fiction = get_letter_probabilities_batched(
                    fiction_question,
                    fiction_accepted,
                    fiction_rejected,
                    model,
                    batch_size=batch_size,
                )
                new_columns = {
                    name + "_truth_chosen": [round(p, 3) for p in p_chosen_truth],
                    name + "_truth_rejected": [round(p, 3) for p in p_rejected_truth],
                    name + "_fiction_chosen": [round(p, 3) for p in p_chosen_fiction],
                    name
                    + "_fiction_rejected": [round(p, 3) for p in p_rejected_fiction],
                }
                gridesearch_data = pd.read_csv(output_filepath)
                df_new_columns = pd.DataFrame(new_columns)
                gridesearch_data = pd.concat([gridesearch_data, df_new_columns], axis=1)

                pbar.update(1)
                gridesearch_data.to_csv(output_filepath, index=False)
