import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer  # If you're using HuggingFace's transformers library for the tokenizer.
system_prompt= "You are a helpful, honest and concise assistant."

def extract_data(test_data):
    valid_questions = []
    valid_chosen = []
    valid_rejected = []

    incorrect_questions = []
    incorrect_chosen = []
    incorrect_rejected = []
    incorrect_premises = []

    for d in test_data:
        try:
            q_valid = d["Valid Question"]["Question"]
            c_valid = d["Valid Question"]["True Answer"]
            r_valid = d["Valid Question"]["Answer Refuting the Premise Incorrectly"]
            

            # Extract data for incorrect questions
            q_incorrect = d["Incorrect Question"]["Question"]
            c_incorrect = d["Incorrect Question"]["Answer Pointing Out Inaccuracy"]
            r_incorrect = d["Incorrect Question"]["Answer Fabricating Information"]
            p_incorrect = d["Incorrect Question"]["Incorrect Premise"]

            valid_questions.append(q_valid)
            valid_chosen.append(c_valid)
            valid_rejected.append(r_valid)

            
            incorrect_questions.append(q_incorrect)
            incorrect_chosen.append(c_incorrect)
            incorrect_rejected.append(r_incorrect)
            incorrect_premises.append(p_incorrect)
        except KeyError:
            pass

    return {
        'valid_questions': valid_questions,
        'valid_chosen': valid_chosen,
        'valid_rejected': valid_rejected,
        'incorrect_questions': incorrect_questions,
        'incorrect_chosen': incorrect_chosen,
        'incorrect_rejected': incorrect_rejected,
        'incorrect_premises': incorrect_premises
    }

def tokenize_batch_ab(self, question_list, system_prompt=system_prompt):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    answer_list=["I choose (" for _ in question_list]
    dialog_content_list = [
        B_INST + B_SYS + system_prompt + E_SYS + question.strip() + E_INST +" "+ answer.strip()
        for question, answer in zip(question_list, answer_list)
    ]
    dialog_tokens_list=[]
    E_INST_tokens=torch.tensor(self.tokenizer.encode(E_INST)[2:])
    for dialog in dialog_content_list:
        dialog_tokens=torch.tensor(self.tokenizer.encode(dialog))
        dialog_tokens_list.append(dialog_tokens)
    
    max_len = max([len(q) for q in dialog_tokens_list])
    filler = self.tokenizer.encode(" ")[1]  # Ensure that filler is a single integer value
    end = torch.tensor(self.tokenizer.encode("I choose (")[1:])
    for idx, dialog in enumerate(dialog_tokens_list):
        padding_length = max_len - len(dialog)
        dialog_tokens_list[idx] = torch.cat([dialog, torch.tensor([filler] * padding_length, dtype=torch.int64), end], dim=0)

    tokens = torch.stack(dialog_tokens_list)
    return tokens
def find_token_ids_for_letter(tokenizer, letter: str):
    letter = letter.lower()  # Convert the letter to lowercase
    matching_ids = []

    # Iterate over tokens and their ids
    for token, token_id in tokenizer.get_vocab().items():
        if token.lower().replace(" ", "") == letter:
            matching_ids.append(token_id)

    return matching_ids

def get_letter_probabilities_batched(question_list, chosen_answer_list, rejected_answer_list, model, batch_size=10):
    assert len(question_list) == len(chosen_answer_list) == len(rejected_answer_list)
    
    correct_answers = [random.choice(["A", "B"]) for _ in range(len(question_list))]
    A_tokens = find_token_ids_for_letter(model.tokenizer, "A")
    B_tokens = find_token_ids_for_letter(model.tokenizer, "B")
    
    prompt_list = []
    for question, chosen_answer, rejected_answer, correct_answer in zip(question_list, chosen_answer_list, rejected_answer_list, correct_answers):
        prompt = f"{question}\n\nChoices:\n(A){chosen_answer if correct_answer=='A' else rejected_answer}\n(B){rejected_answer if correct_answer=='A' else chosen_answer}\nChoose an answer and explain your reasoning."
        prompt_list.append(prompt)
        
    all_tokens = tokenize_batch_ab(model, prompt_list)
    all_logits = []

    # Split and process in smaller batches
    for i in range(0, len(all_tokens), batch_size):
        batch_tokens = all_tokens[i:i+batch_size]
        logits = model.get_logits(batch_tokens.to(model.device))[:, -1, :]
        all_logits.extend(logits)

    probabilities = F.softmax(torch.stack(all_logits), dim=-1).to("cpu")
    p_chosen, p_rejected = [], []

    for prob, correct_answer in zip(probabilities, correct_answers):
        if correct_answer == "A":
            p_chosen.append(prob[A_tokens].sum().item())
            p_rejected.append(prob[B_tokens].sum().item())
        else:
            p_chosen.append(prob[B_tokens].sum().item())
            p_rejected.append(prob[A_tokens].sum().item())

    return p_chosen, p_rejected