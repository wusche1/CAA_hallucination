import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re

token = os.getenv("TOKEN_NAME")


class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.last_hidden_state = None
        self.add_activations = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.last_hidden_state = output[0]
        if self.add_activations is not None:
            output = (output[0] + self.add_activations,) + output[1:]
        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.last_hidden_state = None
        self.add_activations = None


class LlamaHelperBase:
    def __init__(self, pretrained_model):
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model)

        success = False
        for device_id in range(8):  # For cuda:0 to cuda:7
            try:
                self.device = f"cuda:{device_id}"
                self.model = self.model.to(self.device)
                success = True
                break
            except RuntimeError:
                pass

        if not success:
            raise RuntimeError("Failed to move the model to any of the GPUs.")

        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer)

    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(
            inputs.input_ids.to(self.device), max_length=max_length
        )
        return self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def generate_text_batched(self, prompts, max_length=100, max_tokens_per_batch=4096):
        # Convert each prompt in the list to tokens
        tokenized_prompts = [
            self.tokenizer(prompt, return_tensors="pt").input_ids for prompt in prompts
        ]

        # Calculate the total tokens for all prompts
        total_tokens = sum([tp.size(1) for tp in tokenized_prompts])

        # If the total tokens exceed our max tokens per batch, we split the prompts into smaller batches
        if total_tokens > max_tokens_per_batch:
            mid_idx = len(prompts) // 2
            first_half = self.generate_text_batched(
                prompts[:mid_idx], max_length, max_tokens_per_batch
            )
            second_half = self.generate_text_batched(
                prompts[mid_idx:], max_length, max_tokens_per_batch
            )
            return first_half + second_half

        # If we are below the token limit, continue processing as normal

        # Calculate the maximum length of the tokenized prompts to pad others accordingly
        max_token_length = max([tp.size(1) for tp in tokenized_prompts])

        # Pad each tokenized prompt to the max length
        tokens = torch.cat(
            [
                torch.nn.functional.pad(tp, (0, max_token_length - tp.size(1))).to(
                    self.device
                )
                for tp in tokenized_prompts
            ]
        )

        # Generate responses for each prompt in the batch
        generated = self.model.generate(
            input_ids=tokens,
            max_length=max_token_length + max_length,
        )

        # Decode each response in the batch and return as a list
        return clean_strings(self.tokenizer.batch_decode(generated))

    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens.to(self.device)).logits
            return logits

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].last_hidden_state

    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add(activations.to(self.device))

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def get_activations(self, dataset, layers, position=-2):
        diffs = dict([(layer, []) for layer in layers])

        for s_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
            s_out = self.get_logits(s_tokens.unsqueeze(0))
            for layer in layers:
                s_activations = self.get_last_activations(layer)
                s_activations = s_activations[0, position, :].detach().cpu()
                diffs[layer].append(s_activations)
            n_out = self.get_logits(n_tokens.unsqueeze(0))
            for layer in layers:
                n_activations = self.get_last_activations(layer)
                n_activations = n_activations[0, position, :].detach().cpu()
                diffs[layer][-1] -= n_activations

        for layer in layers:
            diffs[layer] = torch.stack(diffs[layer])
        return diffs


class Llama27BHelper:
    def __init__(self, pretrained_model="meta-llama/Llama-2-7b-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model, use_auth_token=token, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model, use_auth_token=token
        )
        success = False
        for device_id in range(8):  # For cuda:0 to cuda:7
            try:
                self.device = f"cuda:{device_id}"
                self.model = self.model.to(self.device)
                success = True
                break
            except RuntimeError:
                pass

        if not success:
            raise RuntimeError("Failed to move the model to any of the GPUs.")
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer)

    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(
            inputs.input_ids.to(self.device), max_length=max_length
        )
        return self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def generate_text_batched(self, prompts, max_length=100, max_tokens_per_batch=4096):
        # Convert each prompt in the list to tokens
        tokenized_prompts = [
            self.tokenizer(prompt, return_tensors="pt").input_ids for prompt in prompts
        ]

        # Calculate the total tokens for all prompts
        total_tokens = sum([tp.size(1) for tp in tokenized_prompts])

        # If the total tokens exceed our max tokens per batch, we split the prompts into smaller batches
        if total_tokens > max_tokens_per_batch:
            mid_idx = len(prompts) // 2
            first_half = self.generate_text_batched(
                prompts[:mid_idx], max_length, max_tokens_per_batch
            )
            second_half = self.generate_text_batched(
                prompts[mid_idx:], max_length, max_tokens_per_batch
            )
            return first_half + second_half

        # If we are below the token limit, continue processing as normal

        # Calculate the maximum length of the tokenized prompts to pad others accordingly
        max_token_length = max([tp.size(1) for tp in tokenized_prompts])

        # Pad each tokenized prompt to the max length
        tokens = torch.cat(
            [
                torch.nn.functional.pad(tp, (0, max_token_length - tp.size(1))).to(
                    self.device
                )
                for tp in tokenized_prompts
            ]
        )

        # Generate responses for each prompt in the batch
        generated = self.model.generate(
            input_ids=tokens,
            max_length=max_token_length + max_length,
        )

        # Decode each response in the batch and return as a list
        return clean_strings(self.tokenizer.batch_decode(generated))

    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens.to(self.device)).logits
            return logits

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].last_hidden_state

    def set_add_activations(self, layer, activations):
        self.model.model.layers[layer].add(activations.to(self.device))

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def get_activations(self, dataset, layers, position=-2):
        diffs = dict([(layer, []) for layer in layers])

        for s_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
            s_out = self.get_logits(s_tokens.unsqueeze(0))
            for layer in layers:
                s_activations = self.get_last_activations(layer)
                s_activations = s_activations[0, position, :].detach().cpu()
                diffs[layer].append(s_activations)
            n_out = self.get_logits(n_tokens.unsqueeze(0))
            for layer in layers:
                n_activations = self.get_last_activations(layer)
                n_activations = n_activations[0, position, :].detach().cpu()
                diffs[layer][-1] -= n_activations

        for layer in layers:
            diffs[layer] = torch.stack(diffs[layer])
        return diffs


class Llama213BHelper(LlamaHelperBase):
    def __init__(self):
        super().__init__("meta-llama/llama213B-hf")


class Llama230BHelper(LlamaHelperBase):
    def __init__(self):
        super().__init__("meta-llama/llama230b-hf")


class Llama270BHelper(LlamaHelperBase):
    def __init__(self):
        super().__init__("meta-llama/llama270B-hf")


class ComparisonDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        s_text = item["chosen"]
        n_text = item["rejected"]

        s_tokens = self.tokenizer.encode(s_text, return_tensors="pt")[0]
        n_tokens = self.tokenizer.encode(n_text, return_tensors="pt")[0]

        return s_tokens, n_tokens


def clean_strings(strings):
    def clean_string(input_str):
        # Remove </s> and <unk> tokens
        cleaned = re.sub(r"</s>|<unk>", "", input_str)
        # Strip any leading or trailing whitespace
        return cleaned.strip()

    return [clean_string(s) for s in strings]
