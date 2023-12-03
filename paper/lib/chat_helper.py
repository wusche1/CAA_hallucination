import requests
import sys
import json
from random import sample
import torch
import re
import torch.nn.functional as F
import subprocess
from torch.nn import DataParallel

from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from math import log
from matplotlib.ticker import ScalarFormatter

token = "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"

system_prompt = (
    "You are a helpful, honest and concise assistant."  # Needed for chat model
)
generate_new_vectors = False  # Set to False if you already have steering vectors saved as "vec_layer_{layer}.pt"


class AttnWrapper(torch.nn.Module):
    """
    Wrapper for attention mechanism to save activations
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


class BlockOutputWrapper(torch.nn.Module):
    """
    Wrapper for block to save activations and unembed them
    """

    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.after_position = None

        self.save_internal_decodings = False
        self.do_projection = False

        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]

        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]


            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = torch.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = torch.dot(last_token_activations, self.calc_dot_product_with)
            self.dot_products.append((top_token, dot_product.cpu().item()))
        if self.add_activations is not None:

            augmented_output = add_vector_after_position(
                matrix=output[0],
                vector=self.add_activations,
                position_ids=kwargs["position_ids"],
                after=self.after_position,
                do_projection=self.do_projection,
            )
            output = (augmented_output + self.add_activations,) + output[1:]

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations, do_projection=False):
        self.add_activations = activations
        self.do_projection = do_projection

    def reset(self):
        self.add_activations = None
        self.activations = None
        self.block.self_attn.activations = None
        self.after_position = None
        self.do_projection = False
        self.calc_dot_product_with = None
        self.dot_products = []

def add_vector_after_position(
    matrix, vector, position_ids, after=None, do_projection=True
):
    # Check the devices of the tensors
    devices = {matrix.device, vector.device, position_ids.device}
    
    # Remove 'cpu' from the set if there are other devices present
    if len(devices) > 1 and torch.device('cpu') in devices:
        devices.remove(torch.device('cpu'))
    
    # Decide the common device
    common_device = next(iter(devices))
    
    # Move tensors to the common device if they are not already there
    matrix = matrix.to(common_device)
    vector = vector.to(common_device)
    position_ids = position_ids.to(common_device)

    after_id = after
    if after_id is None:
        after_id = position_ids.min().item() - 1

    mask = position_ids > after_id
    mask = mask.unsqueeze(-1)

    if do_projection:
        matrix = project_onto_orthogonal_complement(matrix, vector)

    matrix += mask.float() * vector
    return matrix
def find_instruction_end_postion(tokens, end_str):
    start_pos = find_last_subtensor_position(tokens, end_str)
    if start_pos == -1:
        return -1
    return start_pos + len(end_str) - 1
def find_last_subtensor_position(tensor, sub_tensor):
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m, -1, -1):
        if torch.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1

def get_free_memory_on_gpus():
    """
    Get the free memory for each GPU in MB.
    """
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        free_memory = [int(x) for x in result.stdout.decode().strip().split('\n')]
        return free_memory
    except subprocess.CalledProcessError as e:
        print("Error querying nvidia-smi for GPU free memory:", e)
        return []

def get_free_gpus(threshold=0.25):
    """
    Get IDs of GPUs that have at least `threshold` (e.g., 0.25 for 25%) of their memory free.
    """
    free_gpus = []
    free_memory = get_free_memory_on_gpus()
    total_memory = [torch.cuda.get_device_properties(device_id).total_memory / 1e6 for device_id in range(torch.cuda.device_count())]  # Convert bytes to MB

    for device_id, (free, total) in enumerate(zip(free_memory, total_memory)):
        free_mem_ratio = free / total
        
        if free_mem_ratio >= threshold:
            free_gpus.append(device_id)
    
    return free_gpus

size_dict = {
    "<class 'transformers.models.llama.modeling_llama.LlamaDecoderLayer'>": 3286,
    "<class 'torch.nn.modules.sparse.Embedding'>": 1000,
    "<class 'transformers.models.llama.modeling_llama.LlamaRMSNorm'>": 2
}

class Llama2ChatHelperBase:
    def __init__(self, token, system_prompt, model_name, master_device=None,threshold=0.25, add_only_after_end_str=True):

        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token, ignore_mismatched_sizes=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, use_auth_token=token)
        self.model.eval()  # Setting the model to evaluation mode
        self.model.requires_grad_(False)  # Turning off gradient calculation
        self.add_only_after_end_str = add_only_after_end_str
 
        if model_name=="meta-llama/Llama-2-70b-chat-hf":

            for i, layer in enumerate(self.model.model.layers):
                self.model.module.model.layers[i] = BlockOutputWrapper(
                    layer, self.model.module.lm_head, self.model.module.model.norm, self.tokenizer
                )
            object_list=[self.model.model.embed_tokens, self.model.model.norm] +[self.model.model.layers[i] for i in range(len(self.model.model.layers))]
            gpu_ids=chat_helper.get_free_gpus(threshold=threshold)
            current_gpu_idx=0
            for object in tqdm(object_list):
                while get_free_memory_on_gpus()[gpu_ids[current_gpu_idx]]<size_dict[str(type(object))]:
                    current_gpu_idx+=1
                    if current_gpu_idx>=len(gpu_ids):
                        raise ValueError("Not enough memory on GPUs")
                object.to(gpu_ids[current_gpu_idx])
                object.device=gpu_ids[current_gpu_idx]
            def custem_forward(model,inputs):
                
                hidden_states = model.model.embed_tokens(inputs.to(model.model.embed_tokens.device))
                for i, layer in enumerate(model.model.layers):
                    hidden_states = layer(hidden_states.to(model.model.layers[i].device))[0]
                hidden_states = model.model.norm(hidden_states.to(model.model.norm.device))
                return hidden_states
            self.model.forward=custem_forward
        else:


            # Use Data Parallelism if more han one GPU is available

            if master_device!=None:
                self.model = self.model.to(f"cuda:{master_device}")
            free_device_ids = get_free_gpus(threshold)

            # Ensure you have at least one free GPU
            if not free_device_ids:
                raise ValueError("No GPUs meet the free memory threshold.")

            # If master_device is specified, prepend it to the list if not already present
            if master_device is not None:
                free_device_ids = [master_device] + free_device_ids

            # Check and use available GPUs
            if len(free_device_ids) > 1:
                print(f"Using {len(free_device_ids)} GPUs!")
                self.model = DataParallel(self.model, device_ids=free_device_ids)
            for i, layer in enumerate(self.model.module.model.layers):
                self.model.module.model.layers[i] = BlockOutputWrapper(
                    layer, self.model.module.lm_head, self.model.module.model.norm, self.tokenizer
                )
            self.device = next(self.model.module.parameters()).device
        self.END_STR = torch.tensor(self.tokenizer.encode("[/INST]")[1:]).to(self.device)

    def set_save_internal_decodings(self, value):
        for layer in self.model.module.model.layers:
            layer.save_internal_decodings = value
    def set_after_positions(self, pos: int):
        for layer in self.model.module.model.layers:
            layer.after_position = pos

    def set_only_add_to_first_token(self, value):
        for layer in self.model.module.model.layers:
            layer.only_add_to_first_token = value

    def prompt_to_tokens(self, instruction,system_prompt=system_prompt, model_output=""):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
        dialog_tokens = self.tokenizer.encode(
            f"{B_INST} {dialog_content.strip()} {E_INST}{model_output.strip()}"
        )
        return torch.tensor(dialog_tokens).unsqueeze(0)

    def prompt_to_tokens_batched(self, instruction_list, max_length=None):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        dialog_content_list = [
            B_INST + B_SYS + self.system_prompt + E_SYS + instruction.strip()
            for instruction in instruction_list
        ]
        dialog_tokens_list = [self.tokenizer.encode(d) for d in dialog_content_list]
        max_len = max([len(q) for q in dialog_tokens_list])
        filler = self.tokenizer.encode(" ")[1:2]
        end_tokens = self.tokenizer.encode(E_INST)[1:]
        filled_questions = []
        for q in dialog_tokens_list:
            q += filler * (max_len - len(q))
            filled_questions.append(q + end_tokens)
        return [torch.tensor(q).unsqueeze(0) for q in filled_questions]

    def generate_text(self, prompt, max_length=50):
        tokens = self.prompt_to_tokens(prompt)
        return self.generate(tokens, max_new_tokens = max_length)



    def generate(self, tokens, max_new_tokens=50):
        with torch.no_grad():
            tokens = tokens.to(self.device)
            if self.add_only_after_end_str:
                instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            else:
                instr_pos = None
            self.set_after_positions(instr_pos)
            
            generated = self.model.module.generate(
                inputs=tokens, max_new_tokens=max_new_tokens, top_k=1
            )
            return self.tokenizer.batch_decode(generated)[0]

    def generate_text_batched(self, prompts, max_length=100, max_tokens_per_batch=4096):
        # Convert each prompt in the list to tokens
        tokenized_prompts = self.prompt_to_tokens_batched(prompts)

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
        if self.add_only_after_end_str:
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
        else:
            instr_pos = None
        self.set_after_positions(instr_pos)
        # Generate responses for each prompt in the batch
        generated = self.model.module.generate(
            input_ids=tokens,
            max_length=max_length,
        )

        # Decode each response in the batch and return as a list
        return clean_strings(self.tokenizer.batch_decode(generated))

    def get_logits(self, tokens):
        with torch.no_grad():
            if self.add_only_after_end_str:
                instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            else:
                instr_pos = None
            self.set_after_positions(instr_pos)
            logits = self.model.module(tokens).logits
            return logits

    def get_last_activations(self, layer):
        return self.model.module.model.layers[layer].activations

    def set_add_activations(self, layer, activations):
        activations = activations.to(self.device)
        self.model.module.model.layers[layer].add(activations)

    def reset_all(self):
        for layer in self.model.module.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        data = self.get_activation_data(decoded_activations, topk)[0]
        print(label, data)

    def compute_average_loss(self, token_ids_batch):
        logits = self.get_logits(token_ids_batch)
        labels = token_ids_batch[:, 1:].contiguous()
        logits = logits[:, :-1].contiguous()

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean"
        )

        return loss.item()

    def decode_all_layers(
        self,
        tokens,
        topk=10,
        print_attn_mech=True,
        print_intermediate_res=True,
        print_mlp=True,
        print_block=True,
    ):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        for i, layer in enumerate(self.model.model.layers):
            print(f"Layer {i}: Decoded intermediate outputs")
            if print_attn_mech:
                self.print_decoded_activations(
                    layer.attn_out_unembedded, "Attention mechanism", topk=topk
                )
            if print_intermediate_res:
                self.print_decoded_activations(
                    layer.intermediate_resid_unembedded,
                    "Intermediate residual stream",
                    topk=topk,
                )
            if print_mlp:
                self.print_decoded_activations(
                    layer.mlp_out_unembedded, "MLP output", topk=topk
                )
            if print_block:
                self.print_decoded_activations(
                    layer.block_output_unembedded, "Block output", topk=topk
                )

    def plot_decoded_activations_for_layer(self, layer_number, tokens, topk=10):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        layer = self.model.model.layers[layer_number]

        data = {}
        data["Attention mechanism"] = self.get_activation_data(
            layer.attn_out_unembedded, topk
        )[1]
        data["Intermediate residual stream"] = self.get_activation_data(
            layer.intermediate_resid_unembedded, topk
        )[1]
        data["MLP output"] = self.get_activation_data(layer.mlp_out_unembedded, topk)[1]
        data["Block output"] = self.get_activation_data(
            layer.block_output_unembedded, topk
        )[1]

        # Plotting
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
        fig.suptitle(f"Layer {layer_number}: Decoded Intermediate Outputs", fontsize=21)

        for ax, (mechanism, values) in zip(axes.flatten(), data.items()):
            tokens, scores = zip(*values)
            ax.barh(tokens, scores, color="skyblue")
            ax.set_title(mechanism)
            ax.set_xlabel("Value")
            ax.set_ylabel("Token")

            # Set scientific notation for x-axis labels when numbers are small
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def get_activation_data(self, decoded_activations, topk=10):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))


class Llama13BChatHelper(Llama2ChatHelperBase):
    def __init__(self, token, system_prompt, master_device=None):
        super().__init__(token, system_prompt, "meta-llama/Llama-2-13b-chat-hf", master_device=master_device)


class Llama30BChatHelper(Llama2ChatHelperBase):
    def __init__(self, token, system_prompt,master_device=None):
        super().__init__(token, system_prompt, "meta-llama/Llama-2-30b-chat-hf",  master_device=master_device)


class Llama70BChatHelper(Llama2ChatHelperBase):
    def __init__(self, token, system_prompt,master_device=None,threshold=0.25):
        super().__init__(token, system_prompt, "meta-llama/Llama-2-70b-chat-hf",  master_device=master_device, threshold=threshold)


class Llama7BChatHelper(Llama2ChatHelperBase):
    def __init__(self, token, system_prompt, master_device=None,threshold=0.25):
        super().__init__(token, system_prompt, "meta-llama/Llama-2-7b-chat-hf",master_device=master_device, threshold=threshold)


def prompt_to_tokens(tokenizer, system_prompt, instruction, model_output):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
    dialog_tokens = tokenizer.encode(
        f"{B_INST} {dialog_content.strip()} {E_INST}{model_output.strip()}"
    )
    return torch.tensor(dialog_tokens).unsqueeze(0)


class ComparisonDataset(Dataset):
    def __init__(self, data, system_prompt, model_name):
        self.data = data
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        s_text = item["rejected"]
        n_text = item["chosen"]
        q_text = item["prompt"]
        s_tokens = prompt_to_tokens(self.tokenizer, self.system_prompt, q_text, s_text)
        n_tokens = prompt_to_tokens(self.tokenizer, self.system_prompt, q_text, n_text)
        return s_tokens, n_tokens


def generate_and_save_steering_vectors(
    model, dataset, start_layer=15, end_layer=29, data_path=".", save_all_diffs=False
):
    layers = list(range(start_layer, end_layer + 1))
    diffs = dict([(layer, []) for layer in layers])
    model.set_save_internal_decodings(False)
    model.reset_all()
    for s_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
        s_tokens = s_tokens.to(model.device)
        n_tokens = n_tokens.to(model.device)
        model.get_logits(s_tokens)
        for layer in layers:
            s_activations = model.get_last_activations(layer)
            s_activations = s_activations[0, -2, :].detach().cpu()
            diffs[layer].append(s_activations)
        model.get_logits(n_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            n_activations = n_activations[0, -2, :].detach().cpu()
            diffs[layer][-1] -= n_activations
    for layer in layers:
        diffs[layer] = torch.stack(diffs[layer])
        if save_all_diffs:
            torch.save(
                diffs[layer], os.path.join(data_path, f"all_diffs_layer_{layer}.pt")
            )
        vec = diffs[layer].mean(dim=0)
        torch.save(vec, os.path.join(data_path, f"vec_layer_{layer}.pt"))


def transform_df_to_dict_list(df):
    result = []

    for _, row in df.iterrows():
        # Split by "Answer:" to find the common prompt and unique endings
        common_prompt, chosen_ending = row["chosen"].split("Answer:")
        _, rejected_ending = row["rejected"].split("Answer:")

        # Remove any leading or trailing whitespace for accuracy
        chosen_ending = chosen_ending.strip()
        rejected_ending = rejected_ending.strip()

        result.append(
            {
                "prompt": common_prompt.strip(),
                "chosen": chosen_ending,
                "rejected": rejected_ending,
            }
        )

    return result


def get_vec(data_path, layer):
    return torch.load(os.path.join(data_path, f"vec_layer_{layer}.pt"))


def clean_strings(strings):
    def clean_string(input_str):
        # Remove </s> and <unk> tokens
        cleaned = re.sub(r"</s>|<unk>", "", input_str)
        # Strip any leading or trailing whitespace
        return cleaned.strip()

    return [clean_string(s) for s in strings]