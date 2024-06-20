import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
from datetime import datetime
from dataloader import DataLoaderLite
from model import GPT, GPTConfig
import tiktoken
import numpy as np
from utils import load_tokens, get_most_likely_row
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import yaml
from utils import load_yaml_config
import logging

config = load_yaml_config('configuration/training.yaml')


torch.set_float32_matmul_precision(config['matmul_precision'])
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# simple launch:
# python pretraining.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 pretraining.py


# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now I think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"


seed = config['training']['seed']
if seed is None:
    logging.warning("The seed has not been set. Consider doing this if you want reproducible results.")
else:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

enc = tiktoken.get_encoding("gpt2") #gpt-2-small is my smallest version


B = config['training']['micro_batch_size'] # micro batch size
T = config['training']['sequence_length'] # sequence length
accumulation_steps = config['training']['accumulation_steps']
start_from_checkpoint = config['training']['start_from_checkpoint']
total_batch_size = B*T*accumulation_steps #524288 # 2**19, ~0.5M, in number of tokens

eval_step = config['evaluation']['eval_steps']
sampling = config['evaluation']['sampling']
sampling_step = config['evaluation']['sampling_steps']
if sampling_step:
    try:
        sampling_sentence = config['evaluation']['sampling_sentence']
    except Exception as e:
        logging.error("If you set sampling: true, sampling_sentence must be filled")
        exit(-1)

checkpoint_steps = int(config['checkpoint']['checkpoint_steps'])
max_lr = float(config['training']['max_lr'])
min_lr = float(config['training']['min_lr'])
weight_decay = float(config['training']['weight_decay'])
warmup_steps = float(config['training']['warmup_steps'])
epoch = int(config['training']['epoch'])
max_steps = int(config['training']['max_steps'])*epoch # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens


assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(data_folder=config['training']['dataset_root_folder'],
                            master_process=master_process,
                            B=B, 
                            T=T, 
                            process_rank=ddp_rank, 
                            num_processes=ddp_world_size, 
                            split="train")
val_loader = DataLoaderLite(data_folder=config['training']['dataset_root_folder'],
                            master_process=master_process,
                            B=B, 
                            T=T, 
                            process_rank=ddp_rank, 
                            num_processes=ddp_world_size, 
                            split="val")
enc.n_vocab
# create model
starting_step = 0
model = GPT(GPTConfig(vocab_size=enc.n_vocab))
if master_process:
    model.master_process = master_process
if start_from_checkpoint:
    logging.info(f"Starting from checkpoint {config['training']['checkpoint']}")
    state = torch.load(config['training']['checkpoint'],map_location=torch.device(device))
    model.load_state_dict(state['model'])
    starting_step = state['step']
    logging.info(f"Moving to step: {starting_step}")
    for s in range(0,starting_step):
        train_loader.next_batch()

model.to(device)
use_compile = config['training']['use_torch_compile'] # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model



def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=max_lr, device_type=device_type)
if start_from_checkpoint:
    optimizer.load_state_dict(state['optimizer_state_dict'])
    

# create the log directory we will write checkpoints to and log to
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_dir = f"logs/{timestamp}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"{timestamp}_log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass


for step in range(starting_step,max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % eval_step == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
    # -----------------------------------------------------------------------------------------------------------------------
    # Save checkpoint step
    if step > 0 and (step % checkpoint_steps == 0 or last_step):
        # optionally write model checkpoints
        checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        checkpoint = {
            'model': raw_model.state_dict(),
            'config': raw_model.config,
            'step': step,
            'loss': loss.item(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss_accum.item()
        }
        # you might also want to add optimizer.state_dict() and
        # rng seeds etc., if you wanted to more exactly resume training
        torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    '''if (step % eval_step == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")'''

    # -----------------------------------------------------------------------------------------------------------------------
    # Sampling step
    if sampling and ((step > 0 and step % sampling_step == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = int(config['evaluation']['num_sampling_sentences'])
        max_length = int(config['evaluation']['sampling_max_length'])
        tokens = enc.encode(sampling_sentence)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(10 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:        
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
