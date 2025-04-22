# --- START OF FILE preprocess.py ---

import os
import time
import tiktoken
import torch
import numpy as np # <--- Import NumPy
from datasets import load_dataset
from tqdm import tqdm
import yaml
import multiprocessing as mp
from functools import partial
import traceback 


os.environ["HF_TOKEN"] = "Enter token Here"
hf_token = os.environ.get("HF_TOKEN", None) 

def process_shard_worker(dataset_config, cache_dir):
    """Process a dataset subset - designed to be run in a separate process."""
    try:
        
        tokenizer = tiktoken.get_encoding("gpt2")

        context_len = dataset_config.get("context_len", 1024)
        chunk_len = context_len + 1
        bytes_per_token = 8
        approx_shard_size_mb = 256 
        max_chunks_per_shard = (approx_shard_size_mb * 1024 * 1024) // (chunk_len * bytes_per_token)
        if max_chunks_per_shard <= 0:
             max_chunks_per_shard = 1000 # Set a minimum if calculation is too low
        print(f"[{os.getpid()}] {dataset_config['name']} - Max chunks per shard: {max_chunks_per_shard}")


        max_tokens = dataset_config.get("max_tokens", None)
        total_processed_tokens = 0
        shard_idx = 0
        buffer = []

        output_dir = os.path.join(cache_dir, dataset_config['name'])
        os.makedirs(output_dir, exist_ok=True)

        kwargs = dataset_config.get("kwargs", {})
        if hf_token and dataset_config.get("needs_auth", False):
            kwargs["token"] = hf_token
            kwargs["use_auth_token"] = hf_token


        print(f"[{os.getpid()}] Starting processing for: {dataset_config['name']}")

        ds = load_dataset(
            dataset_config["hf_path"],
            split=dataset_config.get("split", "train"),
            streaming=True,
            trust_remote_code=dataset_config.get("trust_remote_code", True), # Default to True or get from config
            **kwargs
        )

        pbar = tqdm(desc=f"[{os.getpid()}] {dataset_config['name']}", unit=" chunks", smoothing=0)


        ds_iterator = iter(ds)

        while True:
            try:
                example = next(ds_iterator)
            except StopIteration:
                print(f"[{os.getpid()}] Reached end of stream for {dataset_config['name']}.")
                break # End of dataset stream
            except Exception as e:
                 print(f"[{os.getpid()}] Error fetching next example for {dataset_config['name']}: {e}. Skipping.")
                 continue # Skip problematic example

            text = example.get(dataset_config.get("text_field", "text"), "")
            if not text: # Skip empty examples
                continue

            # Encode text + EOS token
            try:
                tokens = tokenizer.encode(
                    text + "<|endoftext|>",
                    allowed_special={'<|endoftext|>'}
                )
            except Exception as e:
                print(f"[{os.getpid()}] Error encoding text for {dataset_config['name']}: {e}. Skipping example.")
                continue

            # Process tokens in chunks
            for i in range(0, len(tokens), chunk_len):
                chunk = tokens[i : i + chunk_len]

                # Only add complete chunks
                if len(chunk) == chunk_len:
                    
                    buffer.append(torch.tensor(chunk, dtype=torch.int64))
                    current_chunk_tokens = chunk_len
                    total_processed_tokens += current_chunk_tokens
                    pbar.update(1)

                    # Save buffer when full
                    if len(buffer) >= max_chunks_per_shard:
                        # --- Convert to NumPy and Save ---
                        shard_filename = os.path.join(output_dir, f"shard_{shard_idx}_pid{os.getpid()}.npy") # <--- .npy extension
                        try:
                            # Stack tensors into a single 2D NumPy array
                            shard_data_np = torch.stack(buffer).numpy()
                            np.save(shard_filename, shard_data_np) # <--- Save as .npy
                            print(f"[{os.getpid()}] Saved {shard_filename} ({len(buffer)} chunks, {shard_data_np.nbytes / 1e6:.2f} MB)")
                        except Exception as e:
                             print(f"[{os.getpid()}] !!! ERROR saving shard {shard_filename}: {e}")
                        buffer = []
                        shard_idx += 1


                    if max_tokens and total_processed_tokens >= max_tokens:
                        print(f"[{os.getpid()}] Max tokens ({max_tokens:,}) reached for {dataset_config['name']}.")
                        break 

    
            if max_tokens and total_processed_tokens >= max_tokens:
                break 

        if buffer:
            shard_filename = os.path.join(output_dir, f"shard_{shard_idx}_pid{os.getpid()}.npy") # <--- .npy extension
            try:
                
                shard_data_np = torch.stack(buffer).numpy()
                np.save(shard_filename, shard_data_np)
                print(f"[{os.getpid()}] Saved final {shard_filename} ({len(buffer)} chunks, {shard_data_np.nbytes / 1e6:.2f} MB)")
                # --- End NumPy Save ---
            except Exception as e:
                 print(f"[{os.getpid()}] !!! ERROR saving final shard {shard_filename}: {e}")
                 traceback.print_exc()

        pbar.close()
        print(f"[{os.getpid()}] Finished processing {dataset_config['name']}. Total tokens processed: {total_processed_tokens:,}")
        return total_processed_tokens

    except Exception as e:
        print(f"!!!!!! FATAL ERROR in process {os.getpid()} for dataset {dataset_config.get('name', 'Unknown')} !!!!!")
        traceback.print_exc()
        print(f"Error details: {e}")
        return 0 


if __name__ == "__main__":
  
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        print("Could not set 'spawn' start method, using default.")


    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    dataset_conf = config["dataset"]
    cache_dir = dataset_conf["cache_dir"]
    target_tokens = dataset_conf.get("target_tokens", 5_000_000_000) # Example target
    subsets = dataset_conf["subsets"]

    max_procs = config.get("max_processes", 4)
    num_processes = min(mp.cpu_count(), len(subsets), max_procs) if subsets else 1
    if not subsets:
         print("Warning: No subsets defined in config.yaml")
         num_processes = 0

    if num_processes > 0:
        print(f"Using {num_processes} processes.")
        os.makedirs(cache_dir, exist_ok=True)

        # Create a pool of worker processes
        pool = mp.Pool(processes=num_processes)
        total_tokens_processed = 0

        # Create the partial function with the fixed cache_dir argument
        worker_func = partial(process_shard_worker, cache_dir=cache_dir)

        print("Submitting tasks to process pool...")
        # Use map_async for non-blocking execution and better error handling
        async_result = pool.map_async(worker_func, subsets)

        pool.close() 

        print("Waiting for workers to finish...")
        try:
            # Wait for all processes to complete with a timeout
            processed_tokens_list = async_result.get(timeout=config.get("process_timeout_seconds", 7200)) # e.g., 2 hours timeout
            print("All workers finished processing.")

            # Aggregate results
            for i, subset_tokens in enumerate(processed_tokens_list):
                subset_name = subsets[i].get('name', f'Subset {i+1}')
                if subset_tokens is not None: # Check if worker returned successfully
                     print(f"Subset '{subset_name}' processed {subset_tokens:,} tokens.")
                     total_tokens_processed += subset_tokens
                else:
                     print(f"Subset '{subset_name}' failed or returned None.")


        except mp.TimeoutError:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("TIMEOUT ERROR: One or more processes took too long.")
            print("Total token count might be incomplete.")
            print("Consider increasing 'process_timeout_seconds' in config or checking worker logs/errors.")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # Attempt to terminate the pool if timeout occurs
            pool.terminate()
            pool.join() # Wait for termination

        except KeyboardInterrupt:
             print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             print("KeyboardInterrupt received. Terminating worker processes...")
             print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
             pool.terminate()
             pool.join() # Wait for termination

        except Exception as e:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"An error occurred while coordinating workers: {e}")
            traceback.print_exc()
            print("Attempting to terminate pool...")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            pool.terminate()
            pool.join() # Wait for termination


        if 'pool' in locals() and pool._state == mp.pool.RUN: 
             pool.join()

        print("\n--- Preprocessing Summary ---")
        print(f"Total processed tokens across all successful subsets: {total_tokens_processed:,}")
        if "target_tokens" in dataset_conf:
            print(f"Target tokens: {target_tokens:,}")
            if total_tokens_processed >= target_tokens:
                print("Global target token count reached or exceeded.")
            else:
                print("Global target token count NOT reached.")
        print("Preprocessing finished.")
    else:
        print("No subsets to process or num_processes is 0.")