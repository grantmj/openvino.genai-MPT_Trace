# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import argparse
import openvino_genai as ov_genai
from openvino import get_version

def main():
    parser = argparse.ArgumentParser(description="Help command")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to model and tokenizers base directory")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="Prompt")
    parser.add_argument("-pf", "--prompt_file", type=str, default=None, help="Read prompt from file")
    parser.add_argument("--dataset", type=str, default=None, help="Name of the dataset to use for benchmarking (e.g., openai_humaneval)")
    parser.add_argument("-nw", "--num_warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument("-n", "--num_iter", type=int, default=2, help="Number of iterations")
    parser.add_argument("-mt", "--max_new_tokens", type=int, default=20, help="Maximal number of new tokens")
    parser.add_argument("-d", "--device", type=str, default="CPU", help="Device")
    
    args = parser.parse_args()

    prompts = []
    if args.dataset:
        from datasets import load_dataset
        print(f"Loading dataset: {args.dataset}")
        human_eval_dataset = load_dataset("openai_humaneval", split="test") 
        prompts = [item['prompt'] for item in human_eval_dataset]
        print(f"Loaded {len(prompts)} prompts from the dataset.")
    elif args.prompt_file is not None:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompts = [f.read()]
    elif args.prompt is not None:
        prompts = [args.prompt]
    else:
        prompts = ['The Sky is blue because']

    if len(prompts) == 0:
        raise RuntimeError(f'No prompts were loaded!')

    print(f'openvino runtime version: {get_version()}')

    models_path = args.model
    device = args.device
    num_warmup = args.num_warmup
    num_iter = args.num_iter
    
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens

    scheduler_config = ov_genai.SchedulerConfig()
    scheduler_config.enable_prefix_caching = False
    scheduler_config.max_num_batched_tokens = sys.maxsize

    pipe = ov_genai.LLMPipeline(models_path, device, scheduler_config=scheduler_config)
    
    print("Running warmup...")
    for _ in range(num_warmup):
        pipe.generate([prompts[0]], config)

    print(f"Running benchmark on {len(prompts)} prompts...")
    total_perf_metrics = None

    for i, p in enumerate(prompts):
        res = pipe.generate([p], config)
        print(f"Generated for prompt {i+1}/{len(prompts)}")
        
        if total_perf_metrics is None:
            total_perf_metrics = res.perf_metrics
        else:
            total_perf_metrics += res.perf_metrics
    
    print(f"--- Benchmark Results for {len(prompts)} prompts ---")
    print(f"Load time: {total_perf_metrics.get_load_time():.2f} ms")
    print(f"Average Generate time: {total_perf_metrics.get_generate_duration().mean:.2f} ± {total_perf_metrics.get_generate_duration().std:.2f} ms")
    print(f"Average Tokenization time: {total_perf_metrics.get_tokenization_duration().mean:.2f} ± {total_perf_metrics.get_tokenization_duration().std:.2f} ms")
    print(f"Average Detokenization time: {total_perf_metrics.get_detokenization_duration().mean:.2f} ± {total_perf_metrics.get_detokenization_duration().std:.2f} ms")
    print(f"Average TTFT: {total_perf_metrics.get_ttft().mean:.2f} ± {total_perf_metrics.get_ttft().std:.2f} ms")
    print(f"Average TPOT: {total_perf_metrics.get_tpot().mean:.2f} ± {total_perf_metrics.get_tpot().std:.2f} ms")
    print(f"Average Throughput : {total_perf_metrics.get_throughput().mean:.2f} ± {total_perf_metrics.get_throughput().std:.2f} tokens/s")

if __name__ == "__main__":
    main()
