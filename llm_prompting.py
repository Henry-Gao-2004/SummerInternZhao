import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Llama Prompting Script")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU device id")
    parser.add_argument("--models", type=str, default="Qwen/Qwen2.5-7B,meta-llama/Llama-3.1-8B,deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Comma separated list of model names to use")
    parser.add_argument("--prompt_file", type=str, default="prompts.txt", help="Path to the file containing prompts")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum new tokens for generation")
    return parser.parse_args()

def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        rope_scaling={"type": "dynamic", "factor": 8.0},
        trust_remote_code=True
    )
    model.to(device)
    return tokenizer, model

def load_prompts(prompt_file):
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def generate_output(prompt, tokenizer, model, device, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 从命令行参数获取多个模型，模型名称以逗号分隔
    model_names = [m.strip() for m in args.models.split(",")]
    
    prompts = load_prompts(args.prompt_file)
    
    default_system_prompt = "System: You are a conversational agent playing the role of the Doctor. Based on the conversation context provided below, generate only the next turn’s response as the Doctor. Your response should include only that reply.\n"
    
    # default_system_prompt = "System: You are a conversational agent playing the role of the Listener. Based on the conversation context provided below, generate only the next turn’s response as the Listener.\n"
    
    # 对于每个模型，加载模型并依次生成每个 prompt 的回复
    for model_name in model_names:
        print(f"\n=== Using model: {model_name} ===")
        tokenizer, model = load_model(model_name, device)
        for idx, prompt in enumerate(prompts):
            # 如果 prompt 中未包含对话标识，则自动添加
            if "User:" not in prompt:
                full_prompt = default_system_prompt + "User: " + prompt + "Please think step by step before give me your final response as a Doctor." + "\nAssistant:" 
            else:
                full_prompt = prompt
            print(f"\n=== Prompt {idx + 1} ===")
            output = generate_output(full_prompt, tokenizer, model, device, args.max_new_tokens)
            print(output)
        
        # 释放当前模型占用的 GPU 内存，准备加载下一个模型
        del model, tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()