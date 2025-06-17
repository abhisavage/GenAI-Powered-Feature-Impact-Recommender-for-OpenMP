#!/usr/bin/env python3
import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from clang import cindex
from ast_analyzer import initialize_clang, extract_functions_from_github
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the GitHub token
github_token = os.getenv("GITHUB_TOKEN") # Replace or secure later

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model.to(device), device

def infer(prompt, tokenizer, model, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    output = model.generate(**inputs, max_length=256, num_beams=4, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def parse_model_output(output):
    files = set()
    full_funcs = []
    for part in output.split(","):
        if "::" in part:
            try:
                file, func = part.strip().split("::", 1)
                files.add(file.strip())
                full_funcs.append(f"{file.strip()}::{func.strip()}")
            except ValueError:
                continue
    return sorted(files), sorted(set(full_funcs))

def set_libclang(libclang_path=None):
    if libclang_path:
        cindex.Config.set_library_file(libclang_path)
    else:
        default = "C:/Program Files/LLVM/bin/libclang.dll"
        if os.path.exists(default):
            cindex.Config.set_library_file(default)
        else:
            raise FileNotFoundError("❌ libclang.dll not found. Please install LLVM or pass path via --libclang")

def main():
    parser = argparse.ArgumentParser(description="OpenMP Feature Impact CLI Tool")
    parser.add_argument("prompt", type=str, help="Feature prompt like 'taskwait codegen'")
    parser.add_argument("--model", type=str, default="C:/Users/abhis/Desktop/CD_LLVM/omp_t5_model", help="Path to fine-tuned T5 model")
    parser.add_argument("--libclang", type=str, help="Path to libclang.dll if not in default location")
    args = parser.parse_args()

    # Load model
    tokenizer, model, device = load_model(args.model)

    # Initialize Clang
    set_libclang(args.libclang)
    initialize_clang(args.libclang)

    # Run inference
    output = infer(args.prompt, tokenizer, model, device)
    files, full_funcs = parse_model_output(output)

    print("\n🔮 Predicted Files:")
    for f in files:
        print("  •", f)

    print("\n🔧 Predicted Functions:")
    for ff in full_funcs:
        print("  •", ff)

    # Run AST Analysis
    repo = "llvm/llvm-project"
    ast_map = extract_functions_from_github(repo, files, github_token)

    print("\n🧩 AST Validation:")
    for f in files:
        print(f"\n{f}:")
        if f in ast_map:
            for fn in ast_map[f]:
                match = f"{f}::{fn['name']}"
                flag = "✅" if match in full_funcs else "  "
                print(f"  {flag} {fn['name']} @ line {fn['line']}")
        else:
            print("  ⚠️ No AST data available")

if __name__ == "__main__":
    main()
