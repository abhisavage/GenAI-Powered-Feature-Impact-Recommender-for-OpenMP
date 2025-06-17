import argparse
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from clang import cindex
from ast_analyzer import initialize_clang, extract_and_match_functions
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the GitHub token
github_token = os.getenv("GITHUB_TOKEN")

def load_model(model_path):
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tok, mdl.to(device), device

def suggest(tok, mdl, device, prompt):
    i = tok(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    o = mdl.generate(**i, max_length=256, num_beams=4, early_stopping=True)
    return tok.decode(o[0], skip_special_tokens=True)

def parse_model_output(text):
    files_set, full_funcs = set(), []
    parts = [s.strip() for s in text.split(",") if "::" in s]
    for part in parts:
        try:
            file, func = part.split("::", 1)
            files_set.add(file.strip())
            full_funcs.append(f"{file.strip()}::{func.strip()}")
        except ValueError:
            continue
    return sorted(files_set), sorted(set(full_funcs))

def set_libclang(libclang_path=None):
    if libclang_path:
        cindex.Config.set_library_file(libclang_path)
    else:
        default_path = "C:/Program Files/LLVM/bin/libclang.dll"
        if os.path.exists(default_path):
            cindex.Config.set_library_file(default_path)
        else:
            raise FileNotFoundError("❌ libclang.dll not found. Please install LLVM or pass --libclang path.")

def batch_test(tok, mdl, device):
    prompts = [
        "taskwait codegen", "flush ir target", "parallel parse runtime",
        "atomic sema", "for codegen parse", "sections runtime ast",
        "ordered flush", "barrier codegen", "masked parse ast",
        "taskgroup codegen"
    ]
    for prompt in prompts:
        print(f"\n🧠 Prompt: {prompt}")
        out = suggest(tok, mdl, device, prompt)
        files, full_funcs = parse_model_output(out)
        print("  📁 Predicted Files:")
        for f in files:
            print(f"    • {f}")
        print("  🔧 Predicted Functions:")
        for f in full_funcs:
            print(f"    • {f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", help="Prompt like 'taskwait codegen'")
    parser.add_argument("--model", default="./omp_t5_model", help="Path to T5 model")
    parser.add_argument("--libclang", help="Path to libclang shared library (libclang.dll)")
    parser.add_argument("--batch", action="store_true", help="Run 10 sample prompts instead of interactive mode")
    args = parser.parse_args()

    if not github_token:
        raise ValueError("❌ GitHub token is required.")

    # Setup Clang
    set_libclang(args.libclang)
    initialize_clang(args.libclang)

    # Load model
    tok, mdl, device = load_model(args.model)

    if args.batch:
        batch_test(tok, mdl, device)
        return

    # Get prompt
    if args.prompt:
        feature_prompt = args.prompt.strip()
    else:
        feature_prompt = input("📝 Enter feature prompt (e.g. 'taskwait codegen'): ").strip()

    out = suggest(tok, mdl, device, feature_prompt)
    files, full_funcs = parse_model_output(out)

    print("\n🔮 Predicted Files:")
    for f in files:
        print(f"  • {f}")
    print("\n🔧 Predicted Functions:")
    for ff in full_funcs:
        print(f"  • {ff}")

    # AST Match
    repo = "llvm/llvm-project"
    ast_map = extract_and_match_functions(repo, files, full_funcs, github_token)

    print("\n🧩 AST Match Results:")
    for file in files:
        print(f"\n📄 {file}:")
        for func in ast_map.get(file, []):
            flag = "✅" if f"{file}::{func['name']}" in full_funcs else "  "
            print(f"  {flag} {func['name']} @ line {func['line']}")

if __name__ == "__main__":
    main()
