import os
import re
import json
from pathlib import Path
from tqdm import tqdm
from github import Github
import requests
from unidiff import PatchSet, UnidiffParseError
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the GitHub token
github_token = os.getenv("GITHUB_TOKEN")
REPO_NAME = "llvm/llvm-project"
OUT_FILE = "all_openmp_prs.jsonl"
MAX_PRS = 1500

FEATURE_KEYWORDS = {
    "layer": ["parse", "sema", "codegen", "runtime", "ir", "ast", "ompirbuilder", "frontend", "lexer", "parser"],
    "directives": [
        "task", "taskwait", "taskgroup", "parallel", "for", "sections", "single", "master",
        "critical", "barrier", "atomic", "flush", "ordered", "simd", "target", "teams", "distribute",
        "declare", "threadprivate", "allocate", "defaultmap", "requires", "metadirective", "masked", "detach"
    ]
}

# ==================== INIT ====================
g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)
Path("logs").mkdir(exist_ok=True)

def match_keywords(text):
    text = text.lower()
    layer = [kw for kw in FEATURE_KEYWORDS["layer"] if kw in text]
    directives = [kw for kw in FEATURE_KEYWORDS["directives"] if kw in text]
    return layer, directives

def fetch_patch(pr_number):
    try:
        url = f"https://patch-diff.githubusercontent.com/raw/{REPO_NAME}/pull/{pr_number}.patch"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.text
    except Exception as e:
        with open("logs/failed_downloads.txt", "a") as f:
            f.write(f"{pr_number} - download error: {e}\n")
    return None

def extract_functions_from_patch(patch_text, pr_number):
    try:
        patch = PatchSet(patch_text.splitlines(keepends=True))
    except UnidiffParseError as e:
        with open("logs/failed_parsing.txt", "a") as f:
            f.write(f"{pr_number} - parse error: {e}\n")
        return {}

    file_function_map = {}
    for patched_file in patch:
        fname = patched_file.path
        if not fname.endswith(('.cpp', '.cc', '.c', '.h', '.inc')):
            continue

        func_set = set()
        for hunk in patched_file:
            for line in hunk:
                if line.is_added or line.is_removed:
                    m = re.search(r'^\s*(?:[\w:*&<>\[\]]+\s+)+(?P<name>\w+)\s*\([^;]*\)\s*[{;]?', line.value)
                    if m:
                        func_set.add(m.group("name"))

        if func_set:
            file_function_map[fname] = list(func_set)

    return file_function_map

# ==================== MAIN LOOP ====================
print("[*] Starting PR extraction...")
prs = repo.get_pulls(state="closed", sort="updated", base="main")

with open(OUT_FILE, "a", encoding="utf-8") as outfile:
    count = 0
    for pr in tqdm(prs, desc="Processing PRs"):
        if count >= MAX_PRS:
            break
        try:
            pr_number = pr.number
            title = pr.title or ""
            body = pr.body or ""
            full_text = f"{title}\n{body}"

            layer_tags, directive_tags = match_keywords(full_text)
            if not (layer_tags or directive_tags):
                continue

            patch_text = fetch_patch(pr_number)
            if not patch_text:
                continue

            file_func_map = extract_functions_from_patch(patch_text, pr_number)
            if not file_func_map:
                continue

            record = {
                "pr_number": pr_number,
                "url": pr.html_url,
                "title": title,
                "body": body,
                "feature_layers": layer_tags,
                "feature_directives": directive_tags,
                "files_changed": [
                    {"filename": fname, "functions": fnlist}
                    for fname, fnlist in file_func_map.items()
                ]
            }

            outfile.write(json.dumps(record) + "\n")
            count += 1

        except Exception as e:
            with open("logs/failed_prs.txt", "a") as f:
                f.write(f"{pr.number} - general error: {e}\n")
