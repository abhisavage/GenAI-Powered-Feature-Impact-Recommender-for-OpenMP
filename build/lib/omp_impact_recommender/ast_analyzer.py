import os
import requests
import tempfile
import shutil
from clang import cindex



def initialize_clang(libclang_path=None):
    if libclang_path:
        cindex.Config.set_library_file(libclang_path)

def download_github_file(repo, path, github_token):
    """Download a single file from a GitHub repo using the REST API."""
    headers = {"Authorization": f"token {github_token}"}
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    content = response.json()
    if content.get("encoding") == "base64":
        from base64 import b64decode
        return b64decode(content["content"])
    else:
        raise ValueError("Unexpected file encoding")

def extract_functions_from_github(repo, file_paths, github_token):
    """Downloads files from GitHub and extracts function declarations using Clang."""
    index = cindex.Index.create()
    results = {}

    # Use a temporary directory for downloaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        local_files = []
        for path in file_paths:
            try:
                code = download_github_file(repo, path, github_token)
                local_path = os.path.join(temp_dir, os.path.basename(path))
                with open(local_path, "wb") as f:
                    f.write(code)
                local_files.append((path, local_path))
            except Exception as e:
                print(f"⚠️ Failed to download {path}: {e}")

        for original_path, local_path in local_files:
            try:
                tu = index.parse(local_path, args=["-std=c++17"])
                funcs = []
                def visit(node):
                    if node.kind == cindex.CursorKind.FUNCTION_DECL:
                        funcs.append({
                            "name": node.spelling,
                            "line": node.location.line
                        })
                    for c in node.get_children():
                        visit(c)
                visit(tu.cursor)
                results[original_path] = funcs
            except Exception as e:
                print(f"❌ Failed to parse {original_path}: {e}")

    # The temporary directory and files are deleted here automatically
    return results
