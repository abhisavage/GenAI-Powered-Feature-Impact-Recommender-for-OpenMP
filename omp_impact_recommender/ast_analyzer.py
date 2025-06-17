import os
import requests
import tempfile
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


def extract_and_match_functions(repo, file_paths, predicted_entries, github_token):
    """
    Downloads and parses files. Returns matches for functions, methods, classes, structs, and enums.
    """
    index = cindex.Index.create()
    results = {}

    # Map of filename -> set of expected names
    expected_map = {} 
    for full in predicted_entries:
        if "::" in full:
            file, name = full.split("::", 1)
            expected_map.setdefault(file.strip(), set()).add(name.strip())

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in file_paths:
            try:
                content = download_github_file(repo, file, github_token)
                local_path = os.path.join(temp_dir, os.path.basename(file))
                with open(local_path, "wb") as f:
                    f.write(content)

                tu = index.parse(
                    local_path,
                    args=["-std=c++17"],
                    options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
                )

                # for diag in tu.diagnostics:
                #     print(f"⚠️ Diagnostic in {file}: {diag}")

                matches = []

                def visit(node):
                    if node.kind in {
                        cindex.CursorKind.FUNCTION_DECL,
                        cindex.CursorKind.CXX_METHOD,
                        cindex.CursorKind.CONSTRUCTOR,
                        cindex.CursorKind.CLASS_DECL,
                        cindex.CursorKind.STRUCT_DECL,
                        cindex.CursorKind.ENUM_DECL
                    }:
                        symbol_name = node.spelling or node.displayname
                        if symbol_name and any(expected in symbol_name for expected in expected_map.get(file, set())):
                            matches.append({
                                "name": symbol_name,
                                "line": node.location.line,
                                "kind": node.kind.name
                            })
                    for c in node.get_children():
                        visit(c)

                visit(tu.cursor)
                if matches:
                    results[file] = matches

            except Exception as e:
                print(f"{file}: {e}")

    return results
