import os
import re
import json
import tempfile
import subprocess
import shutil
from typing import Iterable, Tuple, List, Dict, Any
from github import Github, Auth
from git import Repo
from datasets import Dataset, DatasetDict

# --- Configuration and Authentication ---

# INPUT_FILE = "issta_issues.json"
# OUTPUT_FILE = "issta_SWT_structure.json"
INPUT_FILE = "swt_test_issues.json"
OUTPUT_FILE = "swt_test_output.json"
DATASETS_DIR = "datasets"
RETRIEVAL_DIR = "retrieval_results"
PROMPTS_DIR = "prompts"
TOKEN_CAP = 27000 # Approx token cap from build_zeroShotPlus_prompt.py

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
PROMPT_TEMPLATE = """The following text contains a user issue (in <issue/> brackets) posted at a repository.
Further, you are provided with file contents of several files in the repository that
contain relevant code (in <code> brackets). It may be necessary to use code from
third party dependencies or files not contained in the attached documents however.
Your task is to identify the issue and implement a test case that verifies a
proposed solution to this issue. More details at the end of this text.

<issue>
{issue_text}
</issue>

<code>
{code_blob}
</code>

Please generate test cases that check whether an implemented solution
resolves the issue of the user (at the top, within <issue/> brackets).
Present the test cases as a diff (custom format, explained below).

The general format of a diff is as follows.
```custom-diff
diff
<path/filename>
< "rewrite" or "insert" >
< rough line number / EOF / BOF >
< insert function that should be added or rewritten >
end diff
< repeat blocks of diff as necessary >
```

Insertion can only be done at the end or beginning of the file, indicated by EOF or BOF respectively.

As an example for a diff, consider the following two versions of the same file, once before and after a change.
The original version of the file was as follows.
[start of demo/test_file.py]
1 def test_euclidean(a, b):
2     assert euclidean(0, 0) == 0
3     assert euclidean(0, 1) == 1
4     assert euclidean(1, 0) == 1
5     assert euclidean(1, 1) == 1
6
7 @pytest.mark.parametrize("a, b, expected", [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)])
8 def test_gcd(a, b):
9     assert gcd(a, b) == expected
10
[end of demo/file.py]

The diff for fix in function euclidean and adds the function gcd is as follows.
This diff changes the first file into the second file.
```custom-diff
diff
demo/file.py
rewrite
1
def test_euclidean(a, b):
    assert euclidean(0, 0) == 0
    assert euclidean(0, 1) == 1
    assert euclidean(1, 0) == 1
    assert euclidean(1, 1) == 1
    assert euclidean(100, 10) == 10
end diff
diff
demo/file.py
insert
EOF
@ pytest.mark.parametrize("a, b, expected", [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1), (100, 10, 10)])
def test_lcm(a, b):
    assert lcm(a, b) == expected
end diff
```

The new version of the file is as follows.
[start of demo/file.py]
1 def test_euclidean(a, b):
2     assert euclidean(0, 0) == 0
3     assert euclidean(0, 1) == 1
4     assert euclidean(1, 0) == 1
5     assert euclidean(1, 1) == 1
6     assert euclidean(100, 10) == 10
7
8 @pytest.mark.parametrize("a, b, expected", [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)])
9 def test_gcd(a, b):
10     assert gcd(a, b) == expected
11
12 @pytest.mark.parametrize("a, b, expected", [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1), (100, 10, 10)])
13 def test_lcm(a, b):
14     assert lcm(a, b) == expected
15
[end of demo/file.py]

As you can see, you need to indicate the approximate line numbers, function name and the path and file name you want to change,
but there can be as many independent blocks of changes as you need. You may also apply changes to several files.
Apply as much reasoning as you please and see necessary. The format of the solution is fixed and has to follow the custom diff format.
Make sure to implement only test cases and don't try to fix the issue itself.
"""

if not GITHUB_TOKEN:
    raise EnvironmentError("GITHUB_TOKEN not found. Set it before running.")
if not PROMPT_TEMPLATE:
    raise EnvironmentError("PROMPT_TEMPLATE not found. Set it before running.")

auth = Auth.Token(GITHUB_TOKEN)
g = Github(auth=auth)

# --- GitHub API Data Collection ---

def extract_repo_info_from_url(url):
    """Extracts 'owner/repo' from a GitHub URL."""
    match = re.search(r"github\.com/([^/]+/[^/]+)/", url)
    if not match:
        raise ValueError(f"Invalid GitHub URL: {url}")
    return match.group(1)

def get_github_issue_info(issue_url):
    """
    Fetches all required data from the GitHub Issue API.
    """
    repo_full = extract_repo_info_from_url(issue_url)
    repo = g.get_repo(repo_full)

    issue_number_str = issue_url.rstrip('/').split('/')[-1]
    issue_number = int(issue_number_str)
    issue = repo.get_issue(issue_number)

    # 1. instance_id (using '-' as in swe-bench, not '_')
    instance_id = f"{repo_full.replace('/', '-')}-{issue_number}"

    # 2. problem_statement
    title = issue.title.strip() if issue.title else ""
    body = issue.body.strip() if issue.body else ""
    problem_statement = f"{title}\n\n{body}" if body else title

    # 3. hints_text
    comments = issue.get_comments()
    hints = "\n\n".join([f"{c.user.login}: {c.body}" for c in comments])

    return {
        "instance_id": instance_id,
        "repo": repo_full,
        "created_at": issue.created_at.isoformat(),
        "problem_statement": problem_statement,
        "hints_text": hints,
        "repo_clone_url": repo.clone_url,
    }

# --- Git Repository Data Collection ---

def get_git_diff_info(repo_clone_url, token, first_bfc, last_bfc=None):
    """
    Clones a repo to get the base commit and filtered diffs.
    """
    auth_clone_url = repo_clone_url.replace(
        "https://", f"https://oauth2:{token}@"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Repo.clone_from(auth_clone_url, tmpdir)

        try:
            base_commit = repo.commit(first_bfc).parents[0].hexsha
        except IndexError:
            base_commit = repo.git.rev_list('--max-parents=0', first_bfc).strip()

        end_commit = last_bfc or first_bfc

        base_tree = repo.commit(base_commit).tree
        end_tree = repo.commit(end_commit).tree
        diff_index = base_tree.diff(end_tree)

        test_files, code_files = [], []
        for diff_item in diff_index:
            file_path = diff_item.b_path or diff_item.a_path
            if "test" in file_path.lower():
                test_files.append(file_path)
            else:
                code_files.append(file_path)

        patch = repo.git.diff(base_commit, end_commit, "--", *code_files) if code_files else ""
        test_patch = repo.git.diff(base_commit, end_commit, "--", *test_files) if test_files else ""

        return {
            "base_commit": base_commit,
            "patch": patch,
            "test_patch": test_patch,
        }

# --- Orchestrator (Step 1) ---

def collect_swtbench_instance(issue_url, first_bfc, last_bfc):
    """
    Main orchestrator to collect all required data points.
    """
    issue_info = get_github_issue_info(issue_url)
    git_info = get_git_diff_info(
        issue_info["repo_clone_url"],
        GITHUB_TOKEN,
        first_bfc,
        last_bfc
    )
    del issue_info["repo_clone_url"]
    return {**issue_info, **git_info}

# --- Workflow Step 2: Create SWE-bench Dataset ---

def create_swebench_dataset(instance_data: dict, datasets_dir: str) -> str:
    """
    Mimics make_dataset_impacket_1901.py for a given instance.
    Saves to datasets/<instance_id> and returns the path.
    """
    dataset_path = os.path.join(datasets_dir, instance_data["instance_id"])

    # We only need these specific fields for the BM25 script
    row = {
        "instance_id": instance_data["instance_id"],
        "repo": instance_data["repo"],
        "base_commit": instance_data["base_commit"],
        "problem_statement": instance_data["problem_statement"],
    }

    dds = DatasetDict({"dev": Dataset.from_list([row])})

    # Clean up old dir if it exists, then save
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    dds.save_to_disk(dataset_path)

    return dataset_path

# --- Workflow Step 3: Run BM25 Retrieval ---

def run_bm25_retrieval(dataset_path: str, instance_id: str, retrieval_dir: str) -> str:
    """
    Runs bm25_retrieval.py as a subprocess and returns the path to documents.jsonl.
    """

    # --- FIX: Clean up old retrieval results ---
    # The BM25 script skips indexing/retrieval if it finds old results.
    # We must delete them to force a fresh run.
    instance_retrieval_path = os.path.join(retrieval_dir, instance_id)
    if os.path.exists(instance_retrieval_path):
        print(f"    [Cleanup] Removing old retrieval results: {instance_retrieval_path}")
        shutil.rmtree(instance_retrieval_path)
    # --- END FIX ---

    command = [
        'python', 'bm25_retrieval.py',
        '--dataset_name_or_path', dataset_path,
        '--document_encoding_style', 'file_name_and_contents',
        '--output_dir', retrieval_dir,
        '--splits', 'dev'
    ]

    # Run the command and capture output
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    # (Optional diagnostic logging, you can remove this if preferred)
    if result.stdout:
        print(f"    [BM25 STDOUT]: {result.stdout.strip()}")
    if result.stderr:
        print(f"    [BM25 STDERR]: {result.stderr.strip()}")

    # Return the expected path to the results file
    return os.path.join(
        retrieval_dir, instance_id,
        "file_name_and_contents_indexes", instance_id, "documents.jsonl"
    )

# --- Workflow Step 4: Build ZeroShotPlus Prompt (Logic from build_zeroShotPlus_prompt.py) ---

def approx_tokens(s: str) -> int:
    """Very rough token estimator: 1 token ≈ 4 characters."""
    return max(1, (len(s) + 3) // 4)

def iter_docs_jsonl(path: str) -> Iterable[Tuple[str, str]]:
    """Yields (id, contents) from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            yield obj.get("id", "unknown"), obj.get("contents", "")

def make_code_block(file_id: str, contents: str) -> str:
    """Wrap a single file as a block with [start of ...]/[end of ...] markers."""
    body = contents.split("\\n", 1)[1] if contents.startswith(f"{file_id}\\n") else contents
    return f"[start of {file_id}]\\n{body}\\n[end of {file_id}]"

def build_prompt(
    issue_text: str,
    docs: List[Tuple[str, str]],
    template: str,
    token_cap: int = TOKEN_CAP
) -> str:
    """
    Build the final prompt by stuffing code examples into <code>…</code>
    until we hit the token cap.
    """
    tmp_prompt = template.format(issue_text=issue_text, code_blob="{CODE_BLOB_PLACEHOLDER}")
    base_tokens = approx_tokens(tmp_prompt.replace("{CODE_BLOB_PLACEHOLDER}", ""))
    issue_tokens = approx_tokens(issue_text)

    running_code = ""
    running_code_tokens = 0

    for _, (fid, contents) in enumerate(docs):
        block = make_code_block(fid, contents)
        new_code = running_code + (("\\n" if running_code else "") + block)
        new_code_tokens = approx_tokens(new_code)

        if (base_tokens + issue_tokens + new_code_tokens) > token_cap:
            break

        running_code = new_code
        running_code_tokens = new_code_tokens

    return template.format(issue_text=issue_text, code_blob=running_code)

def generate_zeroshot_prompt(
    instance_data: dict,
    docs_file_path: str,
    prompts_dir: str,
    template: str
) -> str:
    """
    Orchestrates reading docs, building the prompt, and saving it.
    Returns the path to the saved prompt.
    """
    if not os.path.exists(docs_file_path):
        raise FileNotFoundError(f"BM25 documents file not found: {docs_file_path}")

    issue_text = instance_data["problem_statement"]
    docs = list(iter_docs_jsonl(docs_file_path))

    # --- DIAGNOSTIC PRINT ---
    if not docs:
        print(f"    [WARNING] No documents found in {docs_file_path}. Prompt will have empty <code> block.")
    # --- END DIAGNOSTIC ---

    prompt = build_prompt(issue_text, docs, template, token_cap=TOKEN_CAP)

    out_path = os.path.join(prompts_dir, f"zero_shot_plus_prompt_{instance_data['instance_id']}.txt")
    os.makedirs(prompts_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    return out_path

# --- Main Execution ---

if __name__ == "__main__":

    input_path = os.path.join(os.getcwd(), INPUT_FILE)
    if not os.path.exists(input_path):
        print(f"FATAL ERROR: Input file not found at {input_path}")
        exit(1)

    with open(input_path, 'r') as f:
        ISSUES = json.load(f)

    all_instances_data = []
    processed_count = 0
    generated_prompts = []

    # === Part 1: Data Collection (Original Script) ===
    for issue in ISSUES:
        issue_url = issue["issue_url"]
        first_bfc = issue["first_bfc"]
        last_bfc = issue.get("last_bfc")

        last_bfc_short = f" to {last_bfc[:7]}..." if last_bfc else ""
        print(f"INFO: Collecting data for {issue_url} (BFC: {first_bfc[:7]}...{last_bfc_short})")

        try:
            data = collect_swtbench_instance(issue_url, first_bfc, last_bfc)
            all_instances_data.append(data)
        except Exception as e:
            print(f"ERROR: Failed to collect data for {issue_url}. {type(e).__name__}: {e}")

    # Write the aggregated JSON data
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_instances_data, f, indent=4)
    print(f"\nSuccessfully collected {len(all_instances_data)} instances to {OUTPUT_FILE}")

    # === Part 2: Workflow for Prompt Generation ===
    print(f"\n--- Starting Prompt Generation Workflow for {len(all_instances_data)} instances ---")
    for data in all_instances_data:
        instance_id = data['instance_id']
        print(f"\nProcessing instance: {instance_id}")

        try:
            # Step 2: Create dataset
            print(f"  [1/4] Creating dataset...")
            dataset_path = create_swebench_dataset(data, DATASETS_DIR)
            print(f"  [1/4] Dataset created at {dataset_path}")

            # Step 3: Run BM25
            print(f"  [2/4] Running BM25 retrieval...")
            docs_file = run_bm25_retrieval(dataset_path, instance_id, RETRIEVAL_DIR)
            print(f"  [2/4] Retrieval results at {docs_file}")

            # Step 4: Build prompt
            print(f"  [3/4] Building prompt...")
            prompt_path = generate_zeroshot_prompt(data, docs_file, PROMPTS_DIR, PROMPT_TEMPLATE)
            print(f"  [3/4] Prompt saved to {prompt_path}")
            generated_prompts.append(prompt_path)

            # Step 5: Cleanup
            print(f"  [4/4] Cleaning up intermediate dataset...")
            shutil.rmtree(dataset_path)
            print(f"  [4/4] Cleanup complete.")

            processed_count += 1

        except FileNotFoundError as e:
            print(f"  ERROR (File Not Found): {e}. Skipping {instance_id}.")
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: BM25 retrieval failed for {instance_id}.")
            print(f"  STDOUT: {e.stdout}")
            print(f"  STDERR: {e.stderr}")
        except Exception as e:
            print(f"  ERROR: Failed workflow for {instance_id}. {type(e).__name__}: {e}")


    print(f"\n--- Workflow Complete ---")
    print(f"Successfully generated {processed_count} prompts in {PROMPTS_DIR}/")
