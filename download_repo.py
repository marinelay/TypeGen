import git
import json
from pathlib import Path
import shutil

def download_repo():
    manytypes4py = Path("/home/wonseokoh/ManyTypes4PyDataset-v0.7")

    with open("data/testset.json", "r") as f:
        testset = json.load(f)

    scope_set = set()
    repo_set = set()

    for test in testset:
        scope_set.add(test["scope"])
        file_name = test["file"]
        repo_name = file_name.split("/")[1:3]
        repo_name = "/".join(repo_name)
        repo_set.add(repo_name)

    with open(manytypes4py / "ManyTypes4PyDataset.spec", "r") as f:
        spec = f.readlines()

    repos_dir = Path("repos")

    if not repos_dir.exists():
        repos_dir.mkdir()

    for repo in repo_set:
        if (repos_dir / repo).exists():
            print(f"{repo} already exists, skipping")
            continue

        print("Downloading", repo)
        github_url = f"https://github.com/{repo}.git"

        commit_id = None
        for line in spec:
            if github_url in line:
                commit_id = line.split()[1]

        repo_dir = repos_dir / repo

        try:
            repo = git.Repo.clone_from(github_url, repo_dir)
            repo.git.checkout(commit_id)
        except git.exc.GitCommandError:
            print(f"Failed to download {repo}, skipping")
            if repo_dir.exists():
                shutil.rmtree(repo_dir)

            continue

        

        print(f"Checked out commit {commit_id} in {repo_dir}")

    

if __name__ == "__main__":
    download_repo()