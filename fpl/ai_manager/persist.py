# fpl/ai_manager/persist.py
import os, json
from pathlib import Path
from git import Repo, InvalidGitRepositoryError, NoSuchPathError
from config import AUTO_MANAGER_STATE_PATH, GIT_AUTO_COMMIT, GIT_COMMIT_MESSAGE, GIT_USERNAME, GIT_EMAIL

STATE_PATH = Path(AUTO_MANAGER_STATE_PATH)

def _ensure_parent():
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_manager_state():
    try:
        if STATE_PATH.exists():
            with open(STATE_PATH, "r") as f:
                return json.load(f)
    except Exception:
        return None
    return None

def save_manager_state(state: dict):
    try:
        _ensure_parent()
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass

def maybe_git_commit_and_push():
    """Auto-commit & push the state file if repo present and env allows."""
    if not GIT_AUTO_COMMIT:
        return
    try:
        repo = Repo(Path.cwd())
    except (InvalidGitRepositoryError, NoSuchPathError):
        return

    # Optional user config (useful on servers)
    if GIT_USERNAME:
        repo.git.config("user.name", GIT_USERNAME)
    if GIT_EMAIL:
        repo.git.config("user.email", GIT_EMAIL)

    rel_path = os.path.relpath(STATE_PATH, repo.working_tree_dir)
    if not os.path.exists(STATE_PATH):
        return

    try:
        repo.git.add(rel_path)
        # Only commit if there's something to commit
        if repo.is_dirty(index=True, working_tree=True, untracked_files=True):
            repo.index.commit(GIT_COMMIT_MESSAGE)
            # Push default remote (requires write access via SSH or PAT)
            repo.remotes.origin.push()
    except Exception:
        # fail silently to avoid interrupting app
        pass
