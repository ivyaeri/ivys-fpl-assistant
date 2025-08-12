# config.py
import os
import pytz

TZ = pytz.timezone("Europe/London")
MODEL_NAME = os.getenv("FPL_LLM_MODEL", "gpt-5-mini")

# Persistence path
AUTO_MANAGER_STATE_PATH = os.getenv("AUTO_MANAGER_STATE_PATH", "data/auto_manager_state.json")

# Git auto-commit
GIT_AUTO_COMMIT = os.getenv("GIT_AUTO_COMMIT", "false").lower() == "true"
GIT_COMMIT_MESSAGE = os.getenv("GIT_COMMIT_MESSAGE", "chore: update AI auto-manager state")
GIT_USERNAME = os.getenv("GIT_USERNAME", "")  # optional override
GIT_EMAIL = os.getenv("GIT_EMAIL", "")        # optional override
