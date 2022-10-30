"""

    Git Utils

"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import git


def git_metadata(path: Optional[str | Path] = None) -> Optional[dict[str, str]]:
    if path and not Path(path).exists():
        return None
    repo = git.Repo(path, search_parent_directories=True)
    return dict(
        sha_short=repo.head.object.hexsha[:8],
        sha_long=repo.head.object.hexsha,
        branch=repo.active_branch.name,
    )
