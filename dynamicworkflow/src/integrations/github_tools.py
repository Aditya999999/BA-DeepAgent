"""GitHub REST tools for Business Requirements Documents."""

import base64
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

_API = "https://api.github.com"


def _headers(config: dict | None = None) -> dict:
    config = config or {}
    token = str(config.get("github_token") or os.getenv("GITHUB_TOKEN", "")).strip()
    if not token:
        raise ValueError("GITHUB_TOKEN must be configured")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _repo(repo_url: str = "", config: dict | None = None) -> tuple[str, str]:
    config = config or {}
    repo_name = config.get("github_repo_full_name")
    url = str(repo_url or repo_name or os.getenv("GITHUB_REPO_URL", "")).strip().rstrip("/")
    if repo_name and "/" in url and "github.com" not in url:
        return tuple(url.split("/", 1))  # type: ignore
    url = re.sub(r"\.git$", "", url)
    match = re.search(r"github\.com/([^/]+)/([^/]+)$", url)
    if not match:
        raise ValueError(
            "GITHUB_REPO_URL must be a GitHub URL such as https://github.com/org/repo"
        )
    return match.group(1), match.group(2)


def _branch(branch: str = "", config: dict | None = None) -> str:
    config = config or {}
    return branch.strip() or str(config.get("github_branch") or os.getenv("GITHUB_DEFAULT_BRANCH", "main")).strip() or "main"


def _file_sha(owner: str, repo: str, path: str, branch: str, headers: dict) -> str:
    with httpx.Client(timeout=20.0) as client:
        response = client.get(
            f"{_API}/repos/{owner}/{repo}/contents/{path}",
            headers=headers,
            params={"ref": branch},
        )
        return response.json().get("sha", "") if response.status_code == 200 else ""


def _download_file(owner: str, repo: str, path: str, branch: str, headers: dict) -> bytes:
    with httpx.Client(timeout=20.0) as client:
        response = client.get(
            f"{_API}/repos/{owner}/{repo}/contents/{path}",
            headers=headers,
            params={"ref": branch},
        )
        response.raise_for_status()
        data = response.json()
        if data.get("encoding") != "base64":
            raise ValueError(f"Github returned unsupported content encoding for {path}")
        return base64.b64decode(data.get("content", ""))


def create_github_tools(
    workspace_dir: str,
    agent_folder: str = "business_analyst",
    run_location=None,
    config: dict | None = None,
):
    """Create GitHub tools for the BRD stored in this agent workspace."""

    backend = None
    if run_location is not None:
        from services.workspace_tools import backend_for_workspace

        backend = backend_for_workspace(workspace_dir, run_location)

    def _brd_path() -> Path:
        return Path(workspace_dir) / "workspace" / agent_folder / "BRD.md"

    def read_brd() -> bytes | None:
        if backend is not None:
            from services.workspace_tools import read_via_backend

            content = read_via_backend(backend, f"workspace/{agent_folder}/BRD.md")
            if content is not None:
                return content.encode("utf-8")
        path = _brd_path()
        return path.read_bytes() if path.exists() else None

    def write_brd(content: bytes) -> None:
        if backend is not None:
            from services.workspace_tools import write_via_backend

            decoded = content.decode("utf-8")
            error = write_via_backend(
                backend, f"workspace/{agent_folder}/BRD.md", decoded
            )
            if error:
                raise RuntimeError(error)
            return
        path = _brd_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    @tool
    def test_github_connection() -> str:
        """Test the configured GitHub token and return the authenticated login."""
        try:
            with httpx.Client(timeout=20.0) as client:
                response = client.get(f"{_API}/user", headers=_headers(config))
                response.raise_for_status()
                return f"Connected to GitHub as {response.json().get('login', '?')}"
        except Exception as exc:  # noqa: BLE001
            return f"GitHub connection failed: {exc}"

    @tool
    def get_github_repos() -> str:
        """List repositories accessible with the configured GitHub token."""
        try:
            with httpx.Client(timeout=20.0) as client:
                response = client.get(
                    f"{_API}/user/repos",
                    headers=_headers(config),
                    params={"per_page": 30, "sort": "updated"},
                )
                response.raise_for_status()
                return "\n".join(
                    f"{repo.get('full_name', '?')} ({repo.get('default_branch', 'main')})"
                    for repo in response.json()
                ) or "No GitHub repositories found"
        except Exception as exc:  # noqa: BLE001
            return f"GitHub repository lookup failed: {exc}"

    @tool
    def get_github_branches(repo_url: str = "") -> str:
        """List branches available in the configured GitHub repository."""
        try:
            owner, repo = _repo(repo_url, config)
            with httpx.Client(timeout=20.0) as client:
                response = client.get(
                    f"{_API}/repos/{owner}/{repo}/branches",
                    headers=_headers(config),
                    params={"per_page": 100},
                )
                response.raise_for_status()
                return "\n".join(branch.get("name", "?") for branch in response.json()) or "No branches found"
        except Exception as exc:  # noqa: BLE001
            return f"GitHub branch lookup failed: {exc}"

    @tool
    def push_brd_to_github(
        branch: str = "",
        commit_message: str = "Add generated BRD",
        repo_url: str = "",
    ) -> str:
        """Push this conversation's BRD.md to docs/BRD.md in GitHub."""
        try:
            owner, repo = _repo(repo_url, config)
            target = _branch(branch, config)
            content = read_brd()
            if content is None:
                return "BRD.md was not found in this conversation workspace"
            headers = _headers(config)
            remote = "docs/BRD.md"
            payload = {
                "message": commit_message,
                "content": base64.b64encode(content).decode(),
                "branch": target,
            }
            sha = _file_sha(owner, repo, remote, target, headers)
            if sha:
                payload["sha"] = sha
            with httpx.Client(timeout=30.0) as client:
                response = client.put(
                    f"{_API}/repos/{owner}/{repo}/contents/{remote}",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                return f"Pushed BRD.md to {owner}/{repo} on {target}"
        except Exception as exc:  # noqa: BLE001
            return f"GitHub BRD push failed: {exc}"

    @tool
    def pull_brd_from_github(branch: str = "", repo_url: str = "") -> str:
        """Pull docs/BRD.md from GitHub into this conversation workspace."""
        try:
            owner, repo = _repo(repo_url, config)
            target = _branch(branch, config)
            content = _download_file(owner, repo, "docs/BRD.md", target, _headers(config))
            write_brd(content)
            return f"Pulled BRD.md into this conversation workspace from {owner}/{repo} on {target}"
        except Exception as exc:  # noqa: BLE001
            return f"GitHub BRD pull failed: {exc}"

    @tool
    def create_github_pr(
        title: str,
        body: str = "",
        head_branch: str = "",
        base_branch: str = "",
        repo_url: str = "",
    ) -> str:
        """Create a GitHub pull request from head_branch into base_branch."""
        try:
            owner, repo = _repo(repo_url, config)
            payload = {
                "title": title,
                "body": body,
                "head": _branch(head_branch, config),
                "base": base_branch.strip() or "main",
            }
            with httpx.Client(timeout=20.0) as client:
                response = client.post(
                    f"{_API}/repos/{owner}/{repo}/pulls",
                    json=payload,
                    headers=_headers(config),
                )
                response.raise_for_status()
                data = response.json()
                return f"Created PR #{data.get('number', '?')}: {data.get('html_url', '?')}"
        except Exception as exc:  # noqa: BLE001
            return f"GitHub PR creation failed: {exc}"

    return [
        test_github_connection,
        get_github_repos,
        get_github_branches,
        push_brd_to_github,
        pull_brd_from_github,
        create_github_pr,
    ]
