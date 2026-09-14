"""Custom BRD Reading Tool for Business Analyst Deep Agent.

Preserves custom workspace / storage backend reading methods for BRD.md
while allowing the agent to deliver the content to modern MCP tools
(e.g., GitHub MCP `create_or_update_file`).
"""

from pathlib import Path
from typing import Any, Optional
from langchain_core.tools import tool


def create_brd_reader_tool(
    workspace_dir: str,
    agent_folder: str = "business_analyst",
    run_location: Optional[Any] = None,
):
    """Create a workspace-aware tool to read the current BRD.md content.

    Follows the custom backend reading logic:
    1. Checks if a persistent backend (Azure Blob / S3) is active via `backend_for_workspace`
    2. Falls back to reading from the local workspace directory
    """
    backend = None
    if run_location is not None:
        try:
            from services.workspace_tools import backend_for_workspace

            backend = backend_for_workspace(workspace_dir, run_location)
        except Exception:
            backend = None

    def _brd_path() -> Path:
        return Path(workspace_dir) / "workspace" / agent_folder / "BRD.md"

    @tool
    def read_current_brd() -> str:
        """Read the generated BRD.md content from the workspace storage.

        Use this custom tool to retrieve the complete Business Requirements Document
        content from the workspace before pushing it to GitHub via the GitHub MCP
        `create_or_update_file` tool.

        Returns:
            The complete markdown string of BRD.md, or an error message if not found.
        """
        # 1. Custom read via cloud backend (Azure Blob / S3) if configured
        if backend is not None:
            try:
                from services.workspace_tools import read_via_backend

                content = read_via_backend(backend, f"workspace/{agent_folder}/BRD.md")
                if content:
                    return content
            except Exception:
                pass

        # 2. Custom read via local filesystem
        path = _brd_path()
        if path.exists():
            return path.read_text(encoding="utf-8")

        return "BRD.md was not found in this conversation workspace."

    return read_current_brd
