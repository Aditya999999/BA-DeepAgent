"""MCP (Model Context Protocol) Manager for BA_Agent.

Provides dynamic, per-session integration with external MCP servers
(Jira and GitHub) using `langchain-mcp-adapters` without tool filtering,
following SOLID principles.
"""

import asyncio
import concurrent.futures
import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool

logger = logging.getLogger("ba_agent.mcp_manager")

# Gracefully import langchain_mcp_adapters
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MultiServerMCPClient = None
    MCP_AVAILABLE = False


def build_mcp_server_configs(runtime_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Construct user-scoped MCP server configurations from runtime_config.

    Single Responsibility: Isolates MCP server launch parameters for Jira and GitHub.
    Open/Closed: Easily extendable to add other MCP servers (Confluence, Slack, etc.).
    """
    jira_cfg = runtime_config.get("jira") or {}
    github_cfg = runtime_config.get("github") or {}
    server_configs: Dict[str, Dict[str, Any]] = {}

    # 1. Jira MCP Server Configuration (stdio transport using user credentials)
    if jira_cfg.get("active"):
        jira_url = str(
            jira_cfg.get("jira_url")
            or jira_cfg.get("project_url")
            or os.getenv("JIRA_PROJECT_URL", "")
        ).strip()
        jira_domain = jira_url.replace("https://", "").replace("http://", "").rstrip("/")
        jira_email = str(
            jira_cfg.get("jira_email")
            or jira_cfg.get("email")
            or os.getenv("JIRA_EMAIL", "")
        ).strip() or os.getenv("JIRA_USERNAME", "")
        jira_token = str(
            jira_cfg.get("jira_access_token")
            or jira_cfg.get("jira_api_token")
            or os.getenv("JIRA_API_TOKEN", "")
        ).strip()

        if jira_domain and jira_email and jira_token:
            server_configs["jira"] = {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-jira"],
                "transport": "stdio",
                "env": {
                    "JIRA_DOMAIN": jira_domain,
                    "JIRA_EMAIL": jira_email,
                    "JIRA_API_TOKEN": jira_token,
                },
            }
        else:
            logger.warning(
                "[MCP] Jira active flag is true, but required credentials (domain/email/token) are incomplete."
            )

    # 2. GitHub MCP Server Configuration (stdio transport using user credentials)
    if github_cfg.get("active"):
        github_token = str(
            github_cfg.get("github_token") or os.getenv("GITHUB_TOKEN", "")
        ).strip()
        if github_token:
            server_configs["github"] = {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "transport": "stdio",
                "env": {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": github_token,
                },
            }
        else:
            logger.warning(
                "[MCP] GitHub active flag is true, but GITHUB_TOKEN is missing."
            )

    return server_configs


async def load_mcp_tools_async(runtime_config: Dict[str, Any]) -> List[BaseTool]:
    """Asynchronously connect to MCP servers and return all discovered tools without filtering."""
    if not MCP_AVAILABLE:
        logger.warning(
            "[MCP] langchain-mcp-adapters is not installed. "
            "Please install via `pip install langchain-mcp-adapters mcp`."
        )
        return []

    server_configs = build_mcp_server_configs(runtime_config)
    if not server_configs:
        return []

    try:
        async with MultiServerMCPClient(server_configs) as client:
            # Per user requirement: No tool filtering applied, all tools returned directly
            tools = await client.get_tools()
            logger.info(
                f"[MCP] Successfully loaded {len(tools)} tools from MCP servers: {list(server_configs.keys())}"
            )
            return list(tools)
    except Exception as exc:
        logger.error(f"[MCP] Failed to load MCP tools: {exc}", exc_info=True)
        return []


def load_mcp_tools_sync(runtime_config: Dict[str, Any]) -> List[BaseTool]:
    """Synchronous adapter to load MCP tools for synchronous agent builders."""
    if not MCP_AVAILABLE:
        return []

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    try:
        if loop and loop.is_running():
            # If running inside an existing asyncio event loop (e.g., FastAPI / Uvicorn worker thread),
            # dispatch to a separate worker thread to avoid "This event loop is already running".
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, load_mcp_tools_async(runtime_config))
                return future.result(timeout=60)
        else:
            return asyncio.run(load_mcp_tools_async(runtime_config))
    except Exception as exc:
        logger.error(f"[MCP] Error in load_mcp_tools_sync: {exc}", exc_info=True)
        return []
