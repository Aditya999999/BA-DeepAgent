"""Read-only Jira REST tools for Business Requirements Documents."""

import base64
import os

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()


def _headers(config: dict | None = None) -> dict:
    config = config or {}
    email = str(config.get("jira_email") or config.get("email") or os.getenv("JIRA_EMAIL", "")).strip() or os.getenv("JIRA_USERNAME", "")
    token = str(config.get("jira_access_token") or config.get("jira_api_token") or os.getenv("JIRA_API_TOKEN", "")).strip()
    if not email or not token:
        raise ValueError("Both JIRA_EMAIL (or JIRA_USERNAME) and JIRA_API_TOKEN must be configured")
    encoded = base64.b64encode(f"{email}:{token}".encode()).decode()
    return {
        "Authorization": f"Basic {encoded}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _base_url(config: dict | None = None) -> str:
    config = config or {}
    url = str(config.get("jira_url") or config.get("project_url") or os.getenv("JIRA_PROJECT_URL", "")).strip()
    url = url.rstrip("/")
    if not url:
        raise ValueError("JIRA_PROJECT_URL must be configured")
    return url


def create_jira_tools(config: dict | None = None):
    """Create read-only Jira discovery tools for the BRD agent."""

    @tool
    def test_jira_connection() -> str:
        """Test Jira credentials and return the authenticated account name."""
        try:
            with httpx.Client(timeout=20.0) as client:
                response = client.get(f"{_base_url(config)}/rest/api/3/myself", headers=_headers(config))
                response.raise_for_status()
                data = response.json()
                return f"Connected to Jira as {data.get('displayName') or data.get('emailAddress', '?')}"
        except Exception as exc:  # noqa: BLE001
            return f"Jira connection failed: {exc}"

    @tool
    def get_jira_projects() -> str:
        """List Jira projects accessible with the configured credentials."""
        try:
            with httpx.Client(timeout=20.0) as client:
                response = client.get(f"{_base_url(config)}/rest/api/3/project", headers=_headers(config))
                response.raise_for_status()
                projects = response.json()
                return "\n".join(
                    f"{project.get('key', '?')}: {project.get('name', '?')}"
                    for project in projects
                ) or "No Jira projects found"
        except Exception as exc:  # noqa: BLE001
            return f"Jira project lookup failed: {exc}"

    @tool
    def get_jira_issues(project_key: str = "", jql: str = "", max_results: int = 50) -> str:
        """List Jira issues from a project or a read-only JQL query for BRD context."""
        key = project_key.strip() or str(config.get("jira_project_key") or os.getenv("JIRA_PROJECT_KEY", "")).strip()
        query = jql.strip() or (
            f'project = "{key}" ORDER BY created DESC' if key else "ORDER BY created DESC"
        )
        try:
            with httpx.Client(timeout=20.0) as client:
                response = client.get(
                    f"{_base_url(config)}/rest/api/3/search/jql",
                    headers=_headers(config),
                    params={
                        "jql": query,
                        "maxResults": min(max(max_results, 1), 100),
                        "fields": "summary,description,status,priority",
                    },
                )
                response.raise_for_status()
                issues = response.json().get("issues", [])
                if not issues:
                    return "No Jira issues found"
                return "\n".join(
                    f"[{issue.get('key', '?')}] {(issue.get('fields', {})).get('summary', '?')}"
                    for issue in issues
                )
        except Exception as exc:  # noqa: BLE001
            return f"Jira issue lookup failed: {exc}"

    return [test_jira_connection, get_jira_projects, get_jira_issues]
