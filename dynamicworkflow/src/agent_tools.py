"""LangChain tools for the Business Analyst Deep Agent.

Thin I/O closures over the BA services (workflow_context, brd_snapshot,
kb_client, skills). All domain reasoning lives in the skill markdown; these
tools only read/write context and side systems.
"""

from typing import List, Optional

from langchain_core.tools import tool

from services.workflow_context import WorkflowContextManager
from services.session_managers import SessionHistoryManager
from services.brd_snapshot import import_upsert_snapshot, get_snapshot
from services.kb_client import lightrag_query
from services.skills import get_active_skill as _gen_active_skill
from services.agent_memory import remember as _remember_memory, recall as _recall_memory

# Agent identifier used for skill routing (Business Analyst = 1 in this service)
BA_AGENT_ID = 1
BA_SKILL_ID = "ba_brd_generation"


def create_ba_tools(
    workspace_id: str,
    user_id: str,
    conversation_id: str,
    job_id: Optional[str] = None,
    auth_token: Optional[str] = None,
    memory_scope: Optional[str] = None,
    run_location=None,
):
    """Create workspace-scoped tools for the Business Analyst agent.

    Tools are closures capturing workspace context to avoid global state.

    Args:
        workspace_id: Workspace identifier
        user_id: User identifier
        conversation_id: Conversation/session identifier
        job_id: Optional workflow job ID for context curation
        auth_token: Optional bearer token for KB search authentication

    Returns:
        List of LangChain tools
    """
    memory_key = memory_scope or conversation_id

    # Initialize service managers (async-safe; construct DB clients lazily).
    workflow_ctx = WorkflowContextManager()
    session_history = SessionHistoryManager()

    @tool
    async def curate_workflow_context(agent_feed: Optional[List[str]] = None) -> str:
        """Fetch upstream project context (idea brief, requirements, BRD) from the workflow database.

        Args:
            agent_feed: Artifact types to fetch (default: [idea/requirement/BRD])

        Returns:
            Curated project context as formatted string.
        """
        context = await workflow_ctx.curate_context(
            workspace_id=workspace_id,
            job_id=job_id,
            agent_feed=agent_feed,
        )
        return context or "No project context available (standalone mode)"

    @tool
    async def load_conversation_history(limit: int = 10) -> str:
        """Load recent conversation history for this session.

        Args:
            limit: Maximum number of messages to retrieve.

        Returns:
            Formatted conversation history.
        """
        history = await session_history.load_history(
            workspace_id=workspace_id,
            user_id=user_id,
            conversation_id=conversation_id,
            limit=limit,
        )
        if not history:
            return "No previous conversation history"

        formatted = []
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            preview = content[:200] + "..." if len(content) > 200 else content
            formatted.append(f"[{role}]: {preview}")
        return "\n".join(formatted)

    @tool
    async def kb_search(query: str, user_prompt: str = "") -> str:
        """Search the knowledge base (LightRAG/KBCurator) for relevant context.

        Args:
            query: The retrieval question / requirement query.
            user_prompt: Optional extra instruction passed to the KB.

        Returns:
            Retrieved context, or a fallback message when nothing is found.
        """
        namespace = {"workspace_id": workspace_id, "user_id": user_id}
        # Construct Authorization header if auth_token is provided
        auth_header = f"Bearer {auth_token}" if auth_token else ""
        return await lightrag_query(
            prompt=query,
            namespace=namespace,
            user_prompt=user_prompt,
            auth_header=auth_header,
        )

    @tool
    async def save_brd_snapshot(brd_markdown: str) -> str:
        """Persist the generated/updated BRD as the active artifact for this conversation.

        Args:
            brd_markdown: Complete BRD markdown content.
        """

    # Note: lines between save_brd_snapshot and get_active_skill instructions not shown in pictures

        if instructions:
            formatted.append("Instructions:")
            formatted.extend([f"- {inst}" for inst in instructions[:10]])

        return "\n".join(formatted) if formatted else "No active skill found"
    except Exception as e:  # noqa: BLE001
        return f"Error loading skill: {e}"

    @tool
    async def remember_fact(key: str, content: str) -> str:
        """Persist a durable note to long-term memory.

        Use this tool to remember important facts, decisions, or context that should
        be recalled in future conversations. Memory persists across sessions.

        Args:
            key: Short identifier for the memory entry (e.g. "tech_stack", "deadline")
            content: The fact or decision to remember

        Returns:
            Confirmation message
        """
        await _remember_memory(
            workspace_id=workspace_id,
            user_id=user_id,
            conversation_id=memory_key,
            key=key,
            content=content,
        )
        return f"Remembered '{key}'."

    @tool
    async def recall_memory() -> str:
        """Load durable long-term memory notes.

        Retrieves all memory entries stored by remember_fact() for this conversation
        (one at the start of a conversation to restore context from previous sessions).

        Returns:
            Formatted string of all memory entries, or a message if no memory exists
        """
        entries = await _recall_memory(
            workspace_id=workspace_id,
            user_id=user_id,
            conversation_id=memory_key,
        )
        if not entries:
            return "No long-term memory stored yet"
        return "\n".join([f"[{e.get('key')}]: {e.get('content', '')}" for e in entries])

    return [
        get_active_skill,
        curate_workflow_context,
        load_conversation_history,
        kb_search,
        save_brd_snapshot,
        get_previous_brd,
        remember_fact,
        recall_memory,
    ]
