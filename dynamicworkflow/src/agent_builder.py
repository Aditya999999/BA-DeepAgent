"""Deep Agent builder for the Business Analyst service.

Mirrors the Architect agent_builder: a module-level LLM singleton, a
workspace-scoped tool set, a skill-driven system prompt, and a storage backend
selected by `CLOUD_STORAGE_PROVIDER`.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI
from azure_claude_chat import AzureClaudeChat
from azure_gpt_chat import AzureGPTChat

# Durable LangGraph checkpointer (graph state per thread_id) backed by MongoDB.
# Optional dependency: if langgraph-checkpoint-mongodb isn't installed we fall
# back to the in-process MemorySaver so the agent still runs.
#
# The async-only "AsyncMongoDBSaver" (langgraph.checkpoint.mongodb.aio) was
# deprecated and removed. The standard "MongoDBSaver" now supports both sync
# and async environments natively (it wraps pymongo, exposing `aget_tuple` etc.
# by running the sync driver in a thread), so we use it for both paths.
try:
    from langgraph.checkpoint.mongodb import MongoDBSaver
except Exception as _ckpt_exc:  # noqa: BLE001
    MongoDBSaver = None
    _checkpoint_import_error = _ckpt_exc

# Azure and S3 backends are rendered under src/ (ported from the forwardingengineering
# reference) since the installed deepagents build ships filesystem only. Imported
# conditionally so a missing optional SDK (azure-storage-blob / aioboto3) degrades
# gracefully to the filesystem backend instead of crashing at startup.
try:
    from deepagents_backends_azure import AzureBlobBackend, AzureBlobConfig
except ImportError:
    try:
        from azure_blob_backend import AzureBlobBackend, AzureBlobConfig
    except Exception as _azure_exc:  # noqa: BLE001
        AzureBlobBackend = None
        AzureBlobConfig = None
        log_import_azure_error = _azure_exc

try:
    from deepagents_backends_aws import S3Backend, S3Config
except ImportError:
    try:
        from aws_s3_backend import S3Backend, S3Config
    except Exception as _s3_exc:  # noqa: BLE001
        S3Backend = None
        S3Config = None
        log_import_s3_error = _s3_exc

from agent_tools import create_ba_tools
from agent_prompts import build_ba_prompt
from agent_memory_templates import BA_SKILLS, MEMORY_FILES
from core.config import settings
from integrations.github_tools import create_github_tools
from integrations.jira_tools import create_jira_tools
from integrations.brd_reader_tool import create_brd_reader_tool
from integrations.mcp_manager import load_mcp_tools_sync

log = logging.getLogger("ba_agent_builder")

# Initialize LLM at module level to prevent middleware interference
_LLM_INSTANCE = None


def _use_legacy_llm() -> bool:
    """Force the non-streaming custom chat models (BA_USE_LEGACY_LLM=1)."""
    return os.getenv("BA_USE_LEGACY_LLM", "").lower() in ("1", "true", "yes")


def _anthropic_max_tokens() -> int:
    """Output token budget for full BRD documents."""
    try:
        return int(os.getenv("BA_MAX_TOKENS", "16000"))
    except ValueError:
        return 16000


def _build_anthropic_llm(selection: Optional[dict] = None):
    """Claude via the Azure AI Foundry Anthropic endpoint."""
    cfg = settings.deep_agent
    selection = selection or {}
    api_key = selection.get("api_key") or cfg.ANTHROPIC_FOUNDRY_API_KEY
    base_url = cfg.ANTHROPIC_FOUNDRY_BASE_URL
    model = selection.get("model") or cfg.ANTHROPIC_DEFAULT_SONNET_MODEL
    max_tokens = _anthropic_max_tokens()

    if not api_key:
        raise ValueError("ANTHROPIC_FOUNDRY_API_KEY not configured")

    if _use_legacy_llm():
        log.info(f"Creating legacy AzureClaudeChat instance (no streaming): {model}")
        return AzureClaudeChat(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
        )

    log.info(f"Creating streaming ChatAnthropic instance: {model}")
    # Azure AI Foundry authenticates via the api-key header; pass it both as
    # api_key (Anthropic SDK) and as an explicit default header so either
    # gateway convention works.
    return ChatAnthropic(
        model=model,
        base_url=base_url,
        api_key=api_key,
        default_headers={"api-key": api_key},
        temperature=0.0,
        max_tokens=max_tokens,
        max_retries=3,
        timeout=300,
        streaming=True,
    )


def _build_openai_llm(selection: Optional[dict] = None):
    """GPT via Azure OpenAI (deployment-scoped endpoint + api-version)."""
    cfg = settings.deep_agent
    selection = selection or {}
    api_key = selection.get("api_key") or cfg.AZURE_OPENAI_API_KEY

    # Note: lines between line 125 and line 172 not shown in pictures

    _LLM_BUILDERS = {
        "openai": _build_openai_llm,
        "anthropic": _build_anthropic_llm,
    }


def get_or_create_llm(runtime_config: Optional[dict] = None):
    """Get or create the global LLM instance for `MODEL_USAGE`."""
    # ...
    # openai = Azure OpenAI (`AzureChatOpenAI`) | anything else -> Claude via
    # the Azure Foundry Anthropic endpoint. Native token streaming is required for
    # `agent.astream(...)` SSE, so the streaming models are the default; the
    # custom AzureClaudeChat / AzureGPTChat models are only used when
    # BA_USE_LEGACY_LLM=1.

    runtime_config = runtime_config or {}
    selection = runtime_config.get("llm") or {}
    model_usage = (selection.get("provider") or settings.deep_agent.MODEL_USAGE or "").strip().lower()
    log.info(f"[Agent Builder] Using request model: {model_usage}/{selection.get('model', '')}")
    builder = _LLM_BUILDERS.get(model_usage, _build_anthropic_llm)
    return builder(selection)


def initialize_llm(temperature: float = 0.0, runtime_config: Optional[dict] = None):
    """Get the global LLM instance."""
    return get_or_create_llm(runtime_config)


def create_ba_agent(
    workspace_id: str,
    user_id: str,
    conversation_id: str,
    agent_id: int = 1,
    job_id: Optional[str] = None,
    extra_tools: Optional[list] = None,
    workspace_dir: Optional[str] = None,
    auth_token: Optional[str] = None,
    agent_profile: Optional[dict] = None,
    run_location=None,
    runtime_config: Optional[dict] = None,
):
    """Create the Business Analyst Deep Agent with workspace context and BA_SKILLS.

    Args:
        workspace_id: Workspace identifier for multi-tenancy
        user_id: User identifier
        conversation_id: Conversation/session identifier
        agent_id: Agent identifier for LLM routing (default: 1)
        job_id: Optional workflow job ID for context curation
        extra_tools: Optional per-session workspace tools (save_output/read_output/...)
        workspace_dir: Optional per-session workspace root for Filesystem backend
        auth_token: Optional bearer token for KB search authentication
        agent_profile: Optional agent registry profile (name, desc from Postgresql)

    Returns:
        Deep Agent instance configured for business analysis
    """
    log.info(
        "Creating Business Analyst Deep Agent",
        extra={
            "workspace_id": workspace_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "agent_id": agent_id,
            "job_id": job_id,
        },
    )

    # Long-term memory partition: shared across a dynamic workflow, or isolated
    # per agent for individual runs (see RunLocation / the blob restructure).
    memory_scope = None
    if run_location is not None:
        if run_location.scope == "dynamic_workflow" and run_location.workflow_job_id:
            memory_scope = f"workflow:{run_location.workflow_job_id}"
        else:
            memory_scope = f"agent:{run_location.agent_folder}"

    # Create workspace-scoped tools
    tools = create_ba_tools(
        workspace_id=workspace_id,
        user_id=user_id,
        conversation_id=conversation_id,
        job_id=job_id,
        auth_token=auth_token,
        memory_scope=memory_scope,
        run_location=run_location,
    )

    # -------------------------------------------------------------------------
    # [PREVIOUS IMPLEMENTATION - Hardcoded Jira & GitHub Tools] (Commented Out)
    # -------------------------------------------------------------------------
    # if workspace_dir:
    #     jira_config = (runtime_config or {}).get("jira") or {}
    #     github_config = (runtime_config or {}).get("github") or {}
    #     if jira_config.get("active"):
    #         tools = list(tools) + create_jira_tools(jira_config)
    #     if github_config.get("active"):
    #         tools = list(tools) + create_github_tools(
    #             workspace_dir,
    #             run_location.agent_folder if run_location else "business_analyst",
    #             run_location,
    #             config=github_config,
    #         )

    # -------------------------------------------------------------------------
    # [NEW IMPLEMENTATION - MCP Integration (Jira & GitHub) + Custom BRD Reader]
    # -------------------------------------------------------------------------
    # 1. Add Custom BRD Reader tool (preserves custom backend / filesystem reading logic)
    if workspace_dir:
        agent_folder = run_location.agent_folder if run_location else "business_analyst"
        brd_reader = create_brd_reader_tool(
            workspace_dir=workspace_dir,
            agent_folder=agent_folder,
            run_location=run_location,
        )
        tools = list(tools) + [brd_reader]

    # 2. Add Dynamic MCP Tools for Jira & GitHub without tool filtering
    if runtime_config:
        mcp_tools = load_mcp_tools_sync(runtime_config)
        if mcp_tools:
            log.info(f"Loaded {len(mcp_tools)} MCP tools (Jira/GitHub)")
            tools = list(tools) + mcp_tools

    # Append any per-session workspace tools (save_output / read_output / list_outputs)
    if extra_tools:
        tools = list(tools) + list(extra_tools)

    # Build system prompt with workspace context
    system_prompt = build_ba_prompt(
        workspace_id=workspace_id,
        conversation_id=conversation_id,
    )

    # Weave registry identity into prompt
    if agent_profile:
        name = agent_profile.get("agent_name")
        desc = agent_profile.get("agent_desc")
        identity_lines = []
        if name:
            identity_lines.append(f"You are operating as: {name}.")
        if desc:
            identity_lines.append(f"Agent description: {desc}")
        if identity_lines:
            system_prompt = system_prompt + "\n\n" + "\n".join(identity_lines)

    if run_location is not None:
        backend = _build_backend(
            workspace_dir=workspace_dir,
            blob_prefix=run_location.blob_prefix,
            db_session_id=run_location.db_session_id,
            doc_meta={
                "workspace_id": run_location.workspace_id,
                "workflow_job_id": run_location.workflow_job_id,
                "scope": run_location.scope,
            },
        )
    else:
        backend = _build_backend(workspace_dir=workspace_dir, workspace_id=workspace_id)

    # Initialize LLM (direct Anthropic integration via Azure Foundry)
    llm = _initialize_llm(temperature=0.0, runtime_config=runtime_config)

    # Build checkpointer (MongoDB if available, else MemorySaver)
    checkpointer = _build_checkpointer()

    # Create Deep Agent
    agent = create_deep_agent(
        model=llm,
        system_prompt=system_prompt,
        backend=backend,
        skills=BA_SKILLS,  # ["/skills/BA_BUSINESS_ANALYST_SKILL.md"]
        memory=MEMORY_FILES,
        tools=tools,
        checkpointer=checkpointer,
    )

    log.info(
        "Business Analyst Deep Agent created successfully",
        extra={
            "workspace_id": workspace_id,
            "skill_count": len(BA_SKILLS),
            "tool_count": len(tools),
        },
    )

    return agent


def _build_backend(
    workspace_dir: Optional[str] = None,
    workspace_id: Optional[str] = None,
    blob_prefix: Optional[str] = None,
    db_session_id: Optional[str] = None,
    doc_meta: Optional[dict] = None,
):
    """Create the storage backend based on CLOUD_STORAGE_PROVIDER setting.

    If `workspace_dir` is given (per-run), the filesystem backend is routed
    there so the agent's file writes land under that run's folder.

    `blob_prefix` explicitly sets the blob/object prefix (the run-scoped
    location -- e.g. `<workspace_id>/dynamic_workflow/<job>`). When omitted it
    degrades to `<workspace_id>/<session_id>` derived from the workspace dir
    basename. `db_session_id` is the key used for document registration (Azure
    config `session_id`); it defaults to the basename. `doc_meta` carries the
    extra columns (workspace_id/workflow_job_id/scope) for the
    `fe_agent_document` registry.
    """
    provider = settings.deep_agent.CLOUD_STORAGE_PROVIDER.lower()
    log.info(f"Initializing backend: {provider}")

    # The session id is the workspace dir basename. The blob/object prefix groups
    # it under the workspace when a workspace_id is available; otherwise it
    # degrades to the session id. An explicit `blob_prefix` wins.
    session_id = Path(workspace_dir).name if workspace_dir else ""
    session_prefix = blob_prefix or (
        f"{workspace_id}/{session_id}" if (workspace_id and session_id) else session_id
    )
    doc_session_id = db_session_id or session_id
    doc_meta = doc_meta or {}

    if provider == "azure":
        if AzureBlobBackend is None or AzureBlobConfig is None:
            log.warning(
                "Azure backend unavailable (%s); falling back to filesystem",
                globals().get("log_import_azure_error", "import failed"),
            )
            provider = "filesystem"
        else:
            conn_str = settings.azure.BLOB_STORAGE_CONNECTION_STRING
            account_name = settings.azure.BLOB_ACCOUNT_NAME
            account_key = settings.azure.BLOB_ACCOUNT_KEY

            # Prefer AZURE_BLOB_STORAGE_CONTAINER_NAME (settings.azure.BLOB_CONTAINER_NAME);
            # WORKSPACE_BLOB_CONTAINER is only an explicit override when set.
            container = (
                settings.deep_agent.WORKSPACE_BLOB_CONTAINER
                or settings.azure.BLOB_CONTAINER_NAME
            )

            kwargs: dict = {
                "container_name": container,
                "prefix": session_prefix,
                "session_id": doc_session_id,
                "workspace_id": doc_meta.get("workspace_id"),
                "workflow_job_id": doc_meta.get("workflow_job_id"),
                "scope": doc_meta.get("scope"),
            }

            if conn_str:
                kwargs["connection_string"] = conn_str
            elif account_name and account_key:
                kwargs["account_url"] = f"https://{account_name}.blob.core.windows.net"
                kwargs["account_key"] = account_key
            else:
                raise ValueError(
                    "Azure provider requires either BLOB_STORAGE_CONNECTION_STRING "
                    "or BLOB_ACCOUNT_NAME + BLOB_ACCOUNT_KEY"
                )

            cfg = AzureBlobConfig(**kwargs)
            log.info(
                f"Using AzureBlobBackend: container={container} prefix={session_prefix}"
            )
            return AzureBlobBackend(cfg)

    if provider in ("s3", "aws"):
        if S3Backend is None or S3Config is None:
            log.warning(
                "S3 backend unavailable (%s); falling back to filesystem. "
                "Install 'aioboto3' to enable it.",
                globals().get("log_import_s3_error", "import failed"),
            )
            provider = "filesystem"
        else:
            bucket = settings.deep_agent.AWS_S3_BUCKET_NAME

            if not bucket:
                raise ValueError("AWS_S3_BUCKET_NAME is required for s3 provider")

            cfg = S3Config(
                bucket=bucket,
                prefix=session_prefix,
                access_key_id=settings.deep_agent.AWS_ACCESS_KEY_ID,
                secret_access_key=settings.deep_agent.AWS_SECRET_ACCESS_KEY,
                region=settings.deep_agent.AWS_DEFAULT_REGION,
            )

            log.info(f"Using S3Backend: bucket={bucket} prefix={session_prefix}")
            return S3Backend(cfg)

    # Filesystem (default)
    if workspace_dir:
        workspace_root = Path(workspace_dir)
    else:
        workspace_root = Path(settings.deep_agent.WORKSPACE_ROOT)
    workspace_root.mkdir(parents=True, exist_ok=True)

    log.info(f"Using FilesystemBackend: root={workspace_root}")
    return FilesystemBackend(
        root_dir=str(workspace_root),
        virtual_mode=True,
    )


# Cached pymongo client dedicated to the checkpointer. `MongoDBSaver` wraps the
# synchronous pymongo driver (running it in a thread for its async methods), so it
# needs a real `pymongo.MongoClient` rather than the Motor async client used
# elsewhere. Cached at module level so we open at most one extra connection pool.
_checkpoint_mongo_client = None


def _get_checkpoint_mongo_client():
    """Lazily create (and cache) the pymongo client for the checkpointer."""
    global _checkpoint_mongo_client
    if _checkpoint_mongo_client is not None:
        return _checkpoint_mongo_client

    uri = settings.database.MONGODB_DATABASE_URI
    if not uri:
        return None
    from pymongo import MongoClient

    _checkpoint_mongo_client = MongoClient(uri)
    return _checkpoint_mongo_client


def _build_checkpointer():
    """Return a durable MongoDB checkpointer, or MemorySaver as a fallback.

    Stores graph state in the configured ``ba_checkpoints`` /
    ``ba_checkpoint_writes`` collections via the standard ``MongoDBSaver``
    (async-capable). Falls back to the in-process ``MemorySaver`` when the
    optional ``langgraph-checkpoint-mongodb`` dependency is missing or no Mongo
    URI is configured.
    """
    if MongoDBSaver is None:
        log.warning(
            "langgraph-checkpoint-mongodb unavailable (%s); using in-memory "
            "MemorySaver (checkpoints will not survive restart)",
            globals().get("_checkpoint_import_error", "import failed"),
        )
        return MemorySaver()

    client = _get_checkpoint_mongo_client()
    if client is None:
        log.warning(
            "MongoDB URI not configured; using in-memory MemorySaver for "
            "checkpoints this run"
        )
        return MemorySaver()

    try:
        checkpointer = MongoDBSaver(
            client,
            db_name=settings.database.BA_DB_NAME,
            checkpoint_collection_name=settings.database.BA_CHECKPOINTS_COLLECTION,
            writes_collection_name=settings.database.BA_CHECKPOINT_WRITES_COLLECTION,
        )
        log.info(
            f"Using MongoDBSaver: db={settings.database.BA_DB_NAME}, "
            f"collections=({settings.database.BA_CHECKPOINTS_COLLECTION}/"
            f"{settings.database.BA_CHECKPOINT_WRITES_COLLECTION})"
        )
        return checkpointer
    except Exception as exc:  # noqa: BLE001
        log.warning(f"Failed to build MongoDB checkpointer: {exc}; using MemorySaver")
        return MemorySaver()
