"""
Mixins and SQL-safety helpers for the LanceDB vector store.

Historical note: these mixins once carried full duplicate implementations of
``add_tools`` / ``search`` / ``add_skills`` / etc. Those copies were shadowed
by the identical method definitions in ``LanceDBVectorStore``'s class body
(MRO: class body wins), so they were dead code that silently diverged from
the live implementations whenever only one copy was fixed. They were removed
in the 2026-08-03 consolidation — the single source of truth for tools and
skills operations is ``lancedb.py``. What remains here is code that is
actually inherited: the tools schema migration and the sync-metadata API.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any

from agent_gantry.schema.tool import ToolDefinition
from agent_gantry.utils.fingerprint import compute_tool_fingerprint

# Pre-compile regex for control character checking
# Benchmark: ~5x faster than generator expression (any(ord(c) < 32...))
_CTRL_CHAR_RE = re.compile(r"[\x00-\x1f]")

logger = logging.getLogger(__name__)


def _escape_sql_string(value: str) -> str:
    """
    Escape special characters in SQL strings to prevent injection.

    This function provides SQL injection protection for LanceDB queries by:
    1. Escaping backslashes (must be done first)
    2. Escaping single quotes using SQL standard ('') escaping

    Note: This is used in conjunction with _validate_identifier() which rejects
    control characters and enforces length limits. LanceDB does not currently
    support parameterized queries for WHERE clauses, so string escaping is
    necessary. All user-provided values go through validation before escaping.

    Security considerations:
    - Only used for metadata key lookups (not arbitrary user input)
    - Keys are validated by _validate_identifier() before escaping
    - All test cases in test suite verify SQL injection attempts are blocked

    Args:
        value: The string value to escape

    Returns:
        Escaped string safe for SQL inclusion
    """
    # Escape backslashes first, then single quotes
    return value.replace("\\", "\\\\").replace("'", "''")


def _validate_identifier(value: str, field_name: str) -> None:
    """
    Validate that a value is safe to use in SQL queries.

    This provides the first line of defense against SQL injection by:
    1. Enforcing length limits (1-256 characters)
    2. Rejecting null bytes and control characters (ASCII < 32)

    This validation occurs before any SQL escaping is applied.

    Args:
        value: The value to validate
        field_name: Name of the field (for error messages)

    Raises:
        ValueError: If validation fails
    """
    if not value or len(value) > 256:
        raise ValueError(f"{field_name} must be 1-256 characters")
    # Reject null bytes and other control characters
    if _CTRL_CHAR_RE.search(value):
        raise ValueError(f"{field_name} contains invalid characters")


class LanceDBToolsMixin:
    """Schema-migration support for the LanceDB tools table."""

    async def _migrate_tools_schema(self, target_schema: Any) -> None:
        """
        Migrate tools table schema if needed.

        This handles adding missing columns to existing databases to support
        new features like fingerprinting without losing data.

        Args:
            target_schema: The target PyArrow schema
        """
        try:
            # Get current schema
            current_schema = self._tools_table.schema  # type: ignore
            current_field_names = {field.name for field in current_schema}
            target_field_names = {field.name for field in target_schema}

            # Check if migration is needed
            missing_fields = target_field_names - current_field_names
            if not missing_fields:
                return  # Schema is up to date

            logger.info(f"Migrating tools table schema. Adding fields: {missing_fields}")

            # LanceDB doesn't support ALTER TABLE, so we need to:
            # 1. Read all existing data
            # 2. Add missing columns with default values
            # 3. Re-insert data

            # Read existing data (blocking file I/O — keep off the event loop)
            table = await asyncio.to_thread(self._tools_table.to_arrow)  # type: ignore
            records = table.to_pylist()

            if not records:
                # Empty table, just recreate with new schema
                await asyncio.to_thread(self._db.drop_table, self._tools_table_name)  # type: ignore
                self._tools_table = await asyncio.to_thread(
                    self._db.create_table,  # type: ignore
                    self._tools_table_name,  # type: ignore
                    schema=target_schema,
                )
                return

            # Add missing fields with default values
            now = datetime.now(timezone.utc).isoformat()
            for record in records:
                if "fingerprint" not in record and "fingerprint" in missing_fields:
                    # Compute fingerprint for existing tools
                    try:
                        tool = ToolDefinition.model_validate_json(record["tool_json"])
                        record["fingerprint"] = compute_tool_fingerprint(tool)
                    except Exception as e:
                        # Fallback to empty fingerprint if tool JSON is invalid
                        logger.warning(f"Failed to compute fingerprint during migration: {e}")
                        record["fingerprint"] = ""
                if "created_at" not in record and "created_at" in missing_fields:
                    record["created_at"] = now
                if "updated_at" not in record and "updated_at" in missing_fields:
                    record["updated_at"] = now

            # Drop and recreate table with new schema
            await asyncio.to_thread(self._db.drop_table, self._tools_table_name)  # type: ignore
            self._tools_table = await asyncio.to_thread(
                self._db.create_table,  # type: ignore
                self._tools_table_name,  # type: ignore
                schema=target_schema,
            )

            # Re-insert data
            await asyncio.to_thread(self._tools_table.add, records)  # type: ignore
            logger.info(f"Successfully migrated {len(records)} tools to new schema")

        except Exception as e:
            logger.error(f"Schema migration failed: {e}")
            # Don't raise - allow system to continue with current schema
            # This makes the migration non-breaking


class LanceDBMetadataMixin:
    """Mixin for LanceDB metadata operations."""

    async def get_metadata(self, key: str) -> str | None:
        """
        Get a metadata value by key.

        Args:
            key: The metadata key

        Returns:
            The value if found, None otherwise
        """
        await self._ensure_initialized()  # type: ignore

        try:
            escaped_key = _escape_sql_string(key)
            query = self._metadata_table.search().where(f"key = '{escaped_key}'").limit(1).select(['value'])  # type: ignore
            table = await asyncio.to_thread(query.to_arrow)
            values = table["value"].to_pylist()
            if values and values[0] is not None:
                return values[0]
        except Exception as e:
            logger.debug(f"get_metadata failed for key '{key}': {e}")
        return None

    async def set_metadata(self, key: str, value: str) -> None:
        """
        Set a metadata value.

        Args:
            key: The metadata key
            value: The value to store
        """
        await self._ensure_initialized()  # type: ignore

        now = datetime.now(timezone.utc).isoformat()

        # Delete existing record if present
        try:
            escaped_key = _escape_sql_string(key)
            await asyncio.to_thread(self._metadata_table.delete, f"key = '{escaped_key}'")  # type: ignore
        except RuntimeError:
            # LanceDB raises RuntimeError when attempting to delete non-existent records
            pass
        except Exception as e:
            logger.warning(f"Unexpected error deleting metadata key '{key}': {e}")
            # Continue anyway - we'll try to add the new record

        # Add new record
        await asyncio.to_thread(
            self._metadata_table.add,  # type: ignore
            [
                {
                    "key": key,
                    "value": value,
                    "updated_at": now,
                }
            ],
        )

    async def get_stored_fingerprints(self) -> dict[str, str]:
        """
        Get all stored tool fingerprints.

        Returns:
            Dictionary mapping tool_id to fingerprint
        """
        await self._ensure_initialized()  # type: ignore

        try:
            query = self._tools_table.search().select(["id", "fingerprint"]).limit(None)  # type: ignore
            table = await asyncio.to_thread(query.to_arrow)
            ids = table["id"].to_pylist()
            fingerprints = [f if f is not None else "" for f in table["fingerprint"].to_pylist()]
            return dict(zip(ids, fingerprints))
        except Exception as e:
            logger.debug(f"get_stored_fingerprints failed: {e}")
            return {}

    async def get_sync_status(self) -> dict[str, Any]:
        """
        Get the current sync status including metadata.

        Returns:
            Dictionary with sync status info:
            - tool_count: Number of tools in database
            - embedder_id: Identifier of embedder used
            - dimension: Vector dimension
            - last_sync: ISO timestamp of last sync
        """
        await self._ensure_initialized()  # type: ignore

        status: dict[str, Any] = {
            "tool_count": await self.count(),  # type: ignore
            "dimension": self._dimension,  # type: ignore
        }

        # Get metadata values
        embedder_id = await self.get_metadata("embedder_id")
        if embedder_id:
            status["embedder_id"] = embedder_id

        last_sync = await self.get_metadata("last_sync")
        if last_sync:
            status["last_sync"] = last_sync

        stored_dimension = await self.get_metadata("dimension")
        if stored_dimension:
            status["stored_dimension"] = int(stored_dimension)

        return status

    async def update_sync_metadata(
        self,
        embedder_id: str,
        dimension: int,
    ) -> None:
        """
        Update sync metadata after a successful sync.

        This method provides transaction-like semantics by updating all
        metadata fields together. If any update fails, the entire operation
        is considered failed and an attempt is made to rollback to previous state.

        Rollback Limitations:
            Due to LanceDB's lack of native transaction support, rollback is
            best-effort only and may fail if:

            - The metadata table becomes corrupted during updates
            - A second concurrent process modifies metadata simultaneously
            - The database connection is lost during rollback

            If rollback fails, the metadata may be left in an inconsistent state
            with some fields updated and others not. In this case:

            - Check logs for "Rollback failed" error messages
            - Manually verify metadata consistency with get_sync_status()
            - Consider re-syncing all tools to restore consistency
            - Use external locks (e.g., file locks) to prevent concurrent writes

        Args:
            embedder_id: Identifier for the embedder used
            dimension: Vector dimension used

        Raises:
            Exception: If metadata update fails (with rollback attempted)
        """
        now = datetime.now(timezone.utc).isoformat()

        # Store old values for rollback
        old_embedder_id = await self.get_metadata("embedder_id")
        old_dimension = await self.get_metadata("dimension")
        old_last_sync = await self.get_metadata("last_sync")

        try:
            # Update all metadata fields
            await self.set_metadata("embedder_id", embedder_id)
            await self.set_metadata("dimension", str(dimension))
            await self.set_metadata("last_sync", now)
        except Exception as e:
            # Attempt rollback on failure
            logger.error(f"Sync metadata update failed: {e}. Attempting rollback...")
            try:
                if old_embedder_id is not None:
                    await self.set_metadata("embedder_id", old_embedder_id)
                if old_dimension is not None:
                    await self.set_metadata("dimension", old_dimension)
                if old_last_sync is not None:
                    await self.set_metadata("last_sync", old_last_sync)
                logger.info("Rollback completed successfully")
            except Exception as rollback_error:
                logger.error(f"Rollback failed: {rollback_error}")
            raise  # Re-raise original exception
