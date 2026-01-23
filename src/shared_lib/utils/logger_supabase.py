"""
Supabase Logger for Session Synchronization

Synchronizes local JSON session logs to Supabase relational tables.
Implements upsert strategy with referential integrity (session_logs -> query_logs).
"""

import os
import logging
import threading
from typing import Dict, Any, Optional
from supabase import create_client, Client

logger = logging.getLogger(__name__)


class SupabaseLogger:
    """
    Manages synchronization of session logs to Supabase.

    Ensures referential integrity by:
    1. Upserting session data (parent table)
    2. Upserting query data (child table) in batch

    Features:
    - Backward compatible: Gracefully handles missing columns in schema
    - Schema detection: Auto-detects available columns
    - Fallback mode: Uses legacy schema if new columns not available
    """

    def __init__(self):
        """
        Initialize Supabase client from environment variables.

        Requires:
            SUPABASE_URL: Project URL
            SUPABASE_API_KEY: Service Role key (recommended for backend)

        Raises:
            ValueError: If required environment variables are missing
        """
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_API_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "ERRO: As variáveis SUPABASE_URL e SUPABASE_API_KEY são obrigatórias."
            )

        # Initialize Supabase client
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        self._lock = threading.Lock()

        # Schema detection cache
        self._schema_version = None  # "v1" (legacy) or "v2" (with per-agent columns)
        self._schema_detection_attempted = False

        logger.info("Supabase client initialized successfully")

    def test_connection(self) -> Dict[str, Any]:
        """
        Test Supabase connection and permissions by attempting operations.

        Returns:
            Dict with diagnostic information about connection status
        """
        results = {
            "connection": False,
            "can_select_session_logs": False,
            "can_insert_session_logs": False,
            "can_select_query_logs": False,
            "errors": [],
            "supabase_url": self.supabase_url,
            "key_prefix": self.supabase_key[:20] + "..." if self.supabase_key else None,
        }

        # Test 1: Basic connection with SELECT on session_logs
        try:
            response = self.supabase.table("session_logs").select("session_id").limit(1).execute()
            results["connection"] = True
            results["can_select_session_logs"] = True
            if hasattr(response, 'error') and response.error:
                results["can_select_session_logs"] = False
                results["errors"].append(f"session_logs SELECT error: {response.error}")
        except Exception as e:
            results["errors"].append(f"session_logs SELECT exception: {str(e)}")

        # Test 2: Test INSERT on session_logs with a test record
        try:
            import uuid
            test_session_id = f"_test_{uuid.uuid4().hex[:8]}"
            test_data = {
                "session_id": test_session_id,
                "user_email": "test@diagnostic.local",
                "session_start": "2024-01-01T00:00:00",
                "session_status": "test",
                "total_queries": 0,
            }
            response = self.supabase.table("session_logs").insert(test_data).execute()

            if hasattr(response, 'error') and response.error:
                results["can_insert_session_logs"] = False
                results["errors"].append(f"session_logs INSERT error: {response.error}")
            elif response.data:
                results["can_insert_session_logs"] = True
                # Clean up test record
                self.supabase.table("session_logs").delete().eq("session_id", test_session_id).execute()
            else:
                results["can_insert_session_logs"] = False
                results["errors"].append("session_logs INSERT returned no data (silent failure)")
        except Exception as e:
            results["errors"].append(f"session_logs INSERT exception: {str(e)}")

        # Test 3: Test SELECT on query_logs
        try:
            response = self.supabase.table("query_logs").select("session_id").limit(1).execute()
            results["can_select_query_logs"] = True
            if hasattr(response, 'error') and response.error:
                results["can_select_query_logs"] = False
                results["errors"].append(f"query_logs SELECT error: {response.error}")
        except Exception as e:
            results["errors"].append(f"query_logs SELECT exception: {str(e)}")

        return results

    def _detect_schema_version(self) -> str:
        """
        Detect the schema version by attempting a test query.

        Returns:
            "v2" if per-agent token columns exist, "v1" otherwise
        """
        if self._schema_detection_attempted:
            return self._schema_version or "v1"

        self._schema_detection_attempted = True

        try:
            # Attempt to query with new column (minimal query, limit 0)
            result = (
                self.supabase.table("query_logs")
                .select("session_id, filter_classifier_input_tokens")
                .limit(0)
                .execute()
            )

            # If no error, v2 schema is available
            self._schema_version = "v2"
            logger.info(
                "Supabase schema: v2 detected (per-agent token columns available)"
            )
            return "v2"

        except Exception as e:
            error_msg = str(e)

            # Check if error is specifically about missing column
            if "filter_classifier_input_tokens" in error_msg or "PGRST204" in error_msg:
                self._schema_version = "v1"
                logger.warning(
                    "Supabase schema: v1 detected (per-agent columns NOT available). "
                    "Using legacy schema. Run migrations/add_per_agent_token_tracking.sql to upgrade."
                )
                return "v1"

            # Unknown error - assume v1 for safety
            logger.warning(
                f"Schema detection failed: {error_msg}. Assuming v1 (legacy schema)"
            )
            self._schema_version = "v1"
            return "v1"

    def _build_query_data_v1(
        self, query: Dict[str, Any], session_id: str
    ) -> Dict[str, Any]:
        """
        Build query data payload for v1 schema (legacy, without per-agent columns).

        Args:
            query: Query data from session log
            session_id: Session ID

        Returns:
            Dict with v1 schema fields only
        """
        agent_tokens = query.get("token_usage_by_agent", {})

        return {
            "session_id": session_id,
            "query_sequence_id": query.get("query_id"),
            "timestamp": query.get("timestamp"),
            "user_query": query.get("user_query"),
            "output_type": query.get("output_type"),
            "status": query.get("status"),
            "execution_time": query.get("execution_time", 0.0),
            "error": query.get("error"),
            # Optional fields
            "chart_type": query.get("chart_type"),
            "query_type": query.get("query_type"),
            # JSONB fields
            "metrics": query.get("metrics", {}),
            "data_quality": query.get("data_quality", {}),
            "filters_applied": query.get("filters_applied", {}),
            # File references
            "formatter_output_reference": query.get("formatter_output_reference"),
            "non_graph_output_reference": query.get("non_graph_output_reference"),
            # Token tracking fields (aggregated only)
            "total_input_tokens": query.get("total_input_tokens", 0),
            "total_output_tokens": query.get("total_output_tokens", 0),
            "total_tokens": query.get("total_tokens", 0),
            "token_usage_by_agent": agent_tokens,
        }

    def _build_query_data_v2(
        self, query: Dict[str, Any], session_id: str
    ) -> Dict[str, Any]:
        """
        Build query data payload for v2 schema (with per-agent token columns).

        Args:
            query: Query data from session log
            session_id: Session ID

        Returns:
            Dict with v2 schema fields (includes per-agent breakdowns)
        """
        # Extract agent-specific token data
        agent_tokens = query.get("token_usage_by_agent", {})

        # Helper function to normalize model names
        def normalize_agent_model_name(model_name: str) -> str:
            """Remove 'models/' prefix from model names"""
            if model_name and model_name.startswith("models/"):
                return model_name.replace("models/", "", 1)
            return model_name

        # Extract per-agent metrics with normalized model names
        filter_classifier_data = agent_tokens.get("filter_classifier", {})
        graphic_classifier_data = agent_tokens.get("graphic_classifier", {})
        insight_generator_data = agent_tokens.get("insight_generator", {})

        # Start with v1 fields
        q_data = self._build_query_data_v1(query, session_id)

        # Add v2-specific fields (per-agent breakdowns)
        q_data.update(
            {
                "filter_classifier_input_tokens": filter_classifier_data.get(
                    "input_tokens", 0
                ),
                "filter_classifier_output_tokens": filter_classifier_data.get(
                    "output_tokens", 0
                ),
                "filter_classifier_total_tokens": filter_classifier_data.get(
                    "total_tokens", 0
                ),
                "filter_classifier_model": normalize_agent_model_name(
                    filter_classifier_data.get("model_name", "unknown")
                ),
                "graphic_classifier_input_tokens": graphic_classifier_data.get(
                    "input_tokens", 0
                ),
                "graphic_classifier_output_tokens": graphic_classifier_data.get(
                    "output_tokens", 0
                ),
                "graphic_classifier_total_tokens": graphic_classifier_data.get(
                    "total_tokens", 0
                ),
                "graphic_classifier_model": normalize_agent_model_name(
                    graphic_classifier_data.get("model_name", "unknown")
                ),
                "insight_generator_input_tokens": insight_generator_data.get(
                    "input_tokens", 0
                ),
                "insight_generator_output_tokens": insight_generator_data.get(
                    "output_tokens", 0
                ),
                "insight_generator_total_tokens": insight_generator_data.get(
                    "total_tokens", 0
                ),
                "insight_generator_model": normalize_agent_model_name(
                    insight_generator_data.get("model_name", "unknown")
                ),
            }
        )

        return q_data

    def sync_log_to_supabase(self, json_log: Dict[str, Any]) -> bool:
        """
        Synchronize local JSON log to Supabase tables.

        Strategy:
        1. Upsert to parent table (session_logs)
        2. Batch upsert to child table (query_logs)

        Args:
            json_log: Session log dictionary with session_metadata, queries, session_summary

        Returns:
            True if sync succeeded, False otherwise
        """
        with self._lock:
            try:
                # Extract log components
                meta = json_log.get("session_metadata", {})
                summary = json_log.get("session_summary", {})
                queries_list = json_log.get("queries", [])

                session_id = meta.get("session_id")

                if not session_id:
                    logger.warning("Supabase sync skipped: missing session_id")
                    return False

                # Log sync attempt for debugging
                logger.debug(
                    f"Supabase sync starting: session_id={session_id}, "
                    f"user_email={meta.get('user_email')}, "
                    f"queries_count={len(queries_list)}"
                )

                # Map to session_logs table schema
                session_data = {
                    "session_id": session_id,
                    "user_email": meta.get("user_email"),
                    "session_start": meta.get("session_start"),
                    "session_last_update": meta.get("session_last_update"),
                    "total_queries": meta.get("total_queries", 0),
                    "session_status": meta.get("session_status"),
                    "session_summary": summary,  # JSONB field
                    # Token tracking fields
                    "total_input_tokens": summary.get("total_input_tokens", 0),
                    "total_output_tokens": summary.get("total_output_tokens", 0),
                    "total_tokens": summary.get("total_tokens", 0),
                    "total_execution_time": summary.get("total_execution_time", 0.0),
                    "average_query_time": summary.get("average_query_time", 0.0),
                    "total_successful_queries": summary.get(
                        "total_successful_queries", 0
                    ),
                    "total_failed_queries": summary.get("total_failed_queries", 0),
                    "chart_types_used": summary.get("chart_types_used", []),
                    "unique_filters_used": summary.get("unique_filters_used", []),
                }

                # Add optional session_end if present
                session_end = meta.get("session_end")
                if session_end:
                    session_data["session_end"] = session_end

                # Step 1: Persist session (parent) - MUST run before queries
                session_response = self.supabase.table("session_logs").upsert(
                    session_data, on_conflict="session_id"
                ).execute()

                # Check for errors in session_logs upsert
                if hasattr(session_response, 'error') and session_response.error:
                    logger.error(
                        f"Supabase session_logs upsert failed: {session_response.error}"
                    )
                    return False

                # Validate data was inserted
                if not session_response.data:
                    logger.warning(
                        f"Supabase session_logs upsert returned no data for session_id={session_id}. "
                        f"This may indicate a silent failure. Check RLS policies and permissions."
                    )
                else:
                    logger.debug(f"Session {session_id} upserted to session_logs successfully")

                # Step 2: Persist queries (children) in batch
                if queries_list:
                    # Detect schema version on first sync attempt
                    schema_version = self._detect_schema_version()

                    queries_data_to_insert = []

                    for query in queries_list:
                        # Build query data based on detected schema version
                        if schema_version == "v2":
                            q_data = self._build_query_data_v2(query, session_id)
                        else:
                            q_data = self._build_query_data_v1(query, session_id)

                        queries_data_to_insert.append(q_data)

                    # Batch upsert for performance
                    queries_response = self.supabase.table("query_logs").upsert(
                        queries_data_to_insert,
                        on_conflict="session_id,query_sequence_id",
                    ).execute()

                    # Check for errors in query_logs upsert
                    if hasattr(queries_response, 'error') and queries_response.error:
                        logger.error(
                            f"Supabase query_logs upsert failed: {queries_response.error}"
                        )
                        return False

                    # Validate data was inserted
                    if not queries_response.data:
                        logger.warning(
                            f"Supabase query_logs upsert returned no data for session_id={session_id}. "
                            f"This may indicate a silent failure (FK constraint, RLS, or permissions issue)."
                        )
                    else:
                        logger.debug(
                            f"{len(queries_data_to_insert)} queries upserted to query_logs "
                            f"(schema: {schema_version})"
                        )

                logger.info(
                    f"Log synced to Supabase: {session_id} (Status: {meta.get('session_status')})"
                )
                return True

            except Exception as e:
                # Log detailed error for debugging
                logger.error(
                    f"Supabase sync error for session_id={json_log.get('session_metadata', {}).get('session_id')}: {e}",
                    exc_info=True
                )
                return False


# Global singleton instance (lazy initialization)
_supabase_logger: Optional[SupabaseLogger] = None


def get_supabase_logger() -> Optional[SupabaseLogger]:
    """
    Get or create the global SupabaseLogger instance.

    Returns:
        SupabaseLogger instance if credentials are available, None otherwise
    """
    global _supabase_logger

    if _supabase_logger is None:
        try:
            _supabase_logger = SupabaseLogger()
        except ValueError as e:
            logger.info(f"Supabase logging disabled: {e}")
            return None

    return _supabase_logger


def sync_log_to_supabase(json_log: Dict[str, Any]) -> bool:
    """
    Convenience function to sync log to Supabase.

    Args:
        json_log: Session log dictionary

    Returns:
        True if sync succeeded, False otherwise
    """
    supabase_logger = get_supabase_logger()
    if supabase_logger:
        return supabase_logger.sync_log_to_supabase(json_log)
    return False


__all__ = ["SupabaseLogger", "sync_log_to_supabase", "get_supabase_logger"]
