"""Tests for analytics module."""

import json
import os
import tempfile
from datetime import datetime, timezone
from unittest import mock

import pytest


class TestAnalyticsRecording:
    """Tests for request recording."""

    @pytest.fixture
    def temp_analytics_dir(self):
        """Create temporary directory for analytics data."""
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch("analytics.DATA_DIR", tmpdir),
            mock.patch("analytics.COUNTS_FILE", os.path.join(tmpdir, "counts.json")),
            mock.patch("analytics.LOCK_FILE", os.path.join(tmpdir, "analytics.lock")),
        ):
            yield tmpdir

    @pytest.mark.asyncio
    async def test_record_request_increments_count(self, temp_analytics_dir):
        """Test record_request increments counter."""
        from analytics import COUNTS_FILE, record_request

        # Record a search request
        await record_request("search")

        # Check count was incremented
        with open(COUNTS_FILE) as f:
            data = json.load(f)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert today in data
        assert data[today]["search"] >= 1

    @pytest.mark.asyncio
    async def test_record_request_search_and_fetch(self, temp_analytics_dir):
        """Test recording both search and fetch requests."""
        from analytics import COUNTS_FILE, record_request

        await record_request("search")
        await record_request("fetch")
        await record_request("search")

        with open(COUNTS_FILE) as f:
            data = json.load(f)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert data[today]["search"] >= 2
        assert data[today]["fetch"] >= 1

    @pytest.mark.asyncio
    async def test_record_request_unknown_tool_defaults_to_search(
        self, temp_analytics_dir
    ):
        """Test unknown tool name defaults to search."""
        from analytics import COUNTS_FILE, record_request

        await record_request("unknown_tool")

        with open(COUNTS_FILE) as f:
            data = json.load(f)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert data[today]["search"] >= 1


class TestAnalyticsDataFrame:
    """Tests for analytics DataFrame generation."""

    @pytest.fixture
    def temp_analytics_with_data(self):
        """Create temp directory with sample data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            counts_file = os.path.join(tmpdir, "counts.json")

            # Create sample data
            sample_data = {
                "2024-01-01": {"search": 10, "fetch": 5},
                "2024-01-02": {"search": 20, "fetch": 15},
                "2024-01-03": {"search": 30, "fetch": 25},
            }
            with open(counts_file, "w") as f:
                json.dump(sample_data, f)

            with (
                mock.patch("analytics.DATA_DIR", tmpdir),
                mock.patch("analytics.COUNTS_FILE", counts_file),
                mock.patch(
                    "analytics.LOCK_FILE", os.path.join(tmpdir, "analytics.lock")
                ),
            ):
                yield tmpdir

    def test_last_n_days_returns_dataframe(self, temp_analytics_with_data):
        """Test last_n_days_count_df returns DataFrame."""
        import pandas as pd

        from analytics import last_n_days_count_df

        df = last_n_days_count_df("search", 7)

        assert isinstance(df, pd.DataFrame)
        assert "date" in df.columns
        assert "count" in df.columns
        assert "full_date" in df.columns

    def test_last_n_days_correct_length(self, temp_analytics_with_data):
        """Test DataFrame has correct number of rows."""
        from analytics import last_n_days_count_df

        df = last_n_days_count_df("search", 14)

        assert len(df) == 14

    def test_last_n_days_unknown_tool_defaults_to_search(
        self, temp_analytics_with_data
    ):
        """Test unknown tool defaults to search."""
        from analytics import last_n_days_count_df

        df = last_n_days_count_df("invalid_tool", 7)

        # Should not raise, defaults to search
        assert len(df) == 7


class TestAnalyticsSchemaCompat:
    """Tests for schema backward compatibility."""

    def test_normalize_old_schema(self):
        """Test normalizing old schema (int values)."""
        from analytics import _normalize_counts_schema

        old_data = {
            "2024-01-01": 100,
            "2024-01-02": 200,
        }

        normalized = _normalize_counts_schema(old_data)

        assert normalized["2024-01-01"] == {"search": 100, "fetch": 0}
        assert normalized["2024-01-02"] == {"search": 200, "fetch": 0}

    def test_normalize_new_schema_unchanged(self):
        """Test new schema passes through unchanged."""
        from analytics import _normalize_counts_schema

        new_data = {
            "2024-01-01": {"search": 50, "fetch": 30},
        }

        normalized = _normalize_counts_schema(new_data)

        assert normalized["2024-01-01"] == {"search": 50, "fetch": 30}

    def test_normalize_mixed_schema(self):
        """Test normalizing mixed old/new schema."""
        from analytics import _normalize_counts_schema

        mixed_data = {
            "2024-01-01": 100,  # old
            "2024-01-02": {"search": 50, "fetch": 30},  # new
        }

        normalized = _normalize_counts_schema(mixed_data)

        assert normalized["2024-01-01"] == {"search": 100, "fetch": 0}
        assert normalized["2024-01-02"] == {"search": 50, "fetch": 30}
