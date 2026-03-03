"""Pytest configuration and shared fixtures."""
import os
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment variables before importing modules
os.environ.setdefault("SERPER_API_KEY", "test-key-for-mocking")
os.environ.setdefault("ANALYTICS_DATA_DIR", "/tmp/test-analytics")


@pytest.fixture
def sample_documents():
    """Sample documents for RAG testing."""
    return [
        {
            "url": "https://example.com/science",
            "content": (
                "The speed of light in vacuum is exactly 299,792,458 metres per second. "
                "This is a fundamental constant of nature denoted by c. Light travels "
                "at this speed regardless of the motion of the source or observer. "
                "Einstein's theory of special relativity is built upon this constant."
            ),
        },
        {
            "url": "https://example.com/history",
            "content": (
                "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
                "in Paris, France. It was constructed from 1887 to 1889 as the entrance "
                "to the 1889 World's Fair. The tower is 330 metres tall and was the "
                "tallest man-made structure in the world for 41 years."
            ),
        },
        {
            "url": "https://example.com/geography",
            "content": (
                "Mount Everest is Earth's highest mountain above sea level, located in "
                "the Mahalangur Himal sub-range of the Himalayas. The China-Nepal border "
                "runs across its summit point. Its elevation of 8,848.86 metres was most "
                "recently established in 2020 by the Chinese and Nepali authorities."
            ),
        },
    ]


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What is the speed of light?"


@pytest.fixture
def sample_chunks():
    """Sample text chunks for reranker testing."""
    return [
        "The speed of light in vacuum is exactly 299,792,458 metres per second.",
        "The Eiffel Tower is located in Paris, France.",
        "Mount Everest is the highest mountain on Earth.",
        "Einstein developed the theory of special relativity.",
    ]


@pytest.fixture
def mock_search_response():
    """Mock response from Serper API."""
    return {
        "organic": [
            {
                "title": "Speed of Light - Wikipedia",
                "link": "https://en.wikipedia.org/wiki/Speed_of_light",
                "snippet": "The speed of light in vacuum is 299,792,458 m/s.",
            },
            {
                "title": "What is the Speed of Light? - NASA",
                "link": "https://science.nasa.gov/speed-of-light",
                "snippet": "Light travels at approximately 300,000 km/s.",
            },
        ]
    }


@pytest.fixture
def mock_fetch_response():
    """Mock HTML response for fetch testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Test Page</title></head>
    <body>
        <article>
            <h1>Speed of Light</h1>
            <p>The speed of light in vacuum is exactly 299,792,458 metres per second.</p>
            <p>This fundamental constant is denoted by the letter c.</p>
        </article>
    </body>
    </html>
    """
