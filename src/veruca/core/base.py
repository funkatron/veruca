"""Base classes for Veruca data sources."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class DataSource(ABC):
    """Abstract base class for all data sources in Veruca.

    This class defines the interface that all data sources must implement.
    Data sources are responsible for loading, indexing, and querying documents.
    """

    @abstractmethod
    def load_documents(self) -> List[Tuple[str, Dict[str, Any], str]]:
        """Load documents from the data source.

        Each document is represented as a tuple of:
        - filename: str - The name of the document
        - metadata: Dict[str, Any] - Document metadata
        - content: str - The document content

        :return: List of document tuples
        """
        pass

    @abstractmethod
    def index_documents(self) -> None:
        """Index the documents for querying.

        This method should create any necessary indices or data structures
        to enable efficient querying of the documents.
        """
        pass

    @abstractmethod
    def query(self, query: str, filters: Optional[Dict[str, str]] = None) -> str:
        """Query the indexed documents.

        :param query: The query string to search for
        :param filters: Optional dictionary of metadata filters to apply
        :return: The query result as a string
        :raises ValueError: If the data source is not indexed
        """
        pass