import unittest
from pathlib import Path
import tempfile
import shutil
from src.veruca.sources.obsidian import ObsidianVault

class TestObsidianVault(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for the test vault
        self.test_dir = tempfile.mkdtemp()
        self.vault_path = Path(self.test_dir)

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    def test_initial_state(self):
        """Test that a newly initialized vault has no vector store."""
        vault = ObsidianVault(str(self.vault_path))
        self.assertIsNone(vault.vector_store, "Vector store should be None at initialization")

        # Test that querying an empty vault returns appropriate message
        result = vault.query("test query")
        self.assertEqual(result, "No documents found in the vault.")

if __name__ == '__main__':
    unittest.main()