"""Makes `import src...` work under plain `pytest` regardless of cwd."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
