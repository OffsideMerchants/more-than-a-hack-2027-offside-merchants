import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IDSSE_DIR = os.path.join(ROOT, "idsse-data")
ARTIFACTS = os.path.join(ROOT, "artifacts")
REPORTS = os.path.join(ROOT, "arsh-research", "results")
BARATH_ART = os.path.join(ARTIFACTS, "barath")  # Red tier: never opened by an LLM session
