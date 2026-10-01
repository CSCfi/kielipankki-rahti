import os
import sys

# Make ``asr`` importable and keep the API from reaching a real Redis.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("ASR_LANGUAGES", "fi,sme")
