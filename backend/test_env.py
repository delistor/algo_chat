import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"d:\daily_stock\.env")

key = os.getenv("OPENAI_API_KEY", "")
base = os.getenv("OPENAI_API_BASE", "")
model = os.getenv("OPENAI_MODEL", "")
print(f"OPENAI_API_KEY: set={len(key)>0}, len={len(key)}, prefix={key[:12] if key else 'NONE'}")
print(f"OPENAI_API_BASE: {base}")
print(f"OPENAI_MODEL: {model}")