import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- DEBUGGING STEP ---
# This will print the environment variable as Python sees it inside the container.
# This will prove if the 'docker run -e' command is working.
print("--- [CONFIG DEBUG] Checking for GROQ_API_KEY ---")
api_key_from_os = os.getenv("GROQ_API_KEY")
if api_key_from_os:
    # For security, we only print a part of the key.
    print(f"SUCCESS: Found GROQ_API_KEY in environment, starting with '{api_key_from_os[:7]}...'")
else:
    print("FAILURE: GROQ_API_KEY was NOT found in the environment variables.")
print("---------------------------------------------")


class Settings(BaseSettings):
    # This defines the required environment variable.
    GROQ_API_KEY: str

    # This is the corrected configuration.
    # By completely removing `env_file`, we force Pydantic to ONLY
    # look for settings in the system's environment variables.
    # This is the standard, most reliable method for Docker.
    model_config = SettingsConfigDict(extra='ignore')

# This line will now read from the environment variables set by the `docker run` command.
# If it fails, the app will crash on startup with a clear error.
settings = Settings()