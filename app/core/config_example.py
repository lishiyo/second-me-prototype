from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Wasabi
    aws_access_key_id: str
    aws_secret_access_key: str
    wasabi_endpoint: str = "https://s3.us-west-1.wasabisys.com"
    bucket: str = "second-me-prototype"

    # Weaviate
    weaviate_url: str
    weaviate_api_key: str

    # Postgres
    database_url: str

    # Modal
    modal_env: str = "second-me-prototype"   # the Modal “Environment” you create

settings = Settings()        # import this everywhere
