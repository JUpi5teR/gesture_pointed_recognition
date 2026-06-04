import os
import json
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_FILE = Path(__file__).parent.parent / ".ar_session.json"

PROVIDER_CHATGPT = "chatgpt"
PROVIDER_DEEPSEEK = "deepseek"
PROVIDER_QWEN = "qwen"
PROVIDER_GLM = "glm"

PROVIDERS = {
    PROVIDER_CHATGPT: "ChatGPT (OpenAI)",
    PROVIDER_DEEPSEEK: "DeepSeek",
    PROVIDER_QWEN: "Qwen (Alibaba)",
    PROVIDER_GLM: "GLM (Zhipu AI)",
}

PROVIDER_DEFAULT_MODEL = {
    PROVIDER_CHATGPT: "gpt-4o",
    PROVIDER_DEEPSEEK: "deepseek-vl",
    PROVIDER_QWEN: "qwen-vl-max",
    PROVIDER_GLM: "glm-4.6V-flash",
}

PROVIDER_BASE_URL = {
    PROVIDER_CHATGPT: "https://api.openai.com/v1",
    PROVIDER_DEEPSEEK: "https://api.deepseek.com/v1",
    PROVIDER_QWEN: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    PROVIDER_GLM: "https://open.bigmodel.cn/api/paas/v4",
}

DEFAULT_CONFIG = {
    "api_key": "",
    "session_token": "",
    "login_time": 0,
    "provider": PROVIDER_CHATGPT,
    "model": PROVIDER_DEFAULT_MODEL[PROVIDER_CHATGPT],
    "logged_in": False
}

class LoginManager:
    def __init__(self, config_path=None):
        self.config_path = Path(config_path) if config_path else CONFIG_FILE
        self.config = self._load()
        self._logged_in = self.config.get("logged_in", False) and bool(self.config.get("api_key", ""))

    def _load(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return {**DEFAULT_CONFIG, **json.load(f)}
            except Exception as e:
                logger.warning("Failed to load session: %s", e)
        return dict(DEFAULT_CONFIG)

    def _save(self):
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save session: %s", e)

    def login(self, api_key, provider=None):
        if not api_key or len(api_key) < 10:
            return False
        if provider and provider in PROVIDERS:
            self.config["provider"] = provider
            self.config["model"] = PROVIDER_DEFAULT_MODEL[provider]
        self.config["api_key"] = api_key
        self.config["session_token"] = "sess_" + str(int(time.time()))
        self.config["login_time"] = time.time()
        self.config["logged_in"] = True
        self._logged_in = True
        self._save()
        logger.info("Login successful")
        return True

    def logout(self):
        self.config["api_key"] = ""
        self.config["session_token"] = ""
        self.config["provider"] = PROVIDER_CHATGPT
        self.config["logged_in"] = False
        self._logged_in = False
        self._save()

    @property
    def is_logged_in(self):
        return self._logged_in and bool(self.config.get("api_key", ""))

    @property
    def api_key(self):
        return self.config.get("api_key", "")

    @property
    def provider(self):
        return self.config.get("provider", PROVIDER_CHATGPT)

    @property
    def base_url(self):
        return PROVIDER_BASE_URL.get(self.provider, PROVIDER_BASE_URL[PROVIDER_CHATGPT])

    @property
    def model_name(self):
        return self.config.get("model", PROVIDER_DEFAULT_MODEL.get(self.provider, PROVIDER_DEFAULT_MODEL[PROVIDER_CHATGPT]))
