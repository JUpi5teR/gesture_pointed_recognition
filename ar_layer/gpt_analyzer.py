import os
import json
import time
import base64
import logging
import hashlib
import cv2
from pathlib import Path
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)

class GPTImageAnalyzer:
    def __init__(self, api_key="", model="gpt-4o", cache_dir=None):
        self.api_key = api_key
        self.model = model
        self._cache = {}
        cache_dir = cache_dir or (Path(__file__).parent.parent / ".gpt_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_cache()

    def _cache_key(self, image_bytes):
        return hashlib.sha256(image_bytes).hexdigest()[:32]

    def _cache_path(self, key):
        return self.cache_dir / f"{key}.json"

    def _load_cache(self):
        for f in self.cache_dir.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    self._cache[data["key"]] = data["result"]
            except Exception:
                pass

    def _save_to_cache(self, key, result):
        self._cache[key] = result
        try:
            with open(self._cache_path(key), "w", encoding="utf-8") as f:
                json.dump({"key": key, "result": result, "time": time.time()}, f, ensure_ascii=False)
        except Exception as e:
            logger.warning("Cache write failed: %s", e)

    def analyze_image(self, image_bytes, prompt=None):
        if not HAS_OPENAI:
            return "[GPT unavailable] Install openai: pip install openai"
        if not self.api_key:
            return "[API key not set] Login first."
        prompt = prompt or "Please describe the main subject in the image with basic information."
        ckey = self._cache_key(image_bytes)
        if ckey in self._cache:
            logger.info("GPT cache hit for %s", ckey[:8])
            return self._cache[ckey]
        try:
            client = OpenAI(api_key=self.api_key)
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}}
                    ]
                }],
                max_tokens=300
            )
            result = resp.choices[0].message.content.strip()
            self._save_to_cache(ckey, result)
            logger.info("GPT analysis cached")
            return result
        except Exception as e:
            err = f"[GPT error: {e}]"
            logger.error(err)
            return err

    def crop_from_frame(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        ret, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ret:
            return buf.tobytes()
        return None
