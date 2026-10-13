import os

import numpy as np
from PIL import Image

try:
    from google.genai import Client, types
    _GENAI_IMPORT_ERROR = None
except Exception as e:  # keep the whole node pack loadable even if the SDK is missing
    Client = None
    types = None
    _GENAI_IMPORT_ERROR = e

# エンドポイント名で記述(先頭がデフォルト)
GEMINI_ENDPOINTS = [
    "gemini-3.1-flash-lite",   # メイン
    "gemini-3.5-flash-lite",   # サブメイン
    "gemini-3.6-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
]
FALLBACK_CHOICES = ["none"] + GEMINI_ENDPOINTS

SHOTS_SCHEMA = {
    "type": "object",
    "properties": {
        "shots": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4,
            "maxItems": 4,
        }
    },
    "required": ["shots"],
}


class MyGeminiNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "この画像について説明してください。"}),
                "model_name": (GEMINI_ENDPOINTS, {"default": GEMINI_ENDPOINTS[0]}),
                "fallback_model": (FALLBACK_CHOICES, {"default": "gemini-3.5-flash-lite"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
                "force_shots_json": ("BOOLEAN", {"default": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "image": ("IMAGE",),  # ComfyUIの画像テンソルを受け取る場合
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text_output", "used_model")
    FUNCTION = "generate_gemini"
    CATEGORY = "Storyboard"

    def generate_gemini(self, prompt, model_name, fallback_model, temperature,
                        force_shots_json, api_key, system_prompt="", image=None):
        if Client is None:
            raise RuntimeError(
                "google-genai がインポートできません。ComfyUIのPython環境で "
                "`pip install google-genai` を実行してください。\n"
                f"原因: {_GENAI_IMPORT_ERROR}"
            )

        # APIキーの優先順位: ノードに入力されたキー > 環境変数
        key = api_key.strip() if api_key.strip() else (
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        )
        if not key:
            raise ValueError("Gemini APIキーが設定されていません。ノードに入力するか、環境変数(GEMINI_API_KEY)を設定してください。")

        client = Client(api_key=key)

        contents = [prompt]

        # ComfyUIのIMAGEテンソル(PyTorch)がある場合はPIL Imageに変換して追加
        if image is not None:
            i = image[0].cpu().numpy()
            img_np = np.clip(i * 255.0, 0, 255).astype(np.uint8)[:, :, :3]
            pil_img = Image.fromarray(img_np)
            pil_img.thumbnail((1024, 1024))
            contents.append(pil_img)

        def build_config(use_schema):
            kwargs = {"temperature": temperature}
            if system_prompt and system_prompt.strip():
                kwargs["system_instruction"] = system_prompt
            if force_shots_json:
                kwargs["response_mime_type"] = "application/json"
                if use_schema:
                    kwargs["response_schema"] = SHOTS_SCHEMA
            return types.GenerateContentConfig(**kwargs)

        order = [model_name]
        if fallback_model != "none" and fallback_model != model_name:
            order.append(fallback_model)

        errors = []
        for ep in order:
            # 1回目: スキーマ付き / 2回目: スキーマなし(スキーマ非対応時の保険)
            for use_schema in ([True, False] if force_shots_json else [False]):
                try:
                    response = client.models.generate_content(
                        model=ep,
                        contents=contents,
                        config=build_config(use_schema),
                    )
                    text = response.text
                    if text and text.strip():
                        return (text, ep)
                    errors.append(f"[{ep}] empty response")
                    break
                except Exception as e:
                    errors.append(f"[{ep}] schema={use_schema}: {e}")
                    print(f"[Storyboard] Gemini call failed ({ep}, schema={use_schema}): {e}")

        raise RuntimeError("全てのGeminiエンドポイントで失敗しました:\n" + "\n".join(errors))


# ComfyUIへの登録用マッピング
NODE_CLASS_MAPPINGS = {
    "MyGeminiNode": MyGeminiNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyGeminiNode": "Gemini API Caller (Custom)"
}