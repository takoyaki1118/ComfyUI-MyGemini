import os
import torch
from PIL import Image
import numpy as np
from google.genai import Client

class MyGeminiNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "この画像について説明してください。"}),
                "model_name": (["gemini-2.5-flash", "gemini-2.5-pro"], {"default": "gemini-2.5-flash"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "image": ("IMAGE",), # ComfyUIの画像テンソルを受け取る場合
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_output",)
    FUNCTION = "generate_gemini"
    CATEGORY = "utils/gemini"

    def generate_gemini(self, prompt, model_name, api_key, image=None):
        # APIキーの優先順位: ノードに入力されたキー > 環境変数
        key = api_key.strip() if api_key.strip() else os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("Gemini APIキーが設定されていません。ノードに入力するか、環境変数を設定してください。")

        # クライアントの初期化
        client = Client(api_key=key)

        contents = [prompt]

        # ComfyUIのIMAGEテンソル（PyTorch）がある場合はPIL Imageに変換して追加
        if image is not None:
            # batchの先頭を取り出し、0-255のnumpy配列(uint8)に変換
            i = image[0].cpu().numpy()
            img_np = np.clip(i * 255.0, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            contents.append(pil_img)

        # Gemini APIの呼び出し
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
        )

        return (response.text,)

# ComfyUIへの登録用マッピング
NODE_CLASS_MAPPINGS = {
    "MyGeminiNode": MyGeminiNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyGeminiNode": "Gemini API Caller (Custom)"
}