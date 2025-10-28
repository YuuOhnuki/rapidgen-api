"""
アプリケーション設定モジュール

環境変数やデフォルト設定を管理します。
Stable Diffusion XL (SDXL) + LCM の設定を含みます。
"""

import os
from typing import Literal
import torch
from pydantic_settings import BaseSettings
from pydantic import Field


class AppSettings(BaseSettings):
    """
    アプリケーション設定クラス
    
    環境変数から設定を読み込み、デフォルト値を提供します。
    """
    
    # FastAPI アプリケーション設定
    app_title: str = Field(default="Stable Diffusion XL LCM Img2Img API", description="アプリケーション名")
    debug: bool = Field(default=False, description="デバッグモード")
    
    # ML モデル設定
    model_id: str = Field(
        default="stabilityai/stable-diffusion-xl-base-1.0",
        description="使用するSDXLベースモデルのID"
    )
    lora_id: str = Field(
        default="latent-consistency/lcm-lora-sdxl",
        description="使用するLCM LoRAのID"
    )
    
    # デバイス設定
    device: Literal["cuda", "cpu"] = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="推論に使用するデバイス"
    )
    
    # デフォルト生成パラメータ
    default_steps: int = Field(default=20, ge=1, le=100, description="デフォルトの推論ステップ数")
    default_guidance: float = Field(default=1.5, ge=0.1, le=20.0, description="デフォルトのガイダンススケール")
    default_strength: float = Field(default=0.7, ge=0.0, le=1.0, description="デフォルトのImg2Img強度")
    
    # 画像設定
    target_width: int = Field(default=1024, ge=256, le=2048, description="出力画像の幅")
    target_height: int = Field(default=1024, ge=256, le=2048, description="出力画像の高さ")
    
    # メモリ最適化設定
    enable_cpu_offload: bool = Field(default=True, description="CPUオフロードを有効にするか")
    enable_attention_slicing: bool = Field(default=True, description="アテンション・スライシングを有効にするか")
    
    model_config = {
        "env_prefix": "SDXL_",
        "case_sensitive": False
    }

    @property
    def torch_dtype(self) -> torch.dtype:
        """
        デバイスに応じた適切なTorchデータ型を返す
        
        Returns:
            torch.dtype: CUDAの場合はfloat16、CPUの場合はfloat32
        """
        return torch.float16 if self.device == "cuda" else torch.float32
    
    @property
    def variant(self) -> str | None:
        """
        モデルロード時に使用するバリアント
        
        Returns:
            str | None: CUDAの場合は"fp16"、CPUの場合はNone
        """
        return "fp16" if self.device == "cuda" else None


# グローバル設定インスタンス
settings = AppSettings()
