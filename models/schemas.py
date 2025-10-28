"""
データモデル定義

画像生成APIで使用するPydanticモデルを定義します。
リクエスト・レスポンス・タスク管理のためのデータ構造を含みます。
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum


class TaskStatus(str, Enum):
    """
    タスク実行状態を示すEnum
    """
    PENDING = "PENDING"         # 実行待機中
    IN_PROGRESS = "IN_PROGRESS" # 実行中
    COMPLETED = "COMPLETED"     # 完了
    FAILED = "FAILED"          # 失敗


class ImageGenerationRequest(BaseModel):
    """
    画像生成リクエストモデル
    
    img2img生成に必要なパラメータを定義します。
    """
    
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="画像生成に使用するプロンプト",
        example="a beautiful landscape with mountains and a lake"
    )
    
    init_image: str = Field(
        ...,
        description="元画像のBase64エンコードデータ (PNG/JPEG)",
        example="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    
    num_inference_steps: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="推論ステップ数（1-100）",
        example=20
    )
    
    guidance_scale: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=20.0,
        description="ガイダンススケール（0.1-20.0）",
        example=1.5
    )
    
    strength: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="変換強度（0.0-1.0、1.0に近いほど元画像から大きく変化）",
        example=0.7
    )
    
    negative_prompt: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="ネガティブプロンプト（生成したくない要素を指定）",
        example="blurry, low quality, watermark"
    )

    @validator('init_image')
    def validate_base64_image(cls, v):
        """
        Base64画像データの基本的な形式チェック
        """
        if not v.startswith(('data:image/', '/9j/', 'iVBORw0KGgo')):
            # 簡易的なBase64画像データの検証
            import base64
            try:
                base64.b64decode(v)
            except Exception:
                raise ValueError('Invalid base64 image data')
        return v


class TaskCreationResponse(BaseModel):
    """
    タスク作成レスポンスモデル
    """
    
    task_id: str = Field(
        ...,
        description="生成されたタスクのユニークID",
        example="123e4567-e89b-12d3-a456-426614174000"
    )


class TaskStatusResponse(BaseModel):
    """
    タスク状態レスポンスモデル
    """
    
    status: TaskStatus = Field(
        ...,
        description="タスクの現在の状態"
    )
    
    progress: int = Field(
        ...,
        ge=0,
        le=100,
        description="進捗率（パーセント）",
        example=75
    )


class TaskCompletedResponse(TaskStatusResponse):
    """
    タスク完了レスポンスモデル
    """
    
    dataUrl: str = Field(
        ...,
        description="生成された画像のBase64エンコードデータ",
        example="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )


class HealthCheckResponse(BaseModel):
    """
    ヘルスチェックレスポンスモデル
    """
    
    status: Literal["ok", "error"] = Field(
        ...,
        description="システムの状態"
    )
    
    device: Literal["cuda", "cpu"] = Field(
        ...,
        description="使用中のデバイス"
    )
    
    cuda_available: bool = Field(
        ...,
        description="CUDA（GPU）が利用可能かどうか"
    )
    
    ml_model_loaded: bool = Field(
        default=True,
        description="機械学習モデルがロード済みかどうか"
    )


class ErrorResponse(BaseModel):
    """
    エラーレスポンスモデル
    """
    
    detail: str = Field(
        ...,
        description="エラーの詳細情報"
    )
    
    error_code: Optional[str] = Field(
        default=None,
        description="エラーコード"
    )
    
    timestamp: Optional[str] = Field(
        default=None,
        description="エラー発生時刻（ISO 8601形式）"
    )
