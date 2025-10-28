"""
カスタム例外クラス定義

アプリケーション固有のエラー処理とエラー分類を提供します。
各種エラー状況に対応した専用例外クラスを定義します。
"""

from typing import Optional, Dict, Any


class BaseImageGenerationError(Exception):
    """
    画像生成関連エラーのベースクラス
    
    すべての画像生成関連エラーの基底となるクラスです。
    エラー分類とログ記録の統一化を図ります。
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ModelLoadError(BaseImageGenerationError):
    """
    機械学習モデルのロード失敗エラー
    
    SDXLモデルやLoRAの読み込みに失敗した場合に発生します。
    """
    
    def __init__(self, model_name: str, original_error: str):
        super().__init__(
            f"モデルの読み込みに失敗しました: {model_name}",
            error_code="MODEL_LOAD_FAILED",
            details={"model_name": model_name, "original_error": original_error}
        )


class ImageProcessingError(BaseImageGenerationError):
    """
    画像処理エラー
    
    画像のデコード、リサイズ、エンコード処理で発生するエラーです。
    """
    
    def __init__(self, operation: str, original_error: str):
        super().__init__(
            f"画像処理エラー ({operation}): {original_error}",
            error_code="IMAGE_PROCESSING_FAILED",
            details={"operation": operation, "original_error": original_error}
        )


class GenerationError(BaseImageGenerationError):
    """
    画像生成処理エラー
    
    Diffusersパイプラインでの実際の画像生成処理で発生するエラーです。
    """
    
    def __init__(self, original_error: str, generation_params: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"画像生成に失敗しました: {original_error}",
            error_code="GENERATION_FAILED",
            details={"original_error": original_error, "generation_params": generation_params}
        )


class ValidationError(BaseImageGenerationError):
    """
    入力パラメータの検証エラー
    
    リクエストパラメータの形式や値が不正な場合に発生します。
    """
    
    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            f"パラメータ検証エラー ({field}): {reason}",
            error_code="VALIDATION_FAILED",
            details={"field": field, "value": str(value), "reason": reason}
        )


class ResourceExhaustionError(BaseImageGenerationError):
    """
    リソース不足エラー
    
    GPU メモリ不足やシステムリソース不足で発生するエラーです。
    """
    
    def __init__(self, resource_type: str, original_error: str):
        super().__init__(
            f"リソース不足エラー ({resource_type}): {original_error}",
            error_code="RESOURCE_EXHAUSTED",
            details={"resource_type": resource_type, "original_error": original_error}
        )


class TaskError(BaseImageGenerationError):
    """
    タスク実行エラー
    
    非同期タスクの実行や管理で発生するエラーです。
    """
    
    def __init__(self, task_id: str, operation: str, original_error: str):
        super().__init__(
            f"タスクエラー ({operation}): {original_error}",
            error_code="TASK_FAILED",
            details={"task_id": task_id, "operation": operation, "original_error": original_error}
        )


class ServiceUnavailableError(BaseImageGenerationError):
    """
    サービス利用不可エラー
    
    サービスが初期化されていない、またはシャットダウン中の場合に発生します。
    """
    
    def __init__(self, service_name: str, reason: str):
        super().__init__(
            f"サービス利用不可 ({service_name}): {reason}",
            error_code="SERVICE_UNAVAILABLE",
            details={"service_name": service_name, "reason": reason}
        )
