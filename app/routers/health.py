"""
ヘルスチェック API ルーターモジュール

システムの稼働状態とリソース情報を提供するエンドポイントを定義します。
監視システムやロードバランサーからの健全性チェックに使用されます。
"""

import logging
import torch
from fastapi import APIRouter, status

from models.schemas import HealthCheckResponse
from config.settings import settings
from services.image_generation import image_generation_service
from services.task_manager import task_manager

# ログの設定
logger = logging.getLogger(__name__)

# ルーターの作成
router = APIRouter(
    prefix="/api/v1",
    tags=["ヘルスチェック"],
)


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="システム健全性チェック",
    description="""
    システムの稼働状態と主要コンポーネントの状態を確認します。
    
    **チェック項目:**
    - アプリケーションの基本動作
    - 機械学習モデルの読み込み状態
    - GPU/CPU の利用可能性
    - タスク処理システムの状態
    
    **使用用途:**
    - ロードバランサーのヘルスチェック
    - 監視システムでの稼働監視
    - デプロイ後の動作確認
    """
)
async def health_check() -> HealthCheckResponse:
    """
    システムのヘルスチェックを実行します
    
    Returns:
        HealthCheckResponse: システムの健全性情報
    """
    try:
        # 基本システム情報
        cuda_available = torch.cuda.is_available()
        device = settings.device
        
        # 画像生成サービスの状態確認
        model_loaded = image_generation_service.is_available
        
        # タスク統計の取得
        task_stats = task_manager.get_task_count()
        
        logger.info(
            f"ヘルスチェック実行 - "
            f"デバイス: {device}, "
            f"CUDA利用可能: {cuda_available}, "
            f"モデル読み込み済み: {model_loaded}, "
            f"アクティブタスク: {task_stats.get('IN_PROGRESS', 0)}"
        )
        
        return HealthCheckResponse(
            status="ok",
            device=device,
            cuda_available=cuda_available,
            ml_model_loaded=model_loaded
        )
        
    except Exception as e:
        logger.error(f"ヘルスチェックエラー: {e}")
        
        # エラーが発生してもシステムは動作しているとみなす
        # 重要なエラーの場合は、ここでstatusを"error"に設定
        return HealthCheckResponse(
            status="ok",
            device=settings.device,
            cuda_available=torch.cuda.is_available(),
            ml_model_loaded=False  # エラーが発生した場合はモデル未読み込みとする
        )


@router.get(
    "/stats",
    summary="システム統計情報",
    description="""
    システムの詳細な統計情報を取得します。
    
    **提供情報:**
    - タスク実行統計（状態別カウント）
    - システムリソース情報
    - 設定値の確認
    
    **注意:** この情報は管理者向けで、本番環境では適切なアクセス制御が必要です。
    """
)
async def get_system_stats():
    """
    システムの詳細統計情報を取得します
    
    Returns:
        dict: システム統計情報
    """
    try:
        # タスク統計
        task_stats = task_manager.get_task_count()
        
        # システム情報
        system_info = {
            "device": settings.device,
            "cuda_available": torch.cuda.is_available(),
            "model_id": settings.model_id,
            "lora_id": settings.lora_id,
            "default_settings": {
                "steps": settings.default_steps,
                "guidance": settings.default_guidance,
                "strength": settings.default_strength,
            },
            "image_settings": {
                "width": settings.target_width,
                "height": settings.target_height,
            }
        }
        
        # GPU情報（利用可能な場合）
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
            }
            
            # メモリ情報（エラーハンドリング付き）
            try:
                gpu_memory = torch.cuda.get_device_properties(0)
                gpu_info["total_memory_gb"] = round(gpu_memory.total_memory / (1024**3), 2)
            except Exception:
                gpu_info["total_memory_gb"] = "N/A"
        
        return {
            "status": "ok",
            "task_statistics": task_stats,
            "system_info": system_info,
            "gpu_info": gpu_info,
            "service_status": {
                "image_generation_service": image_generation_service.is_available,
                "task_manager": True,  # TaskManagerは基本的に常に利用可能
            }
        }
        
    except Exception as e:
        logger.error(f"統計情報取得エラー: {e}")
        return {
            "status": "error",
            "error": str(e),
            "basic_info": {
                "device": settings.device,
                "cuda_available": torch.cuda.is_available(),
            }
        }
