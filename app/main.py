"""
メインアプリケーションモジュール

Stable Diffusion XL (SDXL) + LCM を使用したimg2img生成APIの
FastAPIアプリケーションエントリーポイントです。

アプリケーション構成:
- モジュラー設計による保守性の向上
- 適切な依存性注入とサービス分離
- 包括的なエラーハンドリング
- 非同期タスク処理
- 詳細なログ記録
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from utils.logging_config import setup_logging
from utils.exceptions import BaseImageGenerationError
from app.routers import image_generation, health
from services.task_manager import task_manager

# ログ設定の初期化
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    アプリケーションのライフサイクル管理
    
    起動時と終了時の処理を定義します:
    - 起動時: サービスの初期化とヘルスチェック
    - 終了時: リソースのクリーンアップ
    """
    # === 起動時処理 ===
    logger.info("=" * 60)
    logger.info("Stable Diffusion XL LCM API を起動中...")
    logger.info(f"デバイス: {settings.device}")
    logger.info(f"モデル: {settings.model_id}")
    logger.info(f"LoRA: {settings.lora_id}")
    logger.info("=" * 60)
    
    try:
        # 画像生成サービスの初期化確認
        from services.image_generation import image_generation_service
        if not image_generation_service.is_available:
            logger.error("画像生成サービスの初期化に失敗しました")
            raise RuntimeError("重要なサービスの初期化に失敗しました")
        
        logger.info("すべてのサービスが正常に初期化されました")
        logger.info("API サーバーが利用可能になりました")
        
    except Exception as e:
        logger.error(f"アプリケーション起動エラー: {e}")
        raise
    
    # アプリケーション実行
    yield
    
    # === 終了時処理 ===
    logger.info("アプリケーションを終了中...")
    
    try:
        # タスクマネージャーのシャットダウン
        task_manager.shutdown()
        
        # 古いタスクのクリーンアップ
        cleaned_count = task_manager.cleanup_old_tasks(max_age_hours=1)
        if cleaned_count > 0:
            logger.info(f"終了時に {cleaned_count} 個の古いタスクをクリーンアップしました")
        
        logger.info("アプリケーションが正常に終了しました")
        
    except Exception as e:
        logger.error(f"終了時エラー: {e}")


# FastAPIアプリケーションインスタンスの作成
app = FastAPI(
    title=settings.app_title,
    description="""
    Stable Diffusion XL (SDXL) + Latent Consistency Model (LCM) を使用した
    高品質なimg2img画像生成APIです。
    
    ## 主な機能
    
    * **高速画像生成**: LCMにより少ないステップで高品質な画像を生成
    * **非同期処理**: 長時間の生成処理を背景で実行し、進捗を追跡可能
    * **柔軟なパラメータ**: プロンプト、強度、ガイダンス等を細かく調整
    * **GPU最適化**: CUDAが利用可能な環境でのメモリ効率的な処理
    
    ## 使用方法
    
    1. `/api/v1/generate/` エンドポイントで画像生成タスクを作成
    2. 返されたタスクIDで `/api/v1/generate/tasks/{task_id}` から進捗確認
    3. 完了後、同じエンドポイントから生成画像を取得
    
    ## 技術仕様
    
    * **モデル**: Stable Diffusion XL + LCM LoRA
    * **入力形式**: Base64エンコードされた画像データ
    * **出力形式**: Base64エンコードされたPNG画像
    * **最大解像度**: 1024x1024px (設定可能)
    """,
    version="1.0.0",
    contact={
        "name": "API サポート",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    debug=settings.debug,
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切なオリジンを設定
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ルーターの登録
app.include_router(
    image_generation.router,
    tags=["画像生成"]
)

app.include_router(
    health.router,
    tags=["システム"]
)


# === グローバル例外ハンドラー ===

@app.exception_handler(BaseImageGenerationError)
async def handle_image_generation_error(
    request: Request, 
    exc: BaseImageGenerationError
) -> JSONResponse:
    """
    画像生成関連エラーのハンドラー
    
    カスタム例外を適切なHTTPレスポンスに変換します。
    """
    logger.error(
        f"画像生成エラー: {exc.message} "
        f"(コード: {exc.error_code}, 詳細: {exc.details})"
    )
    
    # エラーコードに応じてHTTPステータスを決定
    status_code = 500  # デフォルト
    if exc.error_code in ["VALIDATION_FAILED"]:
        status_code = 400
    elif exc.error_code in ["SERVICE_UNAVAILABLE"]:
        status_code = 503
    elif exc.error_code in ["RESOURCE_EXHAUSTED"]:
        status_code = 429
    
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": exc.message,
            "error_code": exc.error_code,
            "timestamp": exc.details.get("timestamp") if exc.details else None
        }
    )


@app.exception_handler(HTTPException)
async def handle_http_exception(
    request: Request, 
    exc: HTTPException
) -> JSONResponse:
    """
    FastAPI HTTPExceptionのハンドラー
    
    標準的なHTTP例外のログ記録を行います。
    """
    logger.warning(
        f"HTTP例外: {exc.status_code} - {exc.detail} "
        f"(パス: {request.url.path})"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(500)
async def handle_internal_server_error(
    request: Request, 
    exc: Exception
) -> JSONResponse:
    """
    予期しない内部サーバーエラーのハンドラー
    """
    logger.error(
        f"予期しないエラー: {str(exc)} (パス: {request.url.path})",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "内部サーバーエラーが発生しました",
            "error_code": "INTERNAL_SERVER_ERROR"
        }
    )


# === 追加エンドポイント ===

@app.get(
    "/",
    summary="API情報",
    description="APIの基本情報とドキュメントリンクを提供します。"
)
async def root():
    """
    ルートエンドポイント - API の基本情報を返します
    """
    return {
        "name": settings.app_title,
        "version": "1.0.0",
        "description": "Stable Diffusion XL + LCM による高速img2img生成API",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/api/v1/health",
        "device": settings.device,
        "cuda_available": settings.device == "cuda"
    }


# === デバッグ情報 ===
if settings.debug:
    logger.info("デバッグモードが有効です")
    logger.info(f"設定値: {settings.dict()}")


# === アプリケーション実行 ===
if __name__ == "__main__":
    import uvicorn
    
    # 開発サーバーの起動
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info"
    )
