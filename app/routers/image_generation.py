"""
画像生成API ルーターモジュール

img2img画像生成に関連するAPIエンドポイントを定義します。
非同期タスクの作成と状態確認のエンドポイントを提供します。
"""

import logging
from typing import Union
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from models.schemas import (
    ImageGenerationRequest,
    TaskCreationResponse,
    TaskStatusResponse,
    TaskCompletedResponse,
    TaskStatus,
    ErrorResponse
)
from services.task_manager import task_manager

# ログの設定
logger = logging.getLogger(__name__)

# ルーターの作成
router = APIRouter(
    prefix="/api/v1/generate",
    tags=["画像生成"],
    responses={
        400: {"model": ErrorResponse, "description": "不正なリクエスト"},
        404: {"model": ErrorResponse, "description": "タスクが見つかりません"},
        500: {"model": ErrorResponse, "description": "内部サーバーエラー"},
    }
)


@router.post(
    "/",
    response_model=TaskCreationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="画像生成タスクの作成",
    description="""
    Stable Diffusion XL (SDXL) + LCM を使用したimg2img画像生成タスクを作成します。
    
    **処理の流れ:**
    1. リクエストパラメータの検証
    2. 非同期タスクの作成とキューへの追加
    3. タスクIDを即座に返却
    4. バックグラウンドで画像生成を実行
    
    **注意事項:**
    - `init_image`は有効なBase64エンコードされた画像データである必要があります
    - 生成には時間がかかるため、返されたタスクIDで進捗を確認してください
    """
)
async def create_generation_task(request: ImageGenerationRequest) -> TaskCreationResponse:
    """
    画像生成タスクを作成します
    
    Args:
        request: 画像生成リクエストパラメータ
        
    Returns:
        TaskCreationResponse: 作成されたタスクのID
        
    Raises:
        HTTPException: リクエストが無効な場合（400）
        HTTPException: サービスが利用できない場合（503）
    """
    try:
        # 基本的なバリデーション
        if not request.prompt or not request.prompt.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="プロンプトは必須です"
            )
        
        if not request.init_image:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="元画像（init_image）は必須です"
            )
        
        logger.info(f"画像生成タスクを作成中 - プロンプト: {request.prompt[:50]}...")
        
        # タスクの作成
        task_id = task_manager.create_task(request)
        
        logger.info(f"タスクが作成されました: {task_id}")
        
        return TaskCreationResponse(task_id=task_id)
        
    except HTTPException:
        # FastAPI HTTPExceptionはそのまま再発生
        raise
    except Exception as e:
        logger.error(f"タスク作成エラー: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="タスクの作成に失敗しました"
        )


@router.get(
    "/tasks/{task_id}",
    response_model=Union[TaskStatusResponse, TaskCompletedResponse],
    summary="タスク状態の確認",
    description="""
    指定されたタスクIDの実行状態と進捗を確認します。
    
    **返却される状態:**
    - `PENDING`: 実行待機中
    - `IN_PROGRESS`: 実行中（progress フィールドで進捗確認可能）
    - `COMPLETED`: 完了（dataUrl フィールドで結果画像を取得可能）
    - `FAILED`: 失敗（エラー詳細がHTTPエラーレスポンスで返却）
    
    **ポーリング推奨間隔:**
    - 実行中: 1-2秒間隔
    - 完了後: ポーリング停止
    """
)
async def get_task_status(task_id: str) -> Union[TaskStatusResponse, TaskCompletedResponse]:
    """
    タスクの実行状態を取得します
    
    Args:
        task_id: 確認するタスクのID
        
    Returns:
        Union[TaskStatusResponse, TaskCompletedResponse]: タスクの状態情報
        
    Raises:
        HTTPException: タスクが見つからない場合（404）
        HTTPException: タスクが失敗した場合（500）
    """
    try:
        # タスク情報の取得
        task_info = task_manager.get_task_status(task_id)
        
        if not task_info:
            logger.warning(f"存在しないタスクIDが指定されました: {task_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="指定されたタスクが見つかりません"
            )
        
        # 失敗した場合の処理
        if task_info.status == TaskStatus.FAILED:
            error_detail = task_info.error or "不明なエラーが発生しました"
            logger.error(f"タスク失敗: {task_id} - {error_detail}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"タスクが失敗しました: {error_detail}"
            )
        
        # 完了した場合の処理
        if task_info.status == TaskStatus.COMPLETED:
            if not task_info.result:
                logger.error(f"完了タスクの結果が存在しません: {task_id}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="タスクは完了しましたが、結果データが見つかりません"
                )
            
            logger.info(f"タスク完了結果を返却: {task_id}")
            return TaskCompletedResponse(
                status=task_info.status,
                progress=task_info.progress,
                dataUrl=task_info.result
            )
        
        # 実行中または待機中の場合
        return TaskStatusResponse(
            status=task_info.status,
            progress=task_info.progress
        )
        
    except HTTPException:
        # FastAPI HTTPExceptionはそのまま再発生
        raise
    except Exception as e:
        logger.error(f"タスク状態取得エラー: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="タスク状態の取得に失敗しました"
        )
