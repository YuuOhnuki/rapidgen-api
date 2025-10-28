"""
タスク管理サービスモジュール

非同期画像生成タスクの作成、実行、監視を管理します。
タスクの状態追跡とスレッド管理を含みます。
"""

import threading
import uuid
import logging
from typing import Dict, Optional, Callable
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import traceback

from models.schemas import ImageGenerationRequest, TaskStatus
from services.image_generation import image_generation_service

# ログの設定
logger = logging.getLogger(__name__)


class TaskInfo:
    """
    タスク情報を保持するクラス
    """
    
    def __init__(self, task_id: str, request: ImageGenerationRequest):
        self.task_id = task_id
        self.request = request
        self.status = TaskStatus.PENDING
        self.progress = 0
        self.result: Optional[str] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None


class TaskManager:
    """
    タスク管理クラス
    
    画像生成タスクの非同期実行と状態管理を行います。
    シングルトンパターンで実装され、アプリケーション全体で
    一つのインスタンスを共有します。
    """
    
    _instance: Optional['TaskManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'TaskManager':
        """
        シングルトンパターンの実装（スレッドセーフ）
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        初期化処理
        """
        if not hasattr(self, '_initialized'):
            self._tasks: Dict[str, TaskInfo] = {}
            self._executor = ThreadPoolExecutor(
                max_workers=2,  # 同時実行可能なタスク数（メモリ使用量を考慮）
                thread_name_prefix="ImageGen"
            )
            self._lock = threading.RLock()
            self._initialized = True
            logger.info("タスクマネージャーを初期化しました")
    
    def create_task(self, request: ImageGenerationRequest) -> str:
        """
        新しいタスクを作成し、実行キューに追加します
        
        Args:
            request: 画像生成リクエスト
            
        Returns:
            str: 生成されたタスクID
        """
        task_id = str(uuid.uuid4())
        
        with self._lock:
            task_info = TaskInfo(task_id, request)
            self._tasks[task_id] = task_info
        
        # 非同期でタスクを実行
        self._executor.submit(self._execute_task, task_id)
        
        logger.info(f"タスクを作成しました: {task_id}")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """
        指定されたタスクの状態を取得します
        
        Args:
            task_id: タスクID
            
        Returns:
            TaskInfo: タスク情報、存在しない場合はNone
        """
        with self._lock:
            return self._tasks.get(task_id)
    
    def _execute_task(self, task_id: str) -> None:
        """
        タスクの実行処理（内部メソッド）
        
        Args:
            task_id: 実行するタスクのID
        """
        with self._lock:
            task_info = self._tasks.get(task_id)
            if not task_info:
                logger.error(f"タスクが見つかりません: {task_id}")
                return
        
        try:
            logger.info(f"タスク実行を開始: {task_id}")
            
            # タスクの状態を実行中に変更
            self._update_task_status(task_id, TaskStatus.IN_PROGRESS, 0)
            task_info.started_at = datetime.now(timezone.utc)
            
            # 進捗コールバック関数
            def progress_callback(current_step: int, total_steps: int) -> None:
                # 進捗を20%から90%の間で計算（前後処理分を考慮）
                progress = 20 + int(((current_step + 1) / max(1, total_steps)) * 70)
                progress = min(progress, 90)  # 最大90%まで
                self._update_task_progress(task_id, progress)
            
            # 前処理進捗の更新
            self._update_task_progress(task_id, 10)
            
            # 画像生成の実行
            result = image_generation_service.generate_image(
                task_info.request,
                progress_callback=progress_callback
            )
            
            # 後処理進捗の更新
            self._update_task_progress(task_id, 95)
            
            # タスク完了
            self._complete_task(task_id, result)
            logger.info(f"タスクが完了しました: {task_id}")
            
        except Exception as e:
            error_msg = f"タスク実行エラー: {str(e)}"
            logger.error(f"{error_msg} (Task ID: {task_id})")
            logger.error(f"詳細なエラー情報: {traceback.format_exc()}")
            
            self._fail_task(task_id, error_msg)
    
    def _update_task_status(self, task_id: str, status: TaskStatus, progress: int) -> None:
        """
        タスクの状態を更新します（内部メソッド）
        
        Args:
            task_id: タスクID
            status: 新しい状態
            progress: 進捗率
        """
        with self._lock:
            task_info = self._tasks.get(task_id)
            if task_info:
                task_info.status = status
                task_info.progress = progress
    
    def _update_task_progress(self, task_id: str, progress: int) -> None:
        """
        タスクの進捗のみを更新します（内部メソッド）
        
        Args:
            task_id: タスクID
            progress: 進捗率
        """
        with self._lock:
            task_info = self._tasks.get(task_id)
            if task_info:
                task_info.progress = min(progress, 100)
    
    def _complete_task(self, task_id: str, result: str) -> None:
        """
        タスクを完了状態にします（内部メソッド）
        
        Args:
            task_id: タスクID
            result: 生成結果
        """
        with self._lock:
            task_info = self._tasks.get(task_id)
            if task_info:
                task_info.status = TaskStatus.COMPLETED
                task_info.progress = 100
                task_info.result = result
                task_info.completed_at = datetime.now(timezone.utc)
    
    def _fail_task(self, task_id: str, error_message: str) -> None:
        """
        タスクを失敗状態にします（内部メソッド）
        
        Args:
            task_id: タスクID
            error_message: エラーメッセージ
        """
        with self._lock:
            task_info = self._tasks.get(task_id)
            if task_info:
                task_info.status = TaskStatus.FAILED
                task_info.error = error_message
                task_info.completed_at = datetime.now(timezone.utc)
    
    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """
        古いタスクをクリーンアップします
        
        Args:
            max_age_hours: 保持する最大時間（デフォルト24時間）
            
        Returns:
            int: 削除されたタスク数
        """
        cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        removed_count = 0
        
        with self._lock:
            tasks_to_remove = []
            
            for task_id, task_info in self._tasks.items():
                # 完了または失敗したタスクで、指定時間を過ぎたもの
                if (task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] and
                    task_info.created_at.timestamp() < cutoff_time):
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self._tasks[task_id]
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"古いタスクを{removed_count}個削除しました")
        
        return removed_count
    
    def get_task_count(self) -> Dict[str, int]:
        """
        状態別のタスク数を取得します
        
        Returns:
            Dict[str, int]: 状態別のタスク数
        """
        with self._lock:
            counts = {status.value: 0 for status in TaskStatus}
            
            for task_info in self._tasks.values():
                counts[task_info.status.value] += 1
            
            counts['total'] = len(self._tasks)
        
        return counts
    
    def shutdown(self) -> None:
        """
        タスクマネージャーをシャットダウンします
        """
        logger.info("タスクマネージャーをシャットダウン中...")
        self._executor.shutdown(wait=True)
        logger.info("タスクマネージャーのシャットダウンが完了しました")


# タスクマネージャーインスタンスの作成（シングルトン）
task_manager = TaskManager()
