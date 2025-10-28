"""
ログ設定ユーティリティ

アプリケーション全体で統一されたログ設定を提供します。
開発環境と本番環境で適切なログレベルとフォーマットを適用します。
"""

import logging
import sys
from typing import Dict, Any
from config.settings import settings


def setup_logging() -> None:
    """
    アプリケーションのログ設定を初期化します
    
    設定内容:
    - 統一されたログフォーマット
    - 適切なログレベル
    - コンソール出力の設定
    """
    
    # ログフォーマットの定義
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # ログレベルの決定
    log_level = logging.DEBUG if settings.debug else logging.INFO
    
    # 基本設定
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        stream=sys.stdout,
        force=True  # 既存の設定を上書き
    )
    
    # 特定のライブラリのログレベルを調整
    _configure_library_loggers()
    
    # アプリケーション開始ログ
    logger = logging.getLogger(__name__)
    logger.info(f"ログシステムを初期化しました (レベル: {logging.getLevelName(log_level)})")


def _configure_library_loggers() -> None:
    """
    外部ライブラリのログレベルを調整します
    """
    library_log_levels: Dict[str, int] = {
        # Diffusersのログを制限（大量のログを防ぐ）
        "diffusers": logging.WARNING,
        "transformers": logging.WARNING,
        
        # HTTP関連のログを制限
        "urllib3": logging.WARNING,
        "requests": logging.WARNING,
        
        # PIL/Pillow のログを制限
        "PIL": logging.WARNING,
        
        # FastAPIのアクセスログ（開発時は詳細、本番時は制限）
        "uvicorn.access": logging.INFO if settings.debug else logging.WARNING,
    }
    
    for logger_name, level in library_log_levels.items():
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    指定された名前のロガーインスタンスを取得します
    
    Args:
        name: ロガー名（通常は __name__ を渡す）
        
    Returns:
        logging.Logger: 設定済みのロガーインスタンス
    """
    return logging.getLogger(name)
