"""
画像生成サービスモジュール

Stable Diffusion XL (SDXL) + LCM を使用したimg2img画像生成の
コアロジックを実装します。
"""

import base64
import io
import logging
from typing import Optional, Callable, Any
from PIL import Image
import torch

from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    LCMScheduler,
)

from config.settings import settings
from models.schemas import ImageGenerationRequest

# ログの設定
logger = logging.getLogger(__name__)


class ImageGenerationService:
    """
    画像生成サービスクラス
    
    SDXL + LCMパイプラインを管理し、img2img生成を実行します。
    シングルトンパターンで実装され、アプリケーション起動時に
    一度だけモデルをロードします。
    """
    
    _instance: Optional['ImageGenerationService'] = None
    _pipeline: Optional[StableDiffusionXLImg2ImgPipeline] = None
    _is_initialized: bool = False
    
    def __new__(cls) -> 'ImageGenerationService':
        """
        シングルトンパターンの実装
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        初期化処理（一度だけ実行される）
        """
        if not self._is_initialized:
            self._initialize_pipeline()
            self._is_initialized = True
    
    def _initialize_pipeline(self) -> None:
        """
        Diffusersパイプラインの初期化
        
        SDXLモデルとLCM LoRAをロードし、最適化設定を適用します。
        """
        logger.info(f"画像生成パイプラインを初期化中... デバイス: {settings.device}")
        
        try:
            # メインモデルのロード
            self._pipeline = self._load_base_model()
            
            # LCMスケジューラーの設定
            self._setup_lcm_scheduler()
            
            # LoRAの適用
            self._load_lora_weights()
            
            # デバイスへの移動
            self._pipeline = self._pipeline.to(settings.device)
            
            # メモリ最適化の適用
            self._apply_memory_optimizations()
            
            logger.info("画像生成パイプラインの初期化が完了しました")
            
        except Exception as e:
            logger.error(f"パイプライン初期化エラー: {e}")
            raise RuntimeError(f"画像生成サービスの初期化に失敗しました: {e}")
    
    def _load_base_model(self) -> StableDiffusionXLImg2ImgPipeline:
        """
        ベースモデルのロード
        
        Returns:
            StableDiffusionXLImg2ImgPipeline: ロードされたパイプライン
        """
        try:
            # 最初にfp16バリアントでの読み込みを試行
            return StableDiffusionXLImg2ImgPipeline.from_pretrained(
                settings.model_id,
                dtype=settings.torch_dtype,
                variant=settings.variant,
            )
        except Exception as e:
            logger.warning(f"fp16バリアントでの読み込みに失敗: {e}")
            logger.info("fp32でのフォールバック読み込みを実行中...")
            
            # フォールバック: バリアント指定なしで再試行
            return StableDiffusionXLImg2ImgPipeline.from_pretrained(
                settings.model_id,
                dtype=settings.torch_dtype,
            )
    
    def _setup_lcm_scheduler(self) -> None:
        """
        LCM（Latent Consistency Model）スケジューラーの設定
        """
        try:
            self._pipeline.scheduler = LCMScheduler.from_config(
                self._pipeline.scheduler.config
            )
            logger.info("LCMスケジューラーを設定しました")
        except Exception as e:
            logger.warning(f"LCMスケジューラーの設定に失敗: {e}")
            logger.info("元のスケジューラーを継続使用します")
    
    def _load_lora_weights(self) -> None:
        """
        LoRA重みの読み込み
        """
        try:
            self._pipeline.load_lora_weights(settings.lora_id)
            logger.info(f"LoRA重みを読み込みました: {settings.lora_id}")
        except Exception as e:
            logger.warning(f"LoRA重みの読み込みに失敗: {e}")
            logger.info("LoRAなしで続行します")
    
    def _apply_memory_optimizations(self) -> None:
        """
        メモリ最適化設定の適用
        """
        if settings.device == "cuda":
            if settings.enable_cpu_offload:
                self._pipeline.enable_model_cpu_offload()
                logger.info("CPUオフロードを有効化しました")
            
            if settings.enable_attention_slicing:
                self._pipeline.enable_attention_slicing()
                logger.info("アテンション・スライシングを有効化しました")
    
    def _preprocess_image(self, base64_image: str) -> Image.Image:
        """
        入力画像の前処理
        
        Args:
            base64_image: Base64エンコードされた画像データ
            
        Returns:
            Image.Image: 前処理された画像
        """
        try:
            # Base64デコード
            image_bytes = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # 指定サイズにリサイズ
            target_size = (settings.target_width, settings.target_height)
            
            # 高品質リサンプリングを使用
            try:
                image = image.resize(target_size, resample=Image.Resampling.LANCZOS)
            except AttributeError:
                # 古いPillowバージョン向けのフォールバック
                image = image.resize(target_size, Image.LANCZOS)
            
            return image
            
        except Exception as e:
            raise ValueError(f"画像の前処理に失敗しました: {e}")
    
    def _postprocess_image(self, image: Image.Image) -> str:
        """
        出力画像の後処理
        
        Args:
            image: 生成された画像
            
        Returns:
            str: Base64エンコードされた画像データ
        """
        try:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG", optimize=True)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"画像の後処理に失敗しました: {e}")
    
    def generate_image(
        self,
        request: ImageGenerationRequest,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """
        画像生成の実行
        
        Args:
            request: 画像生成リクエスト
            progress_callback: 進捗コールバック関数（オプション）
            
        Returns:
            str: 生成された画像のBase64エンコードデータ
            
        Raises:
            ValueError: 入力データが無効な場合
            RuntimeError: 生成処理でエラーが発生した場合
        """
        if not self._pipeline:
            raise RuntimeError("画像生成パイプラインが初期化されていません")
        
        logger.info("画像生成を開始します")
        
        try:
            # 前処理: 入力画像の準備
            init_image = self._preprocess_image(request.init_image)
            
            # デフォルト値の適用
            num_steps = request.num_inference_steps or settings.default_steps
            guidance = request.guidance_scale or settings.default_guidance
            strength = request.strength or settings.default_strength
            
            # 進捗コールバックの設定
            def pipeline_callback(step: int, timestep: int, latents: Any) -> None:
                if progress_callback:
                    progress_callback(step, num_steps)
            
            # 画像生成の実行
            logger.info(f"生成パラメータ - Steps: {num_steps}, Guidance: {guidance}, Strength: {strength}")
            
            output = self._pipeline(
                prompt=request.prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                negative_prompt=request.negative_prompt,
                callback=pipeline_callback,
                callback_steps=1,
            )
            
            if not hasattr(output, "images") or not output.images:
                raise RuntimeError("画像生成パイプラインが画像を返しませんでした")
            
            # 後処理: Base64エンコード
            result_image = output.images[0]
            encoded_image = self._postprocess_image(result_image)
            
            logger.info("画像生成が完了しました")
            return encoded_image
            
        except Exception as e:
            logger.error(f"画像生成エラー: {e}")
            raise RuntimeError(f"画像生成に失敗しました: {e}")
    
    @property
    def is_available(self) -> bool:
        """
        サービスが利用可能かどうかを確認
        
        Returns:
            bool: サービスが利用可能な場合True
        """
        return self._pipeline is not None and self._is_initialized


# サービスインスタンスの作成（シングルトン）
image_generation_service = ImageGenerationService()