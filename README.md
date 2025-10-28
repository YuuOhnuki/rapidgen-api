# Stable Diffusion XL LCM API

Stable Diffusion XL (SDXL) + Latent Consistency Model (LCM) を使用した高品質な img2img 画像生成 API です。

## 🌟 主な機能

-   **高速画像生成**: LCM により少ないステップで高品質な画像を生成
-   **非同期処理**: 長時間の生成処理を背景で実行し、進捗を追跡可能
-   **柔軟なパラメータ**: プロンプト、強度、ガイダンス等を細かく調整
-   **GPU 最適化**: CUDA が利用可能な環境でのメモリ効率的な処理
-   **モジュラー設計**: 保守性と拡張性を重視した構造

## 🏗️ プロジェクト構造

```
rapidgen-py/
├── app/                    # FastAPIアプリケーション
│   ├── main.py            # メインアプリケーション
│   └── routers/           # APIルーター
│       ├── image_generation.py
│       └── health.py
├── config/                # 設定管理
│   └── settings.py        # アプリケーション設定
├── models/                # データモデル
│   └── schemas.py         # Pydanticモデル定義
├── services/              # ビジネスロジック
│   ├── image_generation.py # 画像生成サービス
│   └── task_manager.py    # タスク管理サービス
├── utils/                 # ユーティリティ
│   ├── exceptions.py      # カスタム例外
│   └── logging_config.py  # ログ設定
├── requirements.txt       # 依存関係
└── README.md             # このファイル
```

## 🚀 セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定（オプション）

```bash
# モデル設定
export SDXL_MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0"
export SDXL_LORA_ID="latent-consistency/lcm-lora-sdxl"

# デフォルト生成パラメータ
export SDXL_DEFAULT_STEPS=20
export SDXL_DEFAULT_GUIDANCE=1.5
export SDXL_DEFAULT_STRENGTH=0.7

# デバッグモード
export SDXL_DEBUG=true
```

### 3. アプリケーションの起動

```bash
# 開発環境
python -m app.main

# または uvicorn を直接使用
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📚 API 使用方法

### 1. 画像生成タスクの作成

```bash
curl -X POST "http://localhost:8000/api/v1/generate/" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "a beautiful landscape with mountains and a lake",
       "init_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
       "num_inference_steps": 20,
       "guidance_scale": 1.5,
       "strength": 0.7,
       "negative_prompt": "blurry, low quality"
     }'
```

レスポンス:

```json
{
	"task_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

### 2. タスク状態の確認

```bash
curl "http://localhost:8000/api/v1/generate/tasks/123e4567-e89b-12d3-a456-426614174000"
```

実行中:

```json
{
	"status": "IN_PROGRESS",
	"progress": 75
}
```

完了時:

```json
{
	"status": "COMPLETED",
	"progress": 100,
	"dataUrl": "data:image/png;base64,..."
}
```

### 3. ヘルスチェック

```bash
curl "http://localhost:8000/api/v1/health"
```

```json
{
	"status": "ok",
	"device": "cuda",
	"cuda_available": true,
	"model_loaded": true
}
```

## 🛠️ 設定項目

| 環境変数                | デフォルト値                               | 説明                         |
| ----------------------- | ------------------------------------------ | ---------------------------- |
| `SDXL_MODEL_ID`         | `stabilityai/stable-diffusion-xl-base-1.0` | 使用する SDXL モデル         |
| `SDXL_LORA_ID`          | `latent-consistency/lcm-lora-sdxl`         | 使用する LCM LoRA            |
| `SDXL_DEFAULT_STEPS`    | `20`                                       | デフォルト推論ステップ数     |
| `SDXL_DEFAULT_GUIDANCE` | `1.5`                                      | デフォルトガイダンススケール |
| `SDXL_DEFAULT_STRENGTH` | `0.7`                                      | デフォルト Img2Img 強度      |
| `SDXL_TARGET_WIDTH`     | `1024`                                     | 出力画像の幅                 |
| `SDXL_TARGET_HEIGHT`    | `1024`                                     | 出力画像の高さ               |
| `SDXL_DEBUG`            | `false`                                    | デバッグモード               |

## 🔧 開発者向け情報

### コード品質

-   **型ヒント**: 全ての関数で適切な型アノテーション
-   **Pydantic モデル**: 厳密なデータ検証
-   **包括的なログ**: 構造化されたログ出力
-   **エラーハンドリング**: カスタム例外とグレースフルな処理
-   **依存性注入**: モジュラー設計とテスタビリティ

### アーキテクチャ

-   **サービス層**: ビジネスロジックの分離
-   **ルーター層**: API エンドポイントの整理
-   **設定管理**: 環境変数による設定の外部化
-   **非同期処理**: ThreadPoolExecutor による効率的なタスク管理

### テスト実行

```bash
# APIドキュメントの確認
http://localhost:8000/docs

# システム統計情報
http://localhost:8000/api/v1/stats
```

## 📄 ライセンス

MIT License

## 🤝 コントリビューション

1. フォークして機能ブランチを作成
2. 変更をコミット
3. プルリクエストを送信

## 📞 サポート

問題や質問がある場合は、Issue を作成してください。

---

**注意:** この API は GPU リソースを集約的に使用します。本番環境では適切なリソース管理とモニタリングを実装してください。
