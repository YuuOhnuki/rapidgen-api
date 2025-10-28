import torch # torchはCPU環境でも使用
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import io
import base64
import uuid
import time
from diffusers import StableDiffusionImg2ImgPipeline, LCMScheduler

# モデルとLoRAの定義
MODEL_ID = "runwayml/stable-diffusion-v1-5" # 軽量なベースモデル
LORA_ID = "latent-consistency/lcm-lora-sdv1-5" # 高速化LoRA

# CPU環境専用でモデルロード
# torch_dtype=torch.float32 はCPU環境でのデフォルト精度
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_ID, 
    dtype=torch.float32  # 修正: torch_dtype を dtype に変更
)

# スケジューラをLCM対応のものに変更
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# 高速化LoRAをロード
# LCM-LoRAは、少ないステップ数で高品質な画像を生成可能にする技術
pipe.load_lora_weights(LORA_ID)

# パイプライン全体をCPUに配置（デフォルトでCPUですが明示的に）
pipe.to("cpu")

app = FastAPI(title="Stable Diffusion LCM CPU API")

class ImageRequest(BaseModel):
    prompt: str
    init_image: str
    num_inference_steps: int = 4

# グローバルなタスク状態管理（実運用ではRedisなどを使用すべき）
# {task_id: {status: 'PENDING' | 'IN_PROGRESS' | 'COMPLETED' | 'FAILED', result: str | None, progress: int}}
task_statuses = {}

# 非同期タスクのシミュレーション関数
def generate_image_task(task_id: str, request: ImageRequest):
    """画像生成の重い処理をシミュレート"""
    try:
        task_statuses[task_id]['status'] = 'IN_PROGRESS'
        task_statuses[task_id]['progress'] = 10

        # 1. Base64 → PIL画像
        init_image_data = base64.b64decode(request.init_image)
        init_image = Image.open(io.BytesIO(init_image_data)).convert("RGB")
        init_image = init_image.resize((512, 512))

        # 進捗シミュレーション
        for i in range(1, request.num_inference_steps + 1):
            time.sleep(1) # CPU処理の遅延をシミュレート
            task_statuses[task_id]['progress'] = 10 + int(80 * (i / request.num_inference_steps))
            # 擬似的に進捗メッセージも更新可能だが、ここではシンプルに進捗率のみ

        # 2. 画像生成 (実際の処理)
        generated_image = pipe(
            prompt=request.prompt,
            image=init_image,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=1.0
        ).images[0]

        # 3. PIL画像 → Base64
        buffered = io.BytesIO()
        generated_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        task_statuses[task_id]['status'] = 'COMPLETED'
        task_statuses[task_id]['progress'] = 100
        task_statuses[task_id]['result'] = img_str

    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        task_statuses[task_id]['status'] = 'FAILED'
        task_statuses[task_id]['result'] = str(e)


# 1. タスク開始エンドポイント
@app.post("/generate/")
async def create_generate_task(request: ImageRequest):
    """画像生成タスクを開始し、タスクIDを返却"""
    task_id = str(uuid.uuid4())
    task_statuses[task_id] = {
        'status': 'PENDING',
        'result': None,
        'progress': 0
    }
    
    # 実際にはここでceleryなどの非同期ワーカーにタスクを渡す
    # 簡単のため、ここではスレッドで非同期実行をシミュレート
    import threading
    threading.Thread(target=generate_image_task, args=(task_id, request)).start()
    
    return {"task_id": task_id}

# 2. ポーリングエンドポイント
@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """タスクの状態と結果を取得"""
    if task_id not in task_statuses:
        raise HTTPException(status_code=404, detail="Task not found")

    status_data = task_statuses[task_id]
    
    if status_data['status'] == 'FAILED':
        # エラー時は詳細を返却
        raise HTTPException(status_code=500, detail=f"Task failed: {status_data['result']}")
    
    if status_data['status'] == 'COMPLETED':
        # 完了時は結果（Base64画像データ）を返却
        return {
            "status": "COMPLETED",
            "progress": 100,
            "dataUrl": status_data['result'] # resultがBase64画像データ
        }
    
    # 処理中の場合はステータスと進捗を返却
    return {
        "status": status_data['status'],
        "progress": status_data['progress']
    }

# FastAPIは非同期処理を扱うことができますが、AIモデルの推論はCPUバウンドなため、
# 実際にはGunicorn + Uvicornと組み合わせてワーカーを増やすか、
# Celeryなどのタスクキューイングシステムで非同期ワーカーを構築する必要があります。