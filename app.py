from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import io
import base64

app = FastAPI()

# モデルのロード
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe.to("cuda")

class ImageRequest(BaseModel):
    prompt: str
    init_image: str  # base64エンコードされた画像データ

@app.post("/generate/")
async def generate_image(request: ImageRequest):
    try:
        # base64データのデコード
        init_image_data = base64.b64decode(request.init_image)
        init_image = Image.open(io.BytesIO(init_image_data)).convert("RGB")

        # 画像生成
        generated_image = pipe(prompt=request.prompt, init_image=init_image).images[0]

        # 生成された画像をbase64エンコード
        buffered = io.BytesIO()
        generated_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"generated_image": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
