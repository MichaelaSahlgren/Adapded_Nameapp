from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io

# EB looks for this symbol:
application = FastAPI()

@application.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <body>
            <h2>Upload an Image for Name Classification</h2>
            <form action="/classify/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """

# --- CLIP setup copied from your offline script ---
model_name = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_name).to("cpu")
processor = CLIPProcessor.from_pretrained(model_name)

boys = ["Oliver", "Noah", "Henry", "Leo", "Theodore", "Hudson", "Luca", "William", "Charlie", "Jack", "Thomas"]
girls = ["Isla", "Amelia", "Charlotte", "Olivia", "Mia", "Ava", "Matilda", "Harper", "Lily", "Hazel", "Grace"]
unisexs = ["Jessie", "Marion", "Jackie", "Alva", "Ollie", "Jody", "Cleo", "Kerry", "Guadalupe", "Carey", "Tommie"]

def process_names(names, image):
    inputs = processor(text=names, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    results = [{"name": n, "score": round(p.item(), 4)} for n, p in zip(names, probs[0])]
    return sorted(results, key=lambda x: x["score"], reverse=True)

@application.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = image.resize((224, 224))

        return {
            "boys": process_names(boys, image),
            "girls": process_names(girls, image),
            "unisex": process_names(unisexs, image),
        }
    except Exception as e:
        return {"error": str(e)}
