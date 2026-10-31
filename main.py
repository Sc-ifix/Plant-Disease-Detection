import io, json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
import torch, torch.nn.functional as F
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent

# Load labels; accept either idx→class or class→idx and normalize to idx→class
with open(ROOT / "labels.json") as f:
    IDX2CLASS = json.load(f)
    if "0" not in IDX2CLASS:          # file was class→idx
        IDX2CLASS = {str(v): k for k, v in IDX2CLASS.items()}

MODEL = torch.jit.load(str(ROOT / "model_scripted.pt"), map_location="cpu").eval()

# Start with the most common recipe; we'll tweak in step 5 if needed
# match training: 256x256, no normalization
PRE = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def home():
    return '<p>Plant Disease API running. Try <a href="/docs">/docs</a>.</p>'

def logits_from(x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = MODEL(x)
        if isinstance(out, (tuple, list)): out = out[0]
        if isinstance(out, dict): out = out.get("logits", next(iter(out.values())))
        if out.ndim == 1: out = out.unsqueeze(0)   # [C] → [1,C]
    return out

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = PRE(img).unsqueeze(0)
    logits = logits_from(x)                    # [1,C]
    C = logits.shape[-1]
    assert C == len(IDX2CLASS), f"class mismatch: logits={C} labels={len(IDX2CLASS)}"
    probs = F.softmax(logits, dim=1)[0]
    top = torch.topk(probs, k=min(3, C))
    return {"predictions":[
        {"label": IDX2CLASS[str(int(i))], "confidence": float(p)}
        for p,i in zip(top.values.tolist(), top.indices.tolist())
    ], "meta":{"num_classes": C}}

# --- diagnostics ---
@app.get("/_labels")
def _labels():
    return {"labels_len": len(IDX2CLASS),
            "sample": {k: IDX2CLASS[k] for k in list(IDX2CLASS)[:5]}}

from fastapi.responses import HTMLResponse, JSONResponse  # add JSONResponse

# ... keep everything else the same ...

@app.post("/_debug_logits")
async def _debug_logits(file: UploadFile = File(...)):
    import traceback
    try:
        b = await file.read()
        img = Image.open(io.BytesIO(b)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={
            "where": "image_open", "error": repr(e)
        })

    try:
        x = PRE(img).unsqueeze(0)
        l = logits_from(x)[0].detach().cpu()
        return {
            "shape": list(l.shape),
            "min": float(l.amin().item()),
            "max": float(l.amax().item()),
            "argmax": int(l.argmax().item()),
            "labels_len": len(IDX2CLASS)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "where": "forward", "error": repr(e), "trace": traceback.format_exc()
        })
