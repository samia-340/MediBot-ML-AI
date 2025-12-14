import os
import sys
import cv2
import torch
import shutil
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import uvicorn
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

print("=" * 70)
print("🚀 MEDIBOT REST API SERVER STARTUP")
print("=" * 70)
sys.stdout.flush()

# ==========================
#     CHROMA DB SETUP
# ==========================

CSV_PATH = "data\combined_labels.csv"    # <-- your medicine list
COLUMN_NAME = "MEDICINE_NAME"
CHROMA_DIR = "./chroma_store"

# 1. Load CSV
df_meds = pd.read_csv(CSV_PATH, usecols=[COLUMN_NAME])
documents = df_meds[COLUMN_NAME].astype(str).tolist()
ids = [f"id_{i}" for i in range(len(documents))]

# 2. Init Chroma Client
chroma_client = chromadb.Client(
    Settings(persist_directory=CHROMA_DIR)
)

# 3. Create or load collection
medicine_collection = chroma_client.get_or_create_collection(
    name="medicines",
    metadata={"hnsw:space": "cosine"}
)

# 4. Add medicine list (only if empty)
existing = medicine_collection.count()
if existing == 0:
    medicine_collection.add(documents=documents, ids=ids)

print(f"Loaded ChromaDB with {medicine_collection.count()} medicine entries.")

# =================================================================
#                 LOAD ALL MODELS
# =================================================================

print("\n[Step 1/5] Loading YOLO model...")
sys.stdout.flush()
YOLO_MODEL_PATH = "models/yolov8_ROI_trained_model.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)
print("    ✅ YOLO model loaded!")
sys.stdout.flush()

print("\n[Step 2/5] Loading TrOCR Finetuned model...")
sys.stdout.flush()
TROCR_MODEL_PATH = "models/model_v6_e15_trainer_save_model"
trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_PATH)
print("    ✅ Model loaded!")
sys.stdout.flush()

print("\n[Step 3/5] Loading TrOCR Finetuned processor...")
sys.stdout.flush()
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
print("    ✅ TrOCR Finetuned Processor loaded!")
sys.stdout.flush()

print("\n[Step 4/5] Setting up device...")
sys.stdout.flush()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    Device: {DEVICE}")
trocr_model.to(DEVICE)
print("    ✅ Ready!")
sys.stdout.flush()

print("\n[Step 5/5] Loading Pretrained TROCR...")
sys.stdout.flush()
PreTrained_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
PreTrained_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
PreTrained_model.to(DEVICE)
print(f"Using device: {DEVICE}")
sys.stdout.flush()

print("\n" + "=" * 70)
print("✅ ALL MODELS LOADED!")
print("=" * 70)
sys.stdout.flush()
#########################################################################
# Folders
MEDICINE_DIR = "Cropped_Results\medicines_crops"
DETAILS_DIR = "Cropped_Results\details_crops"

os.makedirs(MEDICINE_DIR, exist_ok=True)
os.makedirs(DETAILS_DIR, exist_ok=True)

# Your personal details classes
PERSONAL_DETAIL_CLASSES = [
    "NAME", "AGE", "GENDER", "DATE", "DIAGNOSIS",
    "HISTORY", "TEMP", "BP", "WEIGHT"
]

MEDICINE_CLASS = "MEDICINE NAME"


# =================================================================
#                 CREATE FASTAPI APP
# =================================================================
app = FastAPI(title="MediBot API", description="YOLO + TrOCR Pipeline")


# =================================================================
#                 CORE FUNCTIONS
# =================================================================
def preprocess_for_trocr(img, target_size=(384, 64)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_LANCZOS4)
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(thresh)
    final = cv2.bitwise_not(inverted)
    return final

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

def detect_all_rois(image_path, conf_threshold=0.25):
    """Returns bounding boxes for medicine + personal details."""
    results = yolo_model.predict(source=image_path)
    result = results[0]
    boxes = result.boxes
    class_names = result.names

    med_boxes = []
    detail_boxes = []

    for i in range(len(boxes.xywh)):
        conf = float(boxes.conf[i].item())
        if conf < conf_threshold:
            continue

        class_id = int(boxes.cls[i].item())
        class_name = class_names[class_id]

        cx, cy, w, h = boxes.xywh[i]
        x1 = max(0, int(cx - w / 2))
        y1 = max(0, int(cy - h / 2))
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class": class_name, "confidence": conf}

        if class_name == MEDICINE_CLASS:
            med_boxes.append(bbox)

        elif class_name in PERSONAL_DETAIL_CLASSES:
            detail_boxes.append(bbox)
        # elif class_name.upper() in PERSONAL_DETAIL_CLASSES:
         #   detail_boxes.append(bbox)    

    return med_boxes, detail_boxes


def save_all_crops(image_path, med_boxes, detail_boxes):
    img = cv2.imread(image_path)
    saved_meds = []
    saved_details = []

    # --- Save Medicine Name Crops ---
    for idx, box in enumerate(med_boxes):
        crop = img[box["y1"]:box["y2"], box["x1"]:box["x2"]]
        processed = preprocess_for_trocr(crop)

        file_path = os.path.join(MEDICINE_DIR, f"medicine_{idx}.png")
        cv2.imwrite(file_path, processed)
        saved_meds.append(file_path)

    # --- Save Personal Details Crops ---
    for box in detail_boxes:
        crop = img[box["y1"]:box["y2"], box["x1"]:box["x2"]]
        processed = preprocess_for_trocr(crop)

        safe_name = box["class"].lower().replace(" ", "_")
        index = len(saved_details)
        file_path = os.path.join(DETAILS_DIR, f"{safe_name}_{index}.png")

        cv2.imwrite(file_path, processed)
        saved_details.append({box["class"]: file_path})

    return saved_meds, saved_details

def ocr_medicine_name_impl(image_path: str) -> str:
    """Core OCR logic"""
    image = Image.open(image_path).convert("RGB")
    pixel_values = trocr_processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(DEVICE)
    
    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values)
    
    text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text
def process_image(img_path, model, processor, device):
    """Perform inference on a single image using TrOCR."""
    image = Image.open(img_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def run_ocr_on_details(image_folder):
    """
    Run OCR on all images in the given folder using TrOCR.
    Returns a DataFrame containing results.
    """

    if not os.path.exists(image_folder):
        raise ValueError(f"Folder '{image_folder}' does not exist.")

    

    # List images
    image_files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not image_files:
        raise RuntimeError("No image files found in the folder.")

    print(f"Processing {len(image_files)} images in: {image_folder}")

    idx_list, pred_list, path_list = [], [], []

    for idx, image_name in enumerate(tqdm(image_files, desc="OCR Processing")):
        image_path = os.path.join(image_folder, image_name)

        try:
            text = process_image(image_path, PreTrained_model, PreTrained_processor, DEVICE)
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            text = ""

        idx_list.append(idx)
        pred_list.append(text)
        path_list.append(image_path)

    # Prepare DataFrame
    df_out = pd.DataFrame({
        "idx": idx_list,
        "pred": pred_list,
        "path": path_list
    })

    # Save CSV inside the same folder
    output_csv = os.path.join(image_folder, "ocr_results.csv")
    df_out.to_csv(output_csv, index=False)

    print(f"OCR results saved to: {output_csv}")

    return df_out
def verify_medicine_name(medicine: str) -> str:
    """
    Returns the closest matching medicine name from the vector store.
    """
    if not medicine or len(medicine.strip()) == 0:
        return ""

    results = medicine_collection.query(
        query_texts=[medicine],
        n_results=1
    )
    return results["documents"][0][0]


# =================================================================
#                 REST API ENDPOINTS
# =================================================================

class DetectAllRequest(BaseModel):
    image_path: str
    conf_threshold: float = 0.25


class SaveCropsRequest(BaseModel):
    image_path: str
    conf_threshold: float = 0.25


class OCRRequest(BaseModel):
    image_path: str


class OCRFolderRequest(BaseModel):
    folder_path: str


class MedicineList(BaseModel):
    medicines: list[str]   

@app.post("/api/verify_medicines")
def verify_medicines(req: MedicineList):
    """
    Takes list of predicted medicines (from OCR)
    Returns vector-corrected medicine names
    """

    corrected = []
    for med in req.medicines:
        fixed = verify_medicine_name(med)
        corrected.append({
            "input_text": med,
            "verified_text": fixed
        })

    return {
        "success": True,
        "count": len(corrected),
        "verified": corrected
    }    

@app.post("/api/ocr_details")
async def api_ocr_details(request: OCRFolderRequest):
    """
    Perform OCR on all images in a folder.
    Saves results to ocr_results.csv and returns JSON response.
    """
    try:
        df = run_ocr_on_details(request.folder_path)
        # Convert DataFrame to JSON-friendly format
        results = df.to_dict(orient="records")
        return {
            "success": True,
            "folder": request.folder_path,
            "num_images": len(results),
            "results": results,
            "csv_saved_at": str(os.path.join(request.folder_path, "ocr_results.csv"))
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "✅ MediBot REST API Server Running",
        "endpoints": {
            "detect": "POST /api/detect",
            "crop": "POST /api/crop",
            "ocr": "POST /api/ocr",
            "full_pipeline": "POST /api/extract_all",
            "class_names": "GET /api/classes"
        },
        "docs": "http://127.0.0.1:8000/docs"
    }


@app.post("/api/detect_all")
async def api_detect_all(request: DetectAllRequest):
    """Detect medicine name bounding boxes"""
    med_boxes, detail_boxes = detect_all_rois(
        request.image_path, 
        conf_threshold=request.conf_threshold
    )

    return {
        "medicine_boxes": med_boxes,
        "personal_detail_boxes": detail_boxes,
        "medicine_count": len(med_boxes),
        "personal_count": len(detail_boxes),
        "success": True
    }



@app.post("/api/save_crops")
async def api_save_crops(request: SaveCropsRequest):
    """
    Detect → crop → save all medicine + detail ROIs.
    """
    # Clear old crops before saving new ones
    clear_folder(MEDICINE_DIR)
    clear_folder(DETAILS_DIR)

    med_boxes, detail_boxes = detect_all_rois(
        request.image_path, 
        conf_threshold=request.conf_threshold
    )

    saved_meds, saved_details = save_all_crops(
        request.image_path, 
        med_boxes, 
        detail_boxes
    )

    return {
        "saved_medicine_crops": saved_meds,
        "saved_detail_crops": saved_details,
        "medicine_count": len(saved_meds),
        "details_count": len(saved_details),
        "success": True
    }



@app.post("/api/ocr")
async def api_ocr(request: OCRRequest):
    """Perform OCR on an image"""
    try:
        text = ocr_medicine_name_impl(request.image_path)
        return {"text": text, "success": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/classes")
async def api_classes():
    """Get YOLO class names"""
    return {"classes": list(yolo_model.names.values())}


@app.post("/api/extract_all")
async def api_extract_all(request: DetectAllRequest):
    """
    Full pipeline using existing functions:
    1. detect_all_rois()
    2. save_all_crops()
    3. ocr_medicine_name_impl()
    """
    clear_folder(MEDICINE_DIR)
    clear_folder(DETAILS_DIR)
    try:
        # ---------------------------
        # Step 1: Detect ROIs
        # ---------------------------
        med_boxes, detail_boxes = detect_all_rois(
            request.image_path,
            conf_threshold=request.conf_threshold
        )

        if len(med_boxes) == 0:
            return {
                "success": True,
                "message": "No medicine ROIs detected.",
                "medicines": []
            }

        # ---------------------------
        # Step 2: Crop & Save ROIs
        # ---------------------------
        saved_meds, saved_details = save_all_crops(
            request.image_path,
            med_boxes,
            detail_boxes
        )

        # ---------------------------
        # Step 3: OCR on each medicine crop
        # ---------------------------
        results = []
        for crop_path in saved_meds:
            text = ocr_medicine_name_impl(crop_path)
            results.append({
                "text": text,
                "crop_path": crop_path
            })

        return {
            "success": True,
            "count": len(results),
            "medicines": results,
            "details": saved_details
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =================================================================
#                 RUN SERVER
# =================================================================
if __name__ == "__main__":
    print("\n🚀 Starting REST API Server...")
    print("🌐 Server URL: http://127.0.0.1:8000")
    print("📖 API Docs: http://127.0.0.1:8000/docs")
    print("=" * 70)
    print("\n✅ Ready to accept requests!\n")
    sys.stdout.flush()
    
    uvicorn.run(app, host="127.0.0.1", port=8000)