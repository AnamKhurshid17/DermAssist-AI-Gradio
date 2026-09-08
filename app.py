import os
import warnings
import cv2
import gradio as gr
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from groq import Groq
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    SegformerForSemanticSegmentation,
)

warnings.filterwarnings("ignore")

SEGMENTATION_BASE = "nvidia/mit-b0"
SEGMENTATION_REPO = "mahnoor-2722/isic-segformer"
SEGMENTATION_WEIGHTS = "best_segformer_isic2018.pth"
CLASSIFIER_ID = "medicaldataset/Skin_Cancer-Image_Classification"
GROQ_MODEL = "openai/gpt-oss-120b"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

def prepare_image(image):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.array(image)).convert("RGB")

def clean_label(label):
    return str(label).replace("_", " ").replace("-", " ").title()

print("Loading DermAssist AI...")
print("Device:", DEVICE)

seg_processor = AutoImageProcessor.from_pretrained(SEGMENTATION_BASE)
seg_model = SegformerForSemanticSegmentation.from_pretrained(
    SEGMENTATION_BASE,
    num_labels=1,
    id2label={0: "lesion"},
    label2id={"lesion": 0},
    ignore_mismatched_sizes=True,
)

weights_path = hf_hub_download(
    repo_id=SEGMENTATION_REPO,
    filename=SEGMENTATION_WEIGHTS,
)

checkpoint = torch.load(weights_path, map_location="cpu")
if isinstance(checkpoint, dict):
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

clean_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("module."):
        key = key.replace("module.", "", 1)
    clean_state_dict[key] = value

load_result = seg_model.load_state_dict(clean_state_dict, strict=False)
print("Segmentation missing keys:", len(load_result.missing_keys))
print("Segmentation unexpected keys:", len(load_result.unexpected_keys))

if load_result.missing_keys or load_result.unexpected_keys:
    raise RuntimeError("Segmentation checkpoint did not load cleanly.")

seg_model = seg_model.to(DEVICE)
seg_model.eval()

clf_processor = AutoImageProcessor.from_pretrained(CLASSIFIER_ID)
clf_model = AutoModelForImageClassification.from_pretrained(CLASSIFIER_ID)
clf_model = clf_model.to(DEVICE)
clf_model.eval()

print("Classification classes:")
for i, label in clf_model.config.id2label.items():
    print(i, ":", label)

def segment_lesion(image):
    image = prepare_image(image)
    if image is None:
        raise ValueError("No image was provided.")
    inputs = seg_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = seg_model(**inputs)
    logits = F.interpolate(
        outputs.logits,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )
    probability_map = torch.sigmoid(logits[0, 0]).cpu().numpy()
    mask = (probability_map > 0.5).astype(np.uint8)
    return mask, probability_map

def clean_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = mask.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def make_mask_image(mask):
    return Image.fromarray(mask.astype(np.uint8) * 255)

def create_segmentation_overlay(image, mask):
    image = prepare_image(image)
    img = np.array(image).copy()
    overlay = img.copy()
    overlay[mask > 0] = [255, 80, 170]
    result = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)
    return Image.fromarray(result)

def calculate_lesion_area(mask):
    return float(np.sum(mask > 0) / mask.size * 100)

def classify_lesion(image):
    image = prepare_image(image)
    if image is None:
        raise ValueError("No image was provided.")
    inputs = clf_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clf_model(**inputs)
    return torch.softmax(outputs.logits, dim=-1)[0]

def calculate_confidence(probabilities):
    return float(probabilities.max().item())

def calculate_uncertainty(probabilities):
    p = probabilities.detach().cpu().numpy()
    p = np.clip(p, 1e-10, 1.0)
    entropy = -np.sum(p * np.log(p))
    return float(entropy / np.log(len(p)))

def get_uncertainty_level(value):
    if value < 0.25:
        return "Low"
    if value < 0.50:
        return "Moderate"
    if value < 0.75:
        return "High"
    return "Very High"

def generate_heatmap(image):
    image = prepare_image(image)
    inputs = clf_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE).clone().detach()
    pixel_values.requires_grad_(True)
    clf_model.zero_grad()
    outputs = clf_model(pixel_values=pixel_values)
    predicted_index = int(outputs.logits.argmax(dim=-1))
    outputs.logits[0, predicted_index].backward()
    gradients = pixel_values.grad.detach()[0]
    saliency = gradients.abs().mean(dim=0).cpu().numpy()
    saliency -= saliency.min()
    saliency /= saliency.max() + 1e-8
    saliency = cv2.resize(saliency, (image.width, image.height))
    heatmap = (cm.jet(saliency)[:, :, :3] * 255).astype(np.uint8)
    original = np.array(image).astype(np.float32)
    result = np.clip(0.60 * original + 0.40 * heatmap, 0, 255).astype(np.uint8)
    return Image.fromarray(result)

def generate_ai_explanation(predicted_class, probability, confidence, uncertainty, uncertainty_level):
    if not GROQ_API_KEY:
        return "### AI Explanation Unavailable\n\nThe Groq API key has not been configured. The computer vision results are still available."
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""
You are explaining an experimental skin lesion AI research prototype.

Predicted class: {predicted_class}
Model probability: {probability:.1f}%
Model confidence: {confidence:.1f}%
Uncertainty: {uncertainty:.1f}%
Uncertainty level: {uncertainty_level}

Write 3-5 concise sentences. Do not diagnose cancer, melanoma, or any disease.
Do not invent symptoms or patient history. Explain this as an experimental AI
prediction. State that confidence is not clinical certainty and recommend
professional medical evaluation for concerning lesions.
"""
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Explain medical AI research results cautiously. Do not diagnose patients."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=250,
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Groq error:", e)
        return "### AI Explanation Unavailable\n\nThe AI explanation service could not be reached. The computer vision results are still available."

def analyze_image(image):
    if image is None:
        return None, None, None, "Please upload a skin lesion image.", ""
    try:
        image = prepare_image(image)
        mask, _ = segment_lesion(image)
        mask = clean_mask(mask)
        mask_image = make_mask_image(mask)
        overlay = create_segmentation_overlay(image, mask)
        lesion_area = calculate_lesion_area(mask)

        probabilities = classify_lesion(image)
        predicted_index = int(probabilities.argmax())
        predicted_probability = float(probabilities[predicted_index])
        predicted_label = clean_label(clf_model.config.id2label[predicted_index])

        confidence = calculate_confidence(probabilities)
        uncertainty = calculate_uncertainty(probabilities)
        uncertainty_level = get_uncertainty_level(uncertainty)
        heatmap = generate_heatmap(image)

        explanation = generate_ai_explanation(
            predicted_label,
            predicted_probability * 100,
            confidence * 100,
            uncertainty * 100,
            uncertainty_level,
        )

        technical = f"""### Model Results

**Predicted Class:** {predicted_label}

**Model Probability:** {predicted_probability * 100:.2f}%

**Model Confidence:** {confidence * 100:.2f}%

**Uncertainty:** {uncertainty * 100:.2f}%

**Uncertainty Level:** {uncertainty_level}

**Predicted Lesion Area:** {lesion_area:.2f}%

**Device:** {DEVICE}

---

### Class Probabilities
"""
        for i, probability in enumerate(probabilities):
            label = clean_label(clf_model.config.id2label[i])
            technical += f"- **{label}:** {float(probability) * 100:.2f}%\n"

        technical += """
---

### Methodological Note

The uncertainty is normalized predictive entropy, not a calibrated clinical
uncertainty estimate. The heatmap is a gradient-based input saliency map and
is not a clinically validated localization method.
"""
        return mask_image, overlay, heatmap, explanation, technical
    except Exception as e:
        print("Analysis error:", e)
        return None, None, None, f"### Analysis Error\n\n`{str(e)}`", ""

CUSTOM_CSS = """
.gradio-container { max-width: 1250px !important; margin: auto !important; }
.hero {
    background: linear-gradient(135deg, #0f172a, #164e63);
    padding: 35px; border-radius: 24px; color: white;
    text-align: center; margin-bottom: 25px;
}
.hero h1 { font-size: 44px; margin-bottom: 5px; }
.hero p { font-size: 18px; opacity: 0.9; }
"""

with gr.Blocks(title="DermAssist AI", css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div class="hero">
        <h1>🩺 DermAssist AI</h1>
        <p>Explainable and Uncertainty-Aware Skin Lesion Analysis</p>
        <p>Research Prototype • Medical AI • Computer Vision</p>
    </div>
    """)

    gr.Markdown("""
### About DermAssist AI

Upload a skin lesion image to explore an experimental AI pipeline combining
lesion segmentation, image classification, uncertainty estimation, and
gradient-based explainability.

**This is a research and educational prototype, not a diagnostic medical device.**
""")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Skin Lesion Image")
            analyze_button = gr.Button("🔍 Analyze Lesion", variant="primary")

    gr.Markdown("## 🔬 Analysis Results")
    explanation_output = gr.Markdown(label="AI Explanation")

    with gr.Row():
        mask_output = gr.Image(label="Segmentation Mask")
        overlay_output = gr.Image(label="Segmentation Overlay")
        heatmap_output = gr.Image(label="Explainability Heatmap")

    with gr.Accordion("📊 Technical Model Results", open=False):
        technical_output = gr.Markdown()

    with gr.Accordion("⚠️ Medical Disclaimer", open=True):
        gr.Markdown("""
**DermAssist AI is an experimental research and educational prototype.**

It is **NOT a medical device** and must not be used to diagnose skin cancer,
make treatment decisions, or replace professional medical advice.

AI predictions can be incorrect. Model probability and confidence are not
clinical certainty or medical risk.

If you are concerned about a skin lesion, consult a qualified dermatologist
or healthcare professional.
""")

    analyze_button.click(
        fn=analyze_image,
        inputs=input_image,
        outputs=[mask_output, overlay_output, heatmap_output, explanation_output, technical_output],
    )

if __name__ == "__main__":
    demo.launch()
