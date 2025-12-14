# MediBot: A Multi-Agent AI Pipeline for Handwritten Prescription Digitization and Patient Guidance

MediBot is a multi-agent AI system that automatically digitizes handwritten medical prescriptions, verifies medicines, and generates easy-to-understand patient guidance using vision, OCR, vector databases, and large language models.

## Problem Statement

Handwritten medical prescriptions are difficult to digitize due to inconsistent handwriting, low image quality, and OCR limitations. Traditional OCR pipelines are rigid, error-prone, and fail to handle noisy medical text, dosage interpretation, and patient understanding. There is a need for a scalable, intelligent system that can extract structured information, correct OCR errors, and provide meaningful patient-friendly outputs.

### Tools and Technologies

YOLOv8 – Region of Interest (ROI) detection for medicines and patient details

TrOCR (Fine-tuned & Pretrained) – Handwritten text recognition

FastAPI – REST API backend for vision and OCR services

ChromaDB – Vector database for medicine name verification

Sentence Transformers (all-MiniLM-L6-v2) – Embedding generation for similarity matching

Groq LLM (LLaMA-70B) – Reasoning and patient instruction generation

Python, PyTorch, OpenCV, Pandas – Core implementation stack

#### Key Insights

- Multi-agent architecture improves modularity, fault tolerance, and scalability

- ROI-based OCR significantly reduces noise compared to full-image OCR

- ChromaDB improves medicine recognition accuracy (up to ~98–100%)

- LLM reasoning enables patient-friendly summaries without altering prescriptions

- Fallback mechanisms ensure pipeline robustness even when individual agents fail

**How to Run This Project**
1. Clone the Repository
git clone <MediBot-ML_AI>
cd medibot

2. Install Dependencies
pip install -r requirements.txt

3. Start the REST API Server
python scripts\rest_server.py


The server will run at:

http://127.0.0.1:8000

4. Set Groq API Key 
export GROQ_API_KEY=your_api_key_here

5. Run the streamlit user interface
streamlit run scripts/streamlit_app.py


Provide a prescription image path inside the script or via CLI (if extended).

##### Results

- Successful detection of medicine and patient detail ROIs

- Accurate OCR extraction of patient details (name, BP, temperature, history, etc.)

- Post-OCR medicine correction using ChromaDB

- Robust fallback handling for OCR and ROI failures

Final output includes:

- Verified medicine list

- Confidence score

- Warnings (if any)

- Patient-friendly health guidance summary

###### Conclusion

MediBot demonstrates how an agentic AI pipeline can transform handwritten prescription digitization into an intelligent, scalable, and patient-centric system. By combining vision models, OCR, vector databases, and LLM reasoning, the system improves reliability, accuracy, and usability. The modular design allows easy integration of future models, retraining strategies, and parallel inference for real-world healthcare applications.
###### Author and Contact
Samia Ahmed
samia.ahmed.mughal@gmail.com
www.linkedin.com/in/samiaahmed-datascientist