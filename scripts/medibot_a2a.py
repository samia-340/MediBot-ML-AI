"""
MediBot A2A Agent System
Integrates multi-agent reasoning with existing REST server
"""

import os
import json
import asyncio
import requests
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum
import pandas as pd  # (pandas = library to read csv and tabular data)

# Optional Groq LLM support (Groq = LLM provider used to run Llama models)
try:
    from groq import Groq
except ImportError:
    Groq = None

# ============================================================================
# CONFIGURATION
# ============================================================================

REST_API_BASE = "http://127.0.0.1:8000/api"
DETAILS_CROPS_FOLDER = "Cropped_Results\details_crops"
OCR_RESULTS_FILENAME = "ocr_results.csv"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = None
if Groq is not None and GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        groq_client = None


class AgentRole(str, Enum):
    COORDINATOR = "coordinator"
    VISION = "vision"
    TEXT_READER = "text_reader"
    VERIFIER = "verifier"
    KNOWLEDGE = "knowledge"
    VALIDATOR = "validator"
    LEARNING = "learning"
    SMART_DETAIL = "smart_detail"  # new role for Groq LLM agent

# ============================================================================
# DATA MODELS
# ============================================================================

class Message(BaseModel):
    """Inter-agent message format"""
    sender: AgentRole
    receiver: AgentRole
    message_type: str
    content: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: str(asyncio.get_event_loop().time()))


class PrescriptionData(BaseModel):
    """Structured prescription output"""
    patient_name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    date: Optional[str] = None
    diagnosis: Optional[str] = None
    medicines: List[Dict[str, str]] = []
    confidence_score: float = 0.0
    validation_status: str = "pending"
    warnings: List[str] = []
    smart_summary: Optional[str] = None  # new: Groq LLM reasoning summary


class AgentState(BaseModel):
    """Shared state between agents"""
    image_path: str
    prescription_data: PrescriptionData = Field(default_factory=PrescriptionData)
    roi_detections: Dict[str, Any] = {}
    ocr_results: Dict[str, Any] = {}
    verification_results: List[Dict[str, str]] = []
    agent_messages: List[Message] = []
    current_step: str = "initialized"
    errors: List[str] = []

# ============================================================================
# BASE AGENT CLASS
# ============================================================================

class BaseAgent:
    """Base class for all agents"""

    def __init__(self, role: AgentRole, state: AgentState):
        self.role = role
        self.state = state
        self.logger: List[str] = []

    def log(self, message: str):
        """Log agent activities"""
        log_entry = f"[{self.role.value.upper()}] {message}"
        self.logger.append(log_entry)
        print(log_entry)

    def send_message(self, receiver: AgentRole, msg_type: str, content: Dict):
        """Send message to another agent"""
        msg = Message(
            sender=self.role,
            receiver=receiver,
            message_type=msg_type,
            content=content
        )
        self.state.agent_messages.append(msg)
        self.log(f"Sent {msg_type} to {receiver.value}")
        return msg

    async def execute(self) -> Dict[str, Any]:
        """Override in subclass"""
        raise NotImplementedError

# ============================================================================
# VISION AGENT
# ============================================================================

class VisionAgent(BaseAgent):
    """Handles ROI (ROI = region of interest) detection using YOLO (YOLO = object detection model)"""

    def __init__(self, state: AgentState):
        super().__init__(AgentRole.VISION, state)

    async def execute(self) -> Dict[str, Any]:
        """Detect all ROIs in the prescription"""
        self.log("Starting ROI detection...")

        try:
            # Call REST API for ROI detection
            response = requests.post(
                f"{REST_API_BASE}/detect_all",
                json={
                    "image_path": self.state.image_path,
                    "conf_threshold": 0.25
                }
            )

            if response.status_code == 200:
                data = response.json()
                self.state.roi_detections = data
                self.log(f"Detected {data['medicine_count']} medicines, {data['personal_count']} details")

                # Send results to coordinator
                self.send_message(
                    AgentRole.COORDINATOR,
                    "roi_detection_complete",
                    {"detections": data}
                )

                return {"success": True, "detections": data}
            else:
                error = f"ROI detection failed: {response.text}"
                self.state.errors.append(error)
                return {"success": False, "error": error}

        except Exception as e:
            error = f"Vision agent error: {str(e)}"
            self.log(error)
            self.state.errors.append(error)
            return {"success": False, "error": error}

# ============================================================================
# TEXT READER AGENT
# ============================================================================

class TextReaderAgent(BaseAgent):
    """Handles OCR (OCR = optical character recognition) using TrOCR (TrOCR = text recognition model)"""

    def __init__(self, state: AgentState):
        super().__init__(AgentRole.TEXT_READER, state)

    async def execute(self) -> Dict[str, Any]:
        """Perform OCR on detected ROIs"""
        self.log("Starting OCR processing...")

        try:
            # Step 1: Save crops
            response = requests.post(
                f"{REST_API_BASE}/save_crops",
                json={
                    "image_path": self.state.image_path,
                    "conf_threshold": 0.25
                }
            )

            if response.status_code != 200:
                return {"success": False, "error": "Failed to save crops"}

            crop_data = response.json()
            self.log(f"Saved {crop_data['medicine_count']} medicine crops")

            # Step 2: OCR on medicine crops
            medicine_texts = []
            for crop_path in crop_data['saved_medicine_crops']:
                ocr_response = requests.post(
                    f"{REST_API_BASE}/ocr",
                    json={"image_path": crop_path}
                )

                if ocr_response.status_code == 200:
                    ocr_data = ocr_response.json()
                    medicine_texts.append({
                        "text": ocr_data["text"],
                        "crop_path": crop_path
                    })

            # Step 3: OCR on personal details (also generates CSV in details_crops)
            details_response = requests.post(
                f"{REST_API_BASE}/ocr_details",
                json={"folder_path": DETAILS_CROPS_FOLDER}
            )

            details_data = {}
            if details_response.status_code == 200:
                details_data = details_response.json()

            self.state.ocr_results = {
                "medicines": medicine_texts,
                "details": details_data
            }

            self.log(f"OCR complete: {len(medicine_texts)} medicines read")

            # Send to coordinator
            self.send_message(
                AgentRole.COORDINATOR,
                "ocr_complete",
                {"results": self.state.ocr_results}
            )

            return {"success": True, "results": self.state.ocr_results}

        except Exception as e:
            error = f"Text reader error: {str(e)}"
            self.log(error)
            self.state.errors.append(error)
            return {"success": False, "error": error}

# ============================================================================
# KNOWLEDGE AGENT
# ============================================================================

class KnowledgeAgent(BaseAgent):
    """Verifies medicine names using ChromaDB (ChromaDB = vector database for similarity search)"""

    def __init__(self, state: AgentState):
        super().__init__(AgentRole.KNOWLEDGE, state)

    async def execute(self) -> Dict[str, Any]:
        """Verify medicine names against knowledge base"""
        self.log("Starting medicine verification...")

        try:
            medicines = [m["text"] for m in self.state.ocr_results.get("medicines", [])]

            if not medicines:
                return {"success": True, "verified": []}

            # Call verification endpoint
            response = requests.post(
                f"{REST_API_BASE}/verify_medicines",
                json={"medicines": medicines}
            )

            if response.status_code == 200:
                data = response.json()
                self.state.verification_results = data["verified"]

                self.log(f"Verified {data['count']} medicines")

                # Send to coordinator
                self.send_message(
                    AgentRole.COORDINATOR,
                    "verification_complete",
                    {"verified": data["verified"]}
                )

                return {"success": True, "verified": data["verified"]}
            else:
                error = f"Verification failed: {response.text}"
                self.state.errors.append(error)
                return {"success": False, "error": error}

        except Exception as e:
            error = f"Knowledge agent error: {str(e)}"
            self.log(error)
            self.state.errors.append(error)
            return {"success": False, "error": error}

# ============================================================================
# VERIFIER AGENT (rule-based)
# ============================================================================

class VerifierAgent(BaseAgent):
    """Applies contextual reasoning and validation (currently simple rule-based checks)"""

    def __init__(self, state: AgentState):
        super().__init__(AgentRole.VERIFIER, state)

    async def execute(self) -> Dict[str, Any]:
        """Validate prescription data using simple rules"""
        self.log("Starting verification checks...")

        try:
            warnings: List[str] = []

            # Check for suspicious patterns
            for med in self.state.verification_results:
                input_text = med.get("input_text", "")
                verified_text = med.get("verified_text", "")

                if len(input_text) > 0 and len(verified_text) > 0:
                    input_set = set(input_text.lower())
                    verified_set = set(verified_text.lower())
                    if input_set:
                        similarity = len(input_set & verified_set) / len(input_set)
                    else:
                        similarity = 1.0
                    if similarity < 0.3:
                        warnings.append(f"Low confidence match: '{input_text}' → '{verified_text}'")

            self.state.prescription_data.warnings = warnings
            self.state.prescription_data.validation_status = "verified"

            self.log(f"Validation complete with {len(warnings)} warnings")

            # Send to coordinator
            self.send_message(
                AgentRole.COORDINATOR,
                "validation_complete",
                {"warnings": warnings, "status": "verified"}
            )

            return {"success": True, "warnings": warnings}

        except Exception as e:
            error = f"Verifier agent error: {str(e)}"
            self.log(error)
            self.state.errors.append(error)
            return {"success": False, "error": error}

# ============================================================================
# SMART DETAIL AGENT (Groq LLM)
# ============================================================================

class SmartDetailAgent(BaseAgent):
    """Reads CSV + OCR + verified meds and calls Groq LLM (Groq LLM = Llama model hosted on Groq) for reasoning"""

    def __init__(self, state: AgentState, csv_folder: str = DETAILS_CROPS_FOLDER):
        super().__init__(AgentRole.SMART_DETAIL, state)
        self.csv_folder = csv_folder

    def _get_csv_path(self) -> str:
        return os.path.join(self.csv_folder, OCR_RESULTS_FILENAME)

    def _build_prompt(
    self,
    patient_rows: List[Dict[str, Any]],
    medicine_rows: List[Dict[str, Any]],
    verified_meds: List[Dict[str, Any]]
) -> str:

        details_lines = [
            f"- {row.get('path', '')}: {row.get('pred', '')}"
            for row in patient_rows
    ] or ["(no patient detail OCR rows found)"]

        ocr_med_lines = [
            f"- OCR: {m.get('text', '')} (crop: {m.get('crop_path', '')})"
            for m in medicine_rows
    ] or ["(no medicine OCR results found)"]

        verified_lines = [
            f"- {v.get('input_text', '')} → {v.get('verified_text', '')}"
            for v in verified_meds
    ] or ["(no verified medicines from ChromaDB)"]

        prompt = f"""
You are a professional health education assistant. You are NOT a doctor.
You will receive:
1. Patient personal details (noisy OCR, approximate)
2. List of medicines with interpreted dosage instructions
3. The goal is to generate FRIENDLY, PATIENT-FACING instructions.

Your tasks:
- Explain the treatment in a simple, reassuring tone.
- Suggest general lifestyle improvements (hydration, sleep, diet, stress control).
- Provide organic & natural remedies where safe (honey, turmeric, steam, hydration).
- Include clear disclaimers: "This is not medical advice. Consult your doctor for actual diagnosis or treatment."
- Provide reputable references at the end (WHO, Mayo Clinic, NHS).
- DO NOT adjust medicine dosage. DO NOT prescribe new medicines. DO NOT tell the patient to stop any medicine.

Here is the extracted data:
#####################
PATIENT DETAIL OCR ROWS:
{chr(10).join(details_lines)}

#####################
MEDICINE OCR TEXTS:
{chr(10).join(ocr_med_lines)}

#####################
VERIFIED MEDICINE NAMES:
{chr(10).join(verified_lines)}

#####################
ow write a friendly, helpful explanation for the patient with these sections:
1. Overview (2–3 lines)
2. How toN support your health naturally (bullet points)
3. Helpful diet & routine tips (bullet points)
4. Safety notes (2–3 points)
5. References
"""
        return prompt


    def _call_groq_sync(self, prompt: str) -> str:
        if groq_client is None:
            return "Groq client not configured or GROQ_API_KEY missing. Fallback summary:\n" + prompt[:800]
        try:
            resp = groq_client.chat.completions.create(
                #model="llama-3.1-8b-instant",
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful assistant for understanding handwritten prescriptions. "
                            "You correct OCR mistakes but do not prescribe new medicines."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.2,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"Groq LLM call failed ({type(e).__name__}): fallback summary.\n" + prompt[:800]

    async def execute(self) -> Dict[str, Any]:
        self.log("Starting SmartDetailAgent reasoning (Groq LLM)...")
        try:
            csv_path = self._get_csv_path()
            if not os.path.exists(csv_path):
                err = f"OCR CSV not found at: {csv_path}"
                self.state.errors.append(err)
                self.log(err)
                # still provide minimal summary
                fallback = f"Verified medicines: {len(self.state.verification_results)}. CSV with patient details not found."
                self.state.prescription_data.smart_summary = fallback
                return {"success": False, "error": err, "fallback": fallback}

            # Load CSV with patient detail OCR
            df = pd.read_csv(csv_path)
            patient_rows = df.to_dict(orient="records")

            # Medicine OCR rows come from state.ocr_results["medicines"]
            medicine_rows = self.state.ocr_results.get("medicines", [])
            verified_meds = self.state.verification_results

            prompt = self._build_prompt(patient_rows, medicine_rows, verified_meds)

            # Run blocking LLM call in a thread
            summary = await asyncio.to_thread(self._call_groq_sync, prompt)
            self.state.prescription_data.smart_summary = summary

            # Optional: add small pointer into diagnosis
            if not self.state.prescription_data.diagnosis:
                self.state.prescription_data.diagnosis = "See smart_summary for LLM reasoning output."

            self.send_message(
                AgentRole.COORDINATOR,
                "smart_detail_complete",
                {"summary": summary}
            )
            self.log("SmartDetailAgent reasoning complete.")
            return {"success": True, "summary": summary}

        except Exception as e:
            err = f"SmartDetailAgent error: {str(e)}"
            self.state.errors.append(err)
            self.log(err)
            return {"success": False, "error": err}

# ============================================================================
# COORDINATOR AGENT
# ============================================================================

class CoordinatorAgent(BaseAgent):
    """Orchestrates the multi-agent workflow"""

    def __init__(self, state: AgentState):
        super().__init__(AgentRole.COORDINATOR, state)
        self.agents: Dict[AgentRole, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent):
        """Register an agent"""
        self.agents[agent.role] = agent
        self.log(f"Registered {agent.role.value} agent")

    async def execute(self) -> Dict[str, Any]:
        """Execute the full prescription processing pipeline"""
        self.log("Starting prescription processing pipeline...")

        try:
            # Step 1: Vision Agent - Detect ROIs
            self.state.current_step = "roi_detection"
            vision_result = await self.agents[AgentRole.VISION].execute()
            if not vision_result["success"]:
                raise Exception("ROI detection failed")

            # Step 2: Text Reader Agent - OCR
            self.state.current_step = "ocr"
            reader_result = await self.agents[AgentRole.TEXT_READER].execute()
            if not reader_result["success"]:
                raise Exception("OCR processing failed")

            # Step 3: Knowledge Agent - Verify medicines
            self.state.current_step = "verification"
            knowledge_result = await self.agents[AgentRole.KNOWLEDGE].execute()
            if not knowledge_result["success"]:
                raise Exception("Medicine verification failed")

            # Step 4: Verifier Agent - Validate with reasoning
            self.state.current_step = "validation"
            verifier_result = await self.agents[AgentRole.VERIFIER].execute()
            if not verifier_result["success"]:
                raise Exception("Validation failed")

            # Step 5: Smart Detail Agent - LLM reasoning (optional, non-breaking)
            smart_agent = self.agents.get(AgentRole.SMART_DETAIL)
            if smart_agent is not None:
                self.state.current_step = "smart_detail"
                smart_result = await smart_agent.execute()
                if not smart_result.get("success", False):
                    # Do not fail pipeline; just log
                    self.log(f"SmartDetailAgent did not succeed: {smart_result.get('error', 'unknown error')}")

            # Step 6: Compile final results
            self.compile_final_output()

            self.log("✅ Pipeline complete!")
            self.state.current_step = "completed"

            return {
                "success": True,
                "prescription_data": self.state.prescription_data.dict(),
                "agent_logs": self.get_all_logs()
            }

        except Exception as e:
            error = f"Coordinator error: {str(e)}"
            self.log(error)
            self.state.errors.append(error)
            self.state.current_step = "failed"
            return {"success": False, "error": error, "agent_logs": self.get_all_logs()}

    def compile_final_output(self):
        """Compile verified data into structured format"""
        # Add verified medicines
        for med in self.state.verification_results:
            self.state.prescription_data.medicines.append({
                "name": med["verified_text"],
                "original_text": med["input_text"]
            })

        # Calculate confidence score
        total_meds = len(self.state.verification_results)
        if total_meds > 0:
            high_confidence = sum(
                1 for m in self.state.verification_results
                if len(m["verified_text"]) > 2
            )
            self.state.prescription_data.confidence_score = high_confidence / total_meds

        self.log(f"Compiled {len(self.state.prescription_data.medicines)} medicines")

    def get_all_logs(self) -> Dict[str, List[str]]:
        """Collect logs from all agents"""
        logs: Dict[str, List[str]] = {}
        for role, agent in self.agents.items():
            logs[role.value] = agent.logger
        return logs

# ============================================================================
# MAIN A2A SYSTEM
# ============================================================================

class MediBotA2ASystem:
    """Main A2A (A2A = agent-to-agent) orchestration system"""

    def __init__(self):
        self.state: Optional[AgentState] = None
        self.coordinator: Optional[CoordinatorAgent] = None

    async def process_prescription(self, image_path: str) -> Dict[str, Any]:
        """Process a prescription through the multi-agent system"""

        # Initialize state
        self.state = AgentState(image_path=image_path)

        # Create coordinator
        self.coordinator = CoordinatorAgent(self.state)

        # Create and register specialized agents
        vision_agent = VisionAgent(self.state)
        reader_agent = TextReaderAgent(self.state)
        knowledge_agent = KnowledgeAgent(self.state)
        verifier_agent = VerifierAgent(self.state)
        smart_detail_agent = SmartDetailAgent(self.state)

        self.coordinator.register_agent(vision_agent)
        self.coordinator.register_agent(reader_agent)
        self.coordinator.register_agent(knowledge_agent)
        self.coordinator.register_agent(verifier_agent)
        self.coordinator.register_agent(smart_detail_agent)

        # Execute pipeline
        result = await self.coordinator.execute()

        return result

    def get_agent_communication_log(self) -> List[Dict]:
        """Get inter-agent communication history"""
        if not self.state:
            return []
        return [msg.dict() for msg in self.state.agent_messages]

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def main():
    """Example usage"""
    print("=" * 70)
    print("🤖 MediBot A2A Multi-Agent System (with SmartDetailAgent)")
    print("=" * 70)

    # Initialize system
    system = MediBotA2ASystem()

    # Process prescription
    image_path = "test_prescription.jpg"  # Your prescription image

    print(f"\nProcessing prescription: {image_path}\n")

    result = await system.process_prescription(image_path)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if result["success"]:
        print("\nProcessing successful!")
        print(f"\nPrescription Data:")
        print(json.dumps(result["prescription_data"], indent=2, ensure_ascii=False))
    else:
        print(f"\nProcessing failed: {result['error']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
