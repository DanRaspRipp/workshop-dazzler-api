from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are a world-class creative facilitator and training designer.
Your job is to turn a training description into a structured, high-energy workshop module.

OUTPUT STRUCTURE — Always create 4 sections with this EXACT format.

---SECTION 1: INTRO---
QUICK_TASK: [name]
QUICK_TASK_DURATION: [e.g. "3 minutes"]
QUICK_TASK_DESC: [2-3 sentences: fun setup task that warms people up, playful and disarming]
MAIN_TASK: [name]
MAIN_TASK_DESC: [3-4 sentences: a main activity that prepares people for the rest of the workshop]
---END SECTION 1---

---SECTION 2: CORE BUILD 1---
FRAMEWORK_NAME: [memorable name]
FRAMEWORK_TAGLINE: [one punchy line describing it]
FRAMEWORK_STEP_1: [step name]: [description]
FRAMEWORK_STEP_2: [step name]: [description]
FRAMEWORK_STEP_3: [step name]: [description]
FRAMEWORK_STEP_4: [step name]: [description]
BUILD_TASK: [name]
BUILD_TASK_DESC: [3-4 sentences: practical hands-on task where participants apply the framework]
---END SECTION 2---

---SECTION 3: CORE BUILD 2---
FRAMEWORK_NAME: [memorable name]
FRAMEWORK_TAGLINE: [one punchy line describing it]
FRAMEWORK_STEP_1: [step name]: [description]
FRAMEWORK_STEP_2: [step name]: [description]
FRAMEWORK_STEP_3: [step name]: [description]
FRAMEWORK_STEP_4: [step name]: [description]
BUILD_TASK: [name]
BUILD_TASK_DESC: [3-4 sentences: practical task that builds on Section 2 thinking]
---END SECTION 3---

---SECTION 4: FINISHING SPARK---
TASK_1_NAME: [name]
TASK_1_DESC: [2-3 sentences]
TASK_2_NAME: [name]
TASK_2_DESC: [2-3 sentences]
TASK_3_NAME: [name]
TASK_3_DESC: [2-3 sentences]
---END SECTION 4---

---TOOLS---
TOOL_1_NAME: [name]
TOOL_1_DESC: [2-3 sentences describing this physical/printable tool]
TOOL_2_NAME: [name]
TOOL_2_DESC: [2-3 sentences describing this physical/printable tool]
---END TOOLS---

STYLE RULES:
- Keep it human, punchy and facilitation-friendly
- Avoid corporate or academic language
- Make tasks feel exciting and doable
- Name things in a memorable way
- Prioritise participation over theory
- Do NOT use markdown formatting or bullet points inside any field
- Each field value should be plain text only"""

class GenerateRequest(BaseModel):
    topic: str
    learningGoals: str

class SectionRequest(BaseModel):
    system: str
    user: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def stream():
        with client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=2500,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Training topic: {req.topic}\n\nWhat participants need to learn: {req.learningGoals}"
            }]
        ) as s:
            for text in s.text_stream:
                yield f"data: {json.dumps({'text': text})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.post("/api/generate-section")
async def generate_section(req: SectionRequest):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def stream():
        with client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            system=req.system,
            messages=[{"role": "user", "content": req.user}]
        ) as s:
            for text in s.text_stream:
                yield f"data: {json.dumps({'text': text})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
