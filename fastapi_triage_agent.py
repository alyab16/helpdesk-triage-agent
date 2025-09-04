import os
from typing import Literal, List, TypedDict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------
# Environment & constants
# ---------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

NEG_THRESHOLD = float(os.getenv("TRIAGE_NEG_THRESHOLD", "0.80"))
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

device = "cuda" if torch.cuda.is_available() else (
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
)


# ---------------------------
# Graph state
# ---------------------------
class TicketState(TypedDict, total=False):
    ticket_text: str
    sentiment: Literal["POSITIVE", "NEGATIVE"]
    score: float
    action: Literal["ESCALATE", "RESPOND"]
    response: str


# ---------------------------
# Global singletons loaded at startup
# ---------------------------
tokenizer: Optional[AutoTokenizer] = None
clf_model: Optional[AutoModelForSequenceClassification] = None
compiled_graph = None


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)


# ---------------------------
# PyTorch classifier
# ---------------------------
def classify_with_torch(state: TicketState) -> TicketState:
    assert tokenizer is not None and clf_model is not None, "Model not initialized"

    text = state["ticket_text"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    clf_model.eval()
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = clf_model(**inputs).logits
        else:
            logits = clf_model(**inputs).logits

    probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
    neg_score, pos_score = probs  # SST-2: [NEG, POS]
    sentiment = "NEGATIVE" if neg_score >= pos_score else "POSITIVE"
    score = max(neg_score, pos_score)
    return {"sentiment": sentiment, "score": float(score)}


# ---------------------------
# Policy & routing
# ---------------------------
def decide(state: TicketState) -> TicketState:
    action = "ESCALATE" if (state["sentiment"] == "NEGATIVE" and state["score"] >= NEG_THRESHOLD) else "RESPOND"
    return {"action": action}

def route_after_decide(state: TicketState) -> str:
    return "compose_escalation" if state["action"] == "ESCALATE" else "compose_reply"


# ---------------------------
# LLM prompts
# ---------------------------
compose_reply_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful support agent. Be concise, empathetic, and practical."),
        ("user", """Customer message:
{ticket}

Sentiment: {sentiment} (confidence {score:.2f})

Write a short, friendly reply that either resolves the issue or asks ONLY for the minimum missing details.
If billing/shipping/account access is unclear, include a single bullet list of exactly what you need."""),
    ]
)

compose_escalation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a tier-2 escalation assistant. Summarize crisply for engineers."),
        ("user", """Customer message:
{ticket}

Sentiment: {sentiment} (confidence {score:.2f})

1) Provide a one-paragraph summary and likely root causes.
2) Bullet the top 3 diagnostics to collect.
3) Draft a 2-3 sentence customer reply acknowledging escalation and ETA."""),
    ]
)

def compose_reply(state: TicketState) -> TicketState:
    llm = get_llm()
    resp = (compose_reply_prompt | llm).invoke(
        {"ticket": state["ticket_text"], "sentiment": state["sentiment"], "score": state["score"]}
    )
    return {"response": resp.content}

def compose_escalation(state: TicketState) -> TicketState:
    llm = get_llm()
    resp = (compose_escalation_prompt | llm).invoke(
        {"ticket": state["ticket_text"], "sentiment": state["sentiment"], "score": state["score"]}
    )
    return {"response": resp.content}


# ---------------------------
# Build graph
# ---------------------------
def build_graph():
    workflow = StateGraph(TicketState)
    workflow.add_node("classify_with_torch", classify_with_torch)
    workflow.add_node("decide", decide)
    workflow.add_node("compose_reply", compose_reply)
    workflow.add_node("compose_escalation", compose_escalation)

    workflow.add_edge(START, "classify_with_torch")
    workflow.add_edge("classify_with_torch", "decide")
    workflow.add_conditional_edges(
        "decide",
        route_after_decide,
        {"compose_reply": "compose_reply", "compose_escalation": "compose_escalation"},
    )
    workflow.add_edge("compose_reply", END)
    workflow.add_edge("compose_escalation", END)
    return workflow.compile()


# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="Helpdesk Triage Agent", version="1.0.0")


@app.on_event("startup")
def on_startup():
    global tokenizer, clf_model, compiled_graph

    # Load classifier
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    clf_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    clf_model.to(device)
    if device == "cuda":
        clf_model.half()  # speedup

    # Build graph
    compiled_graph = build_graph()

    print(f"[startup] Device: {device}, Threshold: {NEG_THRESHOLD}")


# ---------------------------
# Schemas
# ---------------------------
class PredictIn(BaseModel):
    ticket_text: str = Field(..., description="Raw customer ticket text")

class PredictOut(BaseModel):
    sentiment: Literal["POSITIVE", "NEGATIVE"]
    score: float
    action: Literal["ESCALATE", "RESPOND"]
    response: str

class BatchIn(BaseModel):
    tickets: List[str]

class BatchOutItem(PredictOut):
    index: int

class BatchOut(BaseModel):
    results: List[BatchOutItem]


# ---------------------------
# Endpoints
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok", "device": device, "threshold": NEG_THRESHOLD}

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    assert compiled_graph is not None, "Graph not ready"
    result: TicketState = compiled_graph.invoke({"ticket_text": payload.ticket_text})
    return PredictOut(
        sentiment=result["sentiment"],
        score=result["score"],
        action=result["action"],
        response=result["response"],
    )

@app.post("/batch", response_model=BatchOut)
def batch(payload: BatchIn):
    assert compiled_graph is not None, "Graph not ready"
    out: List[BatchOutItem] = []
    for i, text in enumerate(payload.tickets):
        result: TicketState = compiled_graph.invoke({"ticket_text": text})
        out.append(BatchOutItem(
            index=i,
            sentiment=result["sentiment"],
            score=result["score"],
            action=result["action"],
            response=result["response"],
        ))
    return BatchOut(results=out)
