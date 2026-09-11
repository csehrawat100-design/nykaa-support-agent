Final Capstone — Nykaa Domain Support Agent

Track: Nykaa (E-commerce & Retail)

A production-minded customer-support agent combining deterministic local RAG, CrewAI multi-agent orchestration, LangChain session memory, guardrails, FastAPI, structured JSONL logging, MOCK_LLM evaluation, AutoGen review, governance, and response caching.

Architecture

User request
  ↓
FastAPI /ask or WebSocket
  ↓
Input guardrails (PII masking + prompt-injection detection)
  ↓
Runtime governance / budget
  ↓
Normalized response cache
  ↓
CrewAI
  ├─ Retrieval Agent → sentence RAG
  ├─ Lookup Agent → synthetic order dataset
  └─ Response Composer
  ↓
Pydantic structured-response validation
  ↓
AutoGen review
  ├─ Policy Compliance Reviewer
  └─ Final Editor
  ↓
Cache store
  ↓
Final response

The complete project is intended to satisfy the capstone through one repository. The graded path uses deterministic MOCK_LLM mode, with local embeddings/vector storage and no external LLM API key or paid account required. CrewAI telemetry is disabled for zero-network execution.

1. Dataset

The synthetic ORDERS dataset is deterministic.

Property

Choice

Records

50

Seed

20260000

Amount range

₹499–₹24,999

Category weights

Apparel 24 / Electronics 20 / Home 18 / Footwear 20 / Beauty 18

Status weights

Placed 20 / Shipped 25 / Delivered 35 / Returned 10 / Refunded 10

Delayed probability

0.20

Random generator

local random.Random(seed)

IDs

NYK-0001, NYK-0002, ...

Fields are record_id, category, status, order_value_inr, days_since_created, and delayed_shipment. The ₹499–₹24,999 range covers realistic synthetic purchases from lower-priced products through higher-value electronics/home purchases. The generator uses seeded randomness rather than hand-editing individual records.

2. Knowledge Base

The project contains 12 authored policy documents covering:

Return window by product category

COD refund timelines

Delivery SLAs

Reverse-pickup eligibility

Warranty terms by category

Order-cancellation policy

Loyalty-points redemption policy

Payment-failure/retry policy

Size-exchange policy

Damaged-item claim process

International shipping restrictions

Customer-support escalation matrix

Each document contains 2–5 sentences as required by the brief.

3. RAG Core

The RAG implementation separates document loading, chunking, embeddings, vector storage, retrieval, generation, and evaluation.

Both required chunking strategies are implemented:

fixed-size with overlap

sentence-based

Each is embedded with a free local SentenceTransformers model and indexed in a separate ChromaDB collection:

fixed_size_collection
sentence_collection

Chunks preserve source, document_id, chunk_id, and chunk_strategy metadata and are indexed with collection.upsert().

Chroma distance is converted to similarity with:

similarity = 1 / (1 + distance)

Threshold calibration

Measured values:

Fixed-size: in 0.6360, 0.6629, 0.6734 | out 0.3621, 0.3753 | threshold 0.5056
Sentence:   in 0.6040, 0.6904, 0.6906 | out 0.3648, 0.3407 | threshold 0.4844

The selected later configuration is:

strategy  = sentence
threshold = 0.4844

The code-level DEFAULT_SIMILARITY_THRESHOLD = 0.45 remains a default/fallback constant and is not automatically overwritten by the experimental calibration.

Strategy comparison

Document-level evaluation after mapping chunks to parent documents and deduplicating:

Strategy

Mean precision

Mean recall

Fixed-size

0.2833

1.0000

Sentence

0.4067

1.0000

Sentence-based chunking was selected because it improved mean precision while maintaining perfect recall. Subsequent agent requests therefore use the sentence collection and threshold 0.4844.

Grounded generation uses retrieved evidence only. Under MOCK_LLM, retrieval similarity is the grounding signal and the calibrated threshold controls the fallback.

4. Order Lookup

The deterministic lookup tool uses the synthetic order dataset.

The escalation score is:

0.60 * delayed_shipment_signal
+ 0.40 * (days_since_created / 30)

Escalation is recommended when:

escalation_score > 0.70

The larger delayed-shipment weight reflects its stronger direct support-risk signal. Only the Lookup Agent receives the check_order_status capability.

5. CrewAI

The crew contains:

Retrieval Agent — sentence RAG collection and calibrated threshold.

Lookup Agent — deterministic check_order_status tool.

Response Composer — combines supported outputs without inventing facts.

The crew runs through .kickoff() and demonstrates both RAG retrieval and order lookup.

The custom MockLLM extends CrewAI's BaseLLM. CrewAI telemetry is disabled using:

CREWAI_DISABLE_TELEMETRY=true
OTEL_SDK_DISABLED=true

This preserves the required zero-network graded mode.

6. Session Memory

Session memory uses LangChain's:

InMemoryChatMessageHistory

RunnableWithMessageHistory

State is carried across turns within a process/session. A separate fresh-conversation demonstration confirms that previous state is absent/reset. Persistence across restarts is not required.

A RunnableWithMessageHistory deprecation warning may appear and is expected for this implementation.

7. Structured Output

Every crew response is validated against the canonical Pydantic ResponseFormat schema:

response
source_type
grounded

source_type is restricted to:

policy
order
mixed
fallback

Unknown fields and invalid source types are rejected.

8. Guardrails

PII masking

The input guardrail masks the fixed-format PII fields required by the brief:

phone number

payment-card last four digits

The same masked text is used for model input and request logging. Names and delivery addresses are treated as out of scope for reliable keyless masking; only fabricated examples are used.

Prompt injection

A prompt-injection guardrail detects and rejects deliberately malicious/instruction-overriding requests.

Groundedness

An output-side groundedness control refuses unsupported answers when sufficient retrieved evidence is unavailable.

Each guardrail has a deliberate firing demonstration in the tests/transcripts.

9. FastAPI

The backend exposes:

POST /ask

POST /add-document

WebSocket /ws/chat

Requests use Pydantic models. The WebSocket handler catches WebSocketDisconnect, allowing the server to continue serving other clients.

10. Structured Logging

Every request produces one JSON-Lines log entry containing trace ID, timing, method/path, status, and masked request text.

Logs are written to:

logs/requests.jsonl

Raw fixed-format PII must not reach the log.

11. Evaluation

A 15-query evaluation set covers every required KB topic and includes at least two deliberately out-of-scope/edge-case queries.

Every query is scored under deterministic MOCK_LLM judging for:

Accuracy

Grounding

Completeness

Safety

The evaluation reports all four scores per query and the average of each metric.

12. AutoGen Review

The CrewAI draft is reviewed by a two-agent AutoGen Round Robin team:

CrewAI draft
  ↓
Policy Compliance Reviewer
  ↓
Final Editor
  ↓
ReviewVerdict

The team uses RoundRobinGroupChat, max_turns=2, MaxMessageTermination(3), and StructuredMessage[ReviewVerdict]. The Final Editor uses Pydantic structured output.

ReviewVerdict contains:

approved: bool
final_answer: str
reason: str

The review stage is demonstrated both approving a grounded draft unchanged and revising a deliberately ungrounded draft.

13. Governance

Application layer

Least autonomy is enforced: only the Lookup Agent can call check_order_status. Retrieval and Composer are not wired with that capability.

Risk

The system is classified as Medium risk under the supplied scheme because it performs customer-support interactions and order-support operations. It is not used for medical, hiring, or financial decision-making.

Runtime

A per-request token/cost budget is enforced before downstream work. An oversized request is deliberately rejected rather than silently exceeding the cap.

14. Cache

Responses are cached in memory using normalized query text.

Budget
  ↓
Cache lookup
  ↓
CrewAI
  ↓
AutoGen review
  ↓
Cache store

A repeated normalized query produces a real cache hit and avoids redundant downstream work. Cache statistics include hits, misses, entries, and calls_avoided.

15. Repository Structure

nykaa-support-agent/
agents/
api/
autogen_review/
caching/
data/
dataset/
evaluation/
governance/
guardrails/
integration/
knowledge_base/
rag/
tests/
transcripts/

16. Setup

Python 3.12 is used for the project.

python -m pip install -r requirements.txt

No API key is required for the graded MOCK_LLM path.

17. Build the RAG Index

From the repository root:

python -m rag.build_index

This builds both Chroma collections.

18. Run Tests

pytest -q

The final integrated implementation has been tested across structured output, API, memory, governance, caching, AutoGen, and CrewAI integration paths.

19. Run CrewAI

python -m agents.crew

20. Start FastAPI

uvicorn api.main:app --reload

Example:

$body = @{query="What is the return window for Beauty products?"} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body

A grounded policy response should state that Beauty products are returnable within 7 days when the item is unused and sealed.

21. Reproducibility

The graded path is deterministic through:

seeded synthetic data

local SentenceTransformers embeddings

local ChromaDB

deterministic CrewAI MockLLM

deterministic AutoGen review client

deterministic evaluation judge

no external LLM API key

disabled CrewAI telemetry

22. Transcripts

The transcripts/ directory documents the experiments and demonstrations required by the capstone, including dataset validation, RAG calibration, precision/recall, lookup, CrewAI, memory, structured output, guardrails, FastAPI, logging, evaluation, AutoGen, governance, and caching.

23. Final Configuration

Selected chunking strategy: sentence
Selected calibrated threshold: 0.4844

The repository is intended to be submitted as the single public GitHub repository containing the dataset, knowledge base, RAG core, CrewAI orchestration, memory, guardrails, API, evaluation, AutoGen review, governance, cache, tests, and transcripts.