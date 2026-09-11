from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(default="default", min_length=1, max_length=100)

class AskResponse(BaseModel):
    response: str
    source_type: str
    grounded: bool
    session_id: str
    trace_id: str

class AddDocumentRequest(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=2, max_length=20000)

class AddDocumentResponse(BaseModel):
    document_id: str
    accepted: bool
    message: str
    trace_id: str

class WebSocketRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)