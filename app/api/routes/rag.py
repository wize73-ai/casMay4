"""
RAG (Retrieval-Augmented Generation) API Routes for CasaLingua

This module defines API endpoints for Retrieval-Augmented Generation
functionality, allowing translations and language processing to be
enhanced with external knowledge.

Author: Exygy Development Team
Version: 1.0.0
License: MIT
"""

import time
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path, Body, Request, status
from pydantic import BaseModel, Field

from app.core.rag.retriever import ingest_sources_from_config

from app.api.schemas.language import SupportedLanguage
from app.api.schemas.base import StatusEnum, MetadataModel, BaseResponse, ErrorDetail
from app.core.rag.generator import AugmentedGenerator
from app.core.rag.memory import ConversationMemory
from app.utils.auth import verify_api_key, get_current_user
def get_user_dev_mode(request: Request):
    if request.app.state.config.get("environment") == "development":
        return {"id": "dev-user"}
    return get_current_user(request)
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter()

# Custom RAG schemas
class RagTranslationRequest(BaseModel):
    """Request model for RAG-enhanced translation."""
    text: str = Field(..., description="Text to translate", min_length=1, max_length=8000)
    source_language: SupportedLanguage = Field(..., description="Source language code")
    target_language: SupportedLanguage = Field(..., description="Target language code")
    context_query: Optional[str] = Field(None, description="Optional query to retrieve specific context")
    document_ids: Optional[List[str]] = Field(None, description="Optional list of specific documents to use")
    max_retrieval_results: Optional[int] = Field(5, description="Maximum number of documents to retrieve")
    include_sources: bool = Field(True, description="Whether to include source information in response")
    model_id: Optional[str] = Field(None, description="Specific model ID to use")

class RagQueryRequest(BaseModel):
    """Request model for document retrieval query."""
    query: str = Field(..., description="Query text", min_length=1, max_length=500)
    language: SupportedLanguage = Field(..., description="Query language")
    max_results: Optional[int] = Field(10, description="Maximum number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters to apply to search")
    include_content: bool = Field(False, description="Whether to include document content in results")

class RagDocumentResult(BaseModel):
    """Model for retrieved document result."""
    document_id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    content_snippet: str = Field(..., description="Snippet of relevant content")
    relevance_score: float = Field(..., description="Relevance score (0-1)")
    source: Optional[str] = Field(None, description="Document source information")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    language: SupportedLanguage = Field(..., description="Document language")

class RagQueryResponse(BaseModel):
    """Response model for RAG query."""
    query: str = Field(..., description="Original query")
    results: List[RagDocumentResult] = Field(..., description="Retrieved documents")
    total_results: int = Field(..., description="Total number of matching documents")
    execution_time: float = Field(..., description="Query execution time in seconds")

class RagTranslationResult(BaseModel):
    """Model for RAG-enhanced translation result."""
    source_text: str = Field(..., description="Original text")
    translated_text: str = Field(..., description="Translated text")
    source_language: SupportedLanguage = Field(..., description="Source language code")
    target_language: SupportedLanguage = Field(..., description="Target language code")
    sources: Optional[List[RagDocumentResult]] = Field(None, description="Source documents used")
    confidence: float = Field(..., description="Translation confidence score")
    model_id: str = Field(..., description="ID of the model used")
    process_time: float = Field(..., description="Processing time in seconds")
    enhanced: bool = Field(..., description="Whether the translation was enhanced with context")

class RagChatRequest(BaseModel):
    """Request model for RAG-enhanced chat."""
    message: str = Field(..., description="User message", min_length=1, max_length=2000)
    language: SupportedLanguage = Field(..., description="Conversation language")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    context_query: Optional[str] = Field(None, description="Optional query to retrieve specific context")
    max_retrieval_results: Optional[int] = Field(3, description="Maximum number of documents to retrieve")
    include_sources: bool = Field(True, description="Whether to include source information in response")

class RagChatMessage(BaseModel):
    """Model for chat message."""
    role: str = Field(..., description="Message role (user or assistant)")
    content: str = Field(..., description="Message content")
    timestamp: float = Field(..., description="Message timestamp")

class RagChatResponse(BaseModel):
    """Response model for RAG-enhanced chat."""
    message: RagChatMessage = Field(..., description="Assistant's response message")
    conversation_id: str = Field(..., description="Conversation identifier")
    sources: Optional[List[RagDocumentResult]] = Field(None, description="Source documents used")
    process_time: float = Field(..., description="Processing time in seconds")

class RagDocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    language: Optional[SupportedLanguage] = Field(None, description="Detected document language")
    status: str = Field(..., description="Processing status")
    indexed: bool = Field(..., description="Whether document is indexed and available")
    chunk_count: int = Field(..., description="Number of chunks document was split into")
    upload_time: float = Field(..., description="Timestamp of upload")
    message: str = Field(..., description="Status message")


# RAG endpoints
@router.post(
    "/translate",
    response_model=BaseResponse[RagTranslationResult],
    status_code=status.HTTP_200_OK,
    summary="Translate text with RAG enhancement",
    description="Translates text using Retrieval-Augmented Generation to improve accuracy and context."
)
async def rag_translate(
    request: Request,
    background_tasks: BackgroundTasks,
    translation_request: RagTranslationRequest = Body(...),
    current_user: Dict[str, Any] = Depends(lambda request: get_user_dev_mode(request))
):
    """
    Translate text with retrieval-augmented generation to improve
    contextual accuracy and domain-specific terminology.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        model_manager = request.app.state.model_manager
        metrics = request.app.state.metrics
        
        # Get the retriever and generator from the processor
        retriever = processor.get_component("retriever")
        generator = processor.get_component("generator")
        
        if not retriever or not generator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG components not available"
            )
            
        # Retrieve relevant documents
        retrieval_query = translation_request.context_query or translation_request.text
        retrieved_docs = await retriever.retrieve(
            query=retrieval_query,
            language=translation_request.source_language,
            max_results=translation_request.max_retrieval_results,
            document_ids=translation_request.document_ids
        )
        
        # Generate augmented translation
        translation_result = await generator.translate(
            text=translation_request.text,
            source_language=translation_request.source_language,
            target_language=translation_request.target_language,
            reference_documents=retrieved_docs,
            model_id=translation_request.model_id
        )
        
        # Prepare response
        sources = None
        if translation_request.include_sources and retrieved_docs:
            sources = [
                RagDocumentResult(
                    document_id=doc.id,
                    title=doc.title,
                    content_snippet=doc.get_snippet(),
                    relevance_score=doc.score,
                    source=doc.source,
                    metadata=doc.metadata,
                    language=doc.language
                )
                for doc in retrieved_docs
            ]
            
        result = RagTranslationResult(
            source_text=translation_request.text,
            translated_text=translation_result.text,
            source_language=translation_request.source_language,
            target_language=translation_request.target_language,
            sources=sources,
            confidence=translation_result.confidence,
            model_id=translation_result.model_id,
            process_time=time.time() - start_time,
            enhanced=len(retrieved_docs) > 0
        )
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="rag_translation",
            operation="translate",
            duration=time.time() - start_time,
            input_size=len(translation_request.text),
            output_size=len(translation_result.text),
            success=True,
            metadata={
                "source_language": translation_request.source_language,
                "target_language": translation_request.target_language,
                "retrieved_docs": len(retrieved_docs),
                "enhanced": len(retrieved_docs) > 0
            }
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Translation completed successfully",
            data=result,
            metadata=MetadataModel(
                request_id=str(request.state.request_id),
                timestamp=translation_result.timestamp,
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=time.time() - start_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"RAG translation error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="rag_translation",
            operation="translate",
            duration=time.time() - start_time,
            input_size=len(translation_request.text),
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "source_language": translation_request.source_language,
                "target_language": translation_request.target_language
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation error: {str(e)}"
        )

@router.post(
    "/query",
    response_model=BaseResponse[RagQueryResponse],
    status_code=status.HTTP_200_OK,
    summary="Query knowledge base",
    description="Retrieves relevant documents from the knowledge base based on query."
)
async def rag_query(
    request: Request,
    background_tasks: BackgroundTasks,
    query_request: RagQueryRequest = Body(...),
    current_user: Dict[str, Any] = Depends(lambda request: get_user_dev_mode(request))
):
    """
    Query the knowledge base to retrieve relevant documents.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        metrics = request.app.state.metrics
        
        # Get the retriever from the processor
        retriever = processor.get_component("retriever")
        
        if not retriever:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG components not available"
            )
            
        # Retrieve relevant documents
        retrieved_docs = await retriever.retrieve(
            query=query_request.query,
            language=query_request.language,
            max_results=query_request.max_results,
            filters=query_request.filters
        )
        
        # Prepare document results
        results = [
            RagDocumentResult(
                document_id=doc.id,
                title=doc.title,
                content_snippet=doc.get_snippet() if not query_request.include_content else doc.content,
                relevance_score=doc.score,
                source=doc.source,
                metadata=doc.metadata,
                language=doc.language
            )
            for doc in retrieved_docs
        ]
        
        # Create query response
        query_response = RagQueryResponse(
            query=query_request.query,
            results=results,
            total_results=len(results),
            execution_time=time.time() - start_time
        )
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="rag_query",
            operation="query",
            duration=time.time() - start_time,
            input_size=len(query_request.query),
            output_size=len(results),
            success=True,
            metadata={
                "language": query_request.language,
                "result_count": len(results)
            }
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Query executed successfully",
            data=query_response,
            metadata=MetadataModel(
                request_id=str(request.state.request_id),
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=time.time() - start_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"RAG query error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="rag_query",
            operation="query",
            duration=time.time() - start_time,
            input_size=len(query_request.query),
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "language": query_request.language
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query error: {str(e)}"
        )

@router.post(
    "/chat",
    response_model=BaseResponse[RagChatResponse],
    status_code=status.HTTP_200_OK,
    summary="Chat with RAG enhancement",
    description="Engage in knowledge-enhanced conversation using RAG."
)
async def rag_chat(
    request: Request,
    background_tasks: BackgroundTasks,
    chat_request: RagChatRequest = Body(...),
    current_user: Dict[str, Any] = Depends(lambda request: get_user_dev_mode(request))
):
    """
    Chat with retrieval-augmented generation to enable knowledge-enhanced
    conversations about language, translations, and domain-specific topics.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        metrics = request.app.state.metrics
        
        # Get the retriever, generator, and memory from the processor
        retriever = processor.get_component("retriever")
        generator = processor.get_component("generator")
        memory = processor.get_component("memory")
        
        if not retriever or not generator or not memory:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG components not available"
            )
            
        # Get or create conversation
        conversation_id = chat_request.conversation_id
        if not conversation_id:
            conversation_id = await memory.create_conversation(user_id=current_user["id"])
        
        # Add user message to memory
        await memory.add_message(
            conversation_id=conversation_id,
            role="user",
            content=chat_request.message
        )
        
        # Retrieve conversation history
        conversation_history = await memory.get_conversation(conversation_id)
        
        # Retrieve relevant documents
        retrieval_query = chat_request.context_query or chat_request.message
        retrieved_docs = await retriever.retrieve(
            query=retrieval_query,
            language=chat_request.language,
            max_results=chat_request.max_retrieval_results
        )
        
        # Generate response
        response_text = await generator.generate_chat_response(
            message=chat_request.message,
            conversation_history=conversation_history,
            reference_documents=retrieved_docs,
            language=chat_request.language
        )
        
        # Add assistant response to memory
        message_timestamp = time.time()
        await memory.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response_text
        )
        
        # Prepare response message
        message = RagChatMessage(
            role="assistant",
            content=response_text,
            timestamp=message_timestamp
        )
        
        # Prepare sources
        sources = None
        if chat_request.include_sources and retrieved_docs:
            sources = [
                RagDocumentResult(
                    document_id=doc.id,
                    title=doc.title,
                    content_snippet=doc.get_snippet(),
                    relevance_score=doc.score,
                    source=doc.source,
                    metadata=doc.metadata,
                    language=doc.language
                )
                for doc in retrieved_docs
            ]
            
        # Create chat response
        chat_response = RagChatResponse(
            message=message,
            conversation_id=conversation_id,
            sources=sources,
            process_time=time.time() - start_time
        )
        
        # Record metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="rag_chat",
            operation="chat",
            duration=time.time() - start_time,
            input_size=len(chat_request.message),
            output_size=len(response_text),
            success=True,
            metadata={
                "language": chat_request.language,
                "conversation_id": conversation_id,
                "retrieved_docs": len(retrieved_docs),
                "enhanced": len(retrieved_docs) > 0
            }
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Chat response generated successfully",
            data=chat_response,
            metadata=MetadataModel(
                request_id=str(request.state.request_id),
                timestamp=message_timestamp,
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=time.time() - start_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except Exception as e:
        logger.error(f"RAG chat error: {str(e)}", exc_info=True)
        
        # Record error metrics in background
        background_tasks.add_task(
            metrics.record_pipeline_execution,
            pipeline_id="rag_chat",
            operation="chat",
            duration=time.time() - start_time,
            input_size=len(chat_request.message),
            output_size=0,
            success=False,
            metadata={
                "error": str(e),
                "language": chat_request.language,
                "conversation_id": chat_request.conversation_id
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat error: {str(e)}"
        )

class UploadDocumentInput(BaseModel):
    title: str
    language: Optional[SupportedLanguage] = None
    document_type: str
    metadata: Optional[Dict[str, Any]] = None

@router.post(
    "/documents/upload",
    response_model=BaseResponse[RagDocumentUploadResponse],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload document to knowledge base",
    description="Upload a document to be indexed in the RAG knowledge base."
)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    document: bytes = Body(...),
    info: UploadDocumentInput = Body(...),
    current_user: Dict[str, Any] = Depends(lambda request: get_user_dev_mode(request))
):
    """
    Upload a document to be indexed in the RAG knowledge base.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        processor = request.app.state.processor

        # Get the document manager from the processor
        doc_manager = processor.get_component("document_manager")

        if not doc_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document management component not available"
            )

        # Process document in background
        document_id = await doc_manager.create_document_id()

        # Start initial processing
        document_info = await doc_manager.start_document_processing(
            document_id=document_id,
            content=document,
            title=info.title,
            document_type=info.document_type,
            language=info.language,
            metadata=info.metadata or {},
            user_id=current_user["id"]
        )

        # Schedule background processing
        background_tasks.add_task(
            doc_manager.process_and_index_document,
            document_id=document_id,
            content=document,
            title=info.title,
            document_type=info.document_type,
            language=info.language,
            metadata=info.metadata or {},
            user_id=current_user["id"]
        )

        # Prepare response
        upload_response = RagDocumentUploadResponse(
            document_id=document_id,
            title=info.title,
            language=document_info.get("language"),
            status="processing",
            indexed=False,
            chunk_count=document_info.get("chunk_count", 0),
            upload_time=time.time(),
            message="Document uploaded and queued for processing"
        )

        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Document uploaded successfully",
            data=upload_response,
            metadata=MetadataModel(
                request_id=str(request.state.request_id),
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=time.time() - start_time
            ),
            errors=None,
            pagination=None
        )

        return response

    except Exception as e:
        logger.error(f"Document upload error: {str(e)}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document upload error: {str(e)}"
        )

@router.get(
    "/documents/{document_id}",
    response_model=BaseResponse[RagDocumentResult],
    status_code=status.HTTP_200_OK,
    summary="Get document by ID",
    description="Retrieve a document from the knowledge base by ID."
)
async def get_document(
    request: Request,
    document_id: str = Path(..., description="Document identifier"),
    include_content: bool = Query(False, description="Whether to include document content"),
    current_user: Dict[str, Any] = Depends(lambda request: get_user_dev_mode(request))
):
    """
    Retrieve a document from the knowledge base by ID.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        
        # Get the document manager from the processor
        doc_manager = processor.get_component("document_manager")
        
        if not doc_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document management component not available"
            )
            
        # Get document
        document = await doc_manager.get_document(document_id)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}"
            )
            
        # Prepare content
        content_snippet = document.get_snippet()
        if include_content:
            content_snippet = document.content
            
        # Prepare response
        document_result = RagDocumentResult(
            document_id=document.id,
            title=document.title,
            content_snippet=content_snippet,
            relevance_score=1.0,  # Direct retrieval
            source=document.source,
            metadata=document.metadata,
            language=document.language
        )
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Document retrieved successfully",
            data=document_result,
            metadata=MetadataModel(
                request_id=str(request.state.request_id),
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=time.time() - start_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document retrieval error: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document retrieval error: {str(e)}"
        )

@router.delete(
    "/documents/{document_id}",
    response_model=BaseResponse[None],
    status_code=status.HTTP_200_OK,
    summary="Delete document",
    description="Delete a document from the knowledge base."
)
async def delete_document(
    request: Request,
    document_id: str = Path(..., description="Document identifier"),
    current_user: Dict[str, Any] = Depends(lambda request: get_user_dev_mode(request))
):
    """
    Delete a document from the knowledge base.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        
        # Get the document manager from the processor
        doc_manager = processor.get_component("document_manager")
        
        if not doc_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document management component not available"
            )
            
        # Delete document
        success = await doc_manager.delete_document(document_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}"
            )
            
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Document deleted successfully",
            data=None,
            metadata=MetadataModel(
                request_id=str(request.state.request_id),
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=time.time() - start_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion error: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document deletion error: {str(e)}"
        )

@router.get(
    "/conversations/{conversation_id}",
    response_model=BaseResponse[List[RagChatMessage]],
    status_code=status.HTTP_200_OK,
    summary="Get conversation history",
    description="Retrieve history of a RAG conversation."
)
async def get_conversation(
    request: Request,
    conversation_id: str = Path(..., description="Conversation identifier"),
    limit: int = Query(20, description="Maximum number of messages to return"),
    current_user: Dict[str, Any] = Depends(lambda request: get_user_dev_mode(request))
):
    """
    Retrieve conversation history.
    """
    start_time = time.time()
    
    try:
        # Get application components from state
        processor = request.app.state.processor
        
        # Get the memory from the processor
        memory = processor.get_component("memory")
        
        if not memory:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Conversation memory component not available"
            )
            
        # Get conversation
        conversation = await memory.get_conversation(conversation_id, limit=limit)
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation not found: {conversation_id}"
            )
            
        # Prepare messages
        messages = [
            RagChatMessage(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg["timestamp"]
            )
            for msg in conversation
        ]
        
        # Create response
        response = BaseResponse(
            status=StatusEnum.SUCCESS,
            message="Conversation retrieved successfully",
            data=messages,
            metadata=MetadataModel(
                request_id=str(request.state.request_id),
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=time.time() - start_time
            ),
            errors=None,
            pagination=None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Conversation retrieval error: {str(e)}", exc_info=True)
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conversation retrieval error: {str(e)}"
        )


# New endpoint: Ingest documents from GitHub config
@router.post(
    "/documents/ingest-from-config",
    response_model=BaseResponse[Dict[str, Any]],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest documents from GitHub config",
    description="Fetch and index documents from GitHub repos listed in rag_sources.json."
)
async def ingest_documents_from_config(
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(lambda request: get_user_dev_mode(request))
):
    """
    Ingest RAG documents from configured GitHub repositories.
    """
    start_time = time.time()

    try:
        processor = request.app.state.processor
        doc_manager = processor.get_component("document_manager")

        if not doc_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document management component not available"
            )

        docs = ingest_sources_from_config()

        indexed_docs = []
        for doc in docs:
            doc_id = await doc_manager.create_document_id()
            await doc_manager.process_and_index_document(
                document_id=doc_id,
                content=doc["content"].encode("utf-8"),
                title=doc["title"],
                document_type="markdown",
                language=None,
                metadata={"source": "github-ingest"},
                user_id=current_user["id"]
            )
            indexed_docs.append(doc_id)

        return BaseResponse(
            status=StatusEnum.SUCCESS,
            message="GitHub documents ingested successfully",
            data={"indexed_document_ids": indexed_docs},
            metadata=MetadataModel(
                request_id=str(request.state.request_id),
                timestamp=time.time(),
                version=request.app.state.config.get("version", "1.0.0"),
                process_time=time.time() - start_time
            ),
            errors=None,
            pagination=None
        )

    except Exception as e:
        logger.error(f"GitHub ingest error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GitHub ingest error: {str(e)}"
        )