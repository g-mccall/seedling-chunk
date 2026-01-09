#!/usr/bin/env python3
"""

Extracted from Cognee's cognify methodology

This server accepts text via HTTP POST requests and creates
relationally-aware chunks that preserve semantic boundaries (paragraphs/sentences).
"""

import os
import re
import json
from functools import wraps
from uuid import uuid4, uuid5, NAMESPACE_OID
from typing import Iterator, Tuple, Dict, Any, Optional
from flask import Flask, request, jsonify
from zep_cloud.client import Zep
from zep_cloud import EpisodeData




SENTENCE_ENDINGS = r"[.;!?…。！？]"
PARAGRAPH_ENDINGS = r"[\n\r]"


def chunk_by_word(data: str) -> Iterator[Tuple[str, str]]:
    """
    Chunk text into words and sentence endings, preserving whitespace.
    Outputs can be joined with "" to recreate the original input.

    Yields:
        (word, type) where type is "word", "sentence_end", or "paragraph_end"
    """
    current_chunk = ""
    i = 0

    while i < len(data):
        character = data[i]
        current_chunk += character

        if character == " ":
            yield (current_chunk, "word")
            current_chunk = ""
            i += 1
            continue

        if re.match(SENTENCE_ENDINGS, character):
            # Look ahead for whitespace
            next_i = i + 1
            while next_i < len(data) and data[next_i] == " ":
                current_chunk += data[next_i]
                next_i += 1

            is_paragraph_end = next_i < len(data) and re.match(PARAGRAPH_ENDINGS, data[next_i])
            yield (current_chunk, "paragraph_end" if is_paragraph_end else "sentence_end")
            current_chunk = ""
            i = next_i
            continue

        i += 1

    if current_chunk:
        yield (current_chunk, "word")


# === SENTENCE CHUNKING (Middle Level) ===

def get_word_size(word: str) -> int:
    """
    Calculate word size in tokens. Simplified version - counts words as tokens.
    For production use, integrate with an actual tokenizer (e.g., tiktoken).
    """
    # Simple approximation: split on spaces and punctuation
    # In Cognee, this uses the embedding engine's tokenizer
    tokens = len(word.split()) or 1
    return max(tokens, 1)


def chunk_by_sentence(
    data: str, maximum_size: Optional[int] = None
) -> Iterator[Tuple[str, str, int, Optional[str]]]:
    """
    Splits text into sentences while preserving word and paragraph boundaries.

    Yields:
        (paragraph_id, sentence_text, sentence_size, end_type)
    """
    sentence = ""
    paragraph_id = str(uuid4())
    sentence_size = 0
    word_type_state = None

    for word, word_type in chunk_by_word(data):
        word_size = get_word_size(word)

        if word_type in ["paragraph_end", "sentence_end"]:
            word_type_state = word_type
        else:
            for character in word:
                if character.isalpha():
                    word_type_state = word_type
                    break

        if maximum_size and (sentence_size + word_size > maximum_size):
            yield (paragraph_id, sentence, sentence_size, word_type_state)
            sentence = word
            sentence_size = word_size

        elif word_type in ["paragraph_end", "sentence_end"]:
            sentence += word
            sentence_size += word_size
            paragraph_id = str(uuid4()) if word_type == "paragraph_end" else paragraph_id

            yield (paragraph_id, sentence, sentence_size, word_type_state)
            sentence = ""
            sentence_size = 0
        else:
            sentence += word
            sentence_size += word_size

    if len(sentence) > 0:
        if maximum_size and sentence_size > maximum_size:
            raise ValueError(f"Input word longer than chunking size {maximum_size}.")

        section_end = "sentence_cut" if word_type_state == "word" else word_type_state
        yield (paragraph_id, sentence, sentence_size, section_end)


# === PARAGRAPH CHUNKING (Top Level) ===

def chunk_by_paragraph(
    data: str,
    max_chunk_size: int,
    batch_paragraphs: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Chunk text by paragraph while enabling exact text reconstruction.

    This groups sentences into paragraph-based chunks up to max_chunk_size,
    respecting semantic boundaries.

    Yields:
        Dict with keys: text, chunk_size, chunk_id, paragraph_ids, chunk_index, cut_type
    """
    current_chunk = ""
    chunk_index = 0
    paragraph_ids = []
    last_cut_type = "default"
    current_chunk_size = 0

    for paragraph_id, sentence, sentence_size, end_type in chunk_by_sentence(
        data, maximum_size=max_chunk_size
    ):
        if current_chunk_size > 0 and (current_chunk_size + sentence_size > max_chunk_size):
            # Yield current chunk
            chunk_dict = {
                "text": current_chunk,
                "chunk_size": current_chunk_size,
                "chunk_id": str(uuid5(NAMESPACE_OID, current_chunk)),
                "paragraph_ids": paragraph_ids,
                "chunk_index": chunk_index,
                "cut_type": last_cut_type,
            }

            yield chunk_dict

            # Start new chunk with current sentence
            paragraph_ids = []
            current_chunk = ""
            current_chunk_size = 0
            chunk_index += 1

        paragraph_ids.append(paragraph_id)
        current_chunk += sentence
        current_chunk_size += sentence_size

        # Handle end of paragraph
        if end_type in ("paragraph_end", "sentence_cut") and not batch_paragraphs:
            chunk_dict = {
                "text": current_chunk,
                "chunk_size": current_chunk_size,
                "paragraph_ids": paragraph_ids,
                "chunk_id": str(uuid5(NAMESPACE_OID, current_chunk)),
                "chunk_index": chunk_index,
                "cut_type": end_type,
            }
            yield chunk_dict
            paragraph_ids = []
            current_chunk = ""
            current_chunk_size = 0
            chunk_index += 1

        if not end_type:
            end_type = "default"

        last_cut_type = end_type

    # Yield any remaining text
    if current_chunk:
        chunk_dict = {
            "text": current_chunk,
            "chunk_size": current_chunk_size,
            "chunk_id": str(uuid5(NAMESPACE_OID, current_chunk)),
            "paragraph_ids": paragraph_ids,
            "chunk_index": chunk_index,
            "cut_type": "sentence_cut" if last_cut_type == "word" else last_cut_type,
        }

        yield chunk_dict


# === TEXT PROCESSING ===

def extract_chunks_from_text(
    text: str,
    document_name: str = "document",
    max_chunk_size: int = 200,
    batch_paragraphs: bool = True,
) -> list[Dict[str, Any]]:
    """
    Extract relationally-aware chunks from text.

    This follows Cognee's methodology:
    1. Chunk by paragraph with semantic boundaries
    2. Create chunk dictionaries with metadata

    Args:
        text: The text to chunk
        document_name: Optional name for the document
        max_chunk_size: Maximum chunk size in tokens (default: 200)
        batch_paragraphs: Whether to batch multiple paragraphs per chunk

    Returns:
        List of chunk dictionaries
    """
    chunks = []

    for chunk_data in chunk_by_paragraph(
        text,
        max_chunk_size=max_chunk_size,
        batch_paragraphs=batch_paragraphs,
    ):
        chunk = {
            "chunk_id": chunk_data["chunk_id"],
            "chunk_index": chunk_data["chunk_index"],
            "chunk_size": chunk_data["chunk_size"],
            "cut_type": chunk_data["cut_type"],
            "document_name": document_name,
            "text": chunk_data["text"],
        }
        chunks.append(chunk)

    return chunks


# === ZEP INTEGRATION ===

def batch_chunks_for_zep(chunks: list[Dict[str, Any]], batch_size: int = 20) -> list[list[EpisodeData]]:
    """
    Convert chunks to EpisodeData and batch into groups.

    Args:
        chunks: List of chunk dictionaries from extract_chunks_from_text
        batch_size: Number of episodes per batch (default: 20)

    Returns:
        List of batches, each containing up to batch_size EpisodeData objects
    """
    episodes = [
        EpisodeData(
            data=json.dumps(chunk),
            type="json"
        )
        for chunk in chunks
    ]

    return [episodes[i:i + batch_size] for i in range(0, len(episodes), batch_size)]


def upload_chunks_to_zep(
    chunks: list[Dict[str, Any]],
    graph_id: str = None,
) -> list[str]:
    """
    Upload chunks to Zep graph database in batches of 20.
    Waits for each batch to complete before uploading the next.

    Args:
        chunks: List of chunk dictionaries
        graph_id: Zep graph ID (defaults to ZEP_GRAPH_ID env var)

    Returns:
        List of task_ids for tracking batch processing
    """
    api_key = os.environ.get("ZEP_API_KEY")
    graph_id = graph_id or os.environ.get("ZEP_GRAPH_ID")

    if not api_key:
        raise ValueError("ZEP_API_KEY environment variable not set")
    if not graph_id:
        raise ValueError("ZEP_GRAPH_ID environment variable not set")

    client = Zep(api_key=api_key)
    batches = batch_chunks_for_zep(chunks)
    task_ids = []

    for i, batch in enumerate(batches):
        print(f"Uploading batch {i + 1}/{len(batches)}...")
        result = client.graph.add_batch(episodes=batch, graph_id=graph_id)
        batch_task_ids = [episode.task_id for episode in result]
        task_ids.extend(batch_task_ids)

    return task_ids


# === HTTP SERVER ===

app = Flask(__name__)


def require_api_token(f):
    """Decorator to require API token authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        api_token = os.environ.get("RAILWAY_API_TOKEN")

        if not api_token:
            return jsonify({"error": "Server misconfigured: API token not set"}), 500

        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return jsonify({"error": "Missing Authorization header"}), 401

        if auth_header.startswith("Bearer "):
            provided_token = auth_header[7:]
        else:
            provided_token = auth_header

        if provided_token != api_token:
            return jsonify({"error": "Invalid API token"}), 401

        return f(*args, **kwargs)
    return decorated


def process_chunk_request(data: dict):
    """
    Process a chunking request from JSON data and upload to Zep.

    Args:
        data: Dictionary containing 'body.text' and optional parameters

    Returns:
        Tuple of (response_dict, status_code)
    """
    # Extract from nested body structure
    if "body" in data:
        data = data["body"]

    if "text" not in data:
        return {"error": "Missing required field: text"}, 400

    text = data["text"]
    document_name = data.get("document_name", "document")
    max_chunk_size = data.get("max_chunk_size", 200)
    batch_paragraphs = data.get("batch_paragraphs", True)
    graph_id = data.get("graph_id")

    if not isinstance(text, str):
        return {"error": "Field 'text' must be a string"}, 400

    if not isinstance(max_chunk_size, int) or max_chunk_size <= 0:
        return {"error": "Field 'max_chunk_size' must be a positive integer"}, 400

    try:
        chunks = extract_chunks_from_text(
            text=text,
            document_name=document_name,
            max_chunk_size=max_chunk_size,
            batch_paragraphs=batch_paragraphs,
        )

        task_ids = upload_chunks_to_zep(chunks, graph_id)

        response = {
            "chunks": chunks,
            "total_chunks": len(chunks),
            "document_name": document_name,
            "max_chunk_size": max_chunk_size,
            "batch_paragraphs": batch_paragraphs,
            "zep_task_ids": task_ids,
            "zep_batches": (len(chunks) + 19) // 20,
        }

        return response, 200

    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/api", methods=["POST"])
@require_api_token
def api_chunk():
    """
    Main API endpoint for chunking text and uploading to Zep.

    Expects JSON body:
    {
        "text": "The text to chunk",
        "document_name": "optional document name",
        "max_chunk_size": 200,  // optional, default 200
        "batch_paragraphs": true,  // optional, default true
        "graph_id": "optional graph id override"  // optional, uses ZEP_GRAPH_ID env var if not provided
    }

    Returns JSON:
    {
        "chunks": [...],
        "total_chunks": N,
        "document_name": "...",
        "max_chunk_size": N,
        "batch_paragraphs": bool,
        "zep_task_ids": [...],
        "zep_batches": N
    }
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    response, status_code = process_chunk_request(data)
    return jsonify(response), status_code


@app.route("/chunk", methods=["POST"])
@require_api_token
def chunk_text():
    """
    HTTP endpoint to chunk text and upload to Zep (alias for /api).
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    response, status_code = process_chunk_request(data)
    return jsonify(response), status_code


@app.route("/health", methods=["GET"])
@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "text-chunker"})


# === MAIN PROGRAM ===

def main():
    """Main program entry point - starts the HTTP server"""
    # Railway sets PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "false").lower() == "true"

    print("\n" + "=" * 80)
    print("TEXT RELATIONAL CHUNKER - HTTP SERVER")
    print("Based on Cognee's cognify methodology")
    print("=" * 80)
    print(f"\nServer starting on http://{host}:{port}")
    print("\nEndpoints:")
    print("  POST /api    - Chunk text and upload to Zep (JSON body with 'text' field)")
    print("  POST /chunk  - Alias for /api")
    print("  GET  /health - Health check")
    print("  GET  /       - Health check")
    print("\nRequired environment variables:")
    print("  ZEP_API_KEY   - Your Zep Cloud API key")
    print("  ZEP_GRAPH_ID  - Target graph ID for uploads")
    print("\n" + "=" * 80 + "\n")

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
