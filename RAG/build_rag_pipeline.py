#!/usr/bin/env python3
"""
Build RAG Pipeline for XML Data

This script loads XML documents from the data directory, chunks them,
creates embeddings, and builds a vector database using Golden Retriever.

Usage:
    python build_rag_pipeline.py [--data-dir DATA_DIR] [--persist-dir PERSIST_DIR]
                                  [--device DEVICE] [--chunk-size CHUNK_SIZE]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Ensure parent directory is in path for module imports when run directly
_PARENT_DIR = str(Path(__file__).parent.parent.absolute())
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from golden.golden_retriever import Golden_Retriever
from golden.golden_embeddings import Embedding
from core.config import RAG_PERSIST_DIR, RAG_EMBEDDING_MODEL, RAG_MAX_SEQ_LENGTH

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_text_from_xml(xml_file: Path) -> str:
    """
    Extract all text content from an XML file.

    Args:
        xml_file: Path to XML file

    Returns:
        Extracted text content
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract all text from the XML, joining with spaces
        text_parts = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                text_parts.append(elem.text.strip())
            if elem.tail and elem.tail.strip():
                text_parts.append(elem.tail.strip())

        text = " ".join(text_parts)

        # Add metadata about the source file
        file_name = xml_file.name
        header = f"Document: {file_name}\n\n"

        return header + text

    except Exception as e:
        logger.error(f"Error parsing {xml_file}: {e}")
        return ""


def load_xml_documents(data_dir: str) -> List[str]:
    """
    Load all XML documents from the data directory.

    Args:
        data_dir: Path to directory containing XML files

    Returns:
        List of document texts
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    documents = []

    # Find all XML files recursively
    xml_files = list(data_path.rglob("*.xml"))

    logger.info(f"Found {len(xml_files)} XML files in {data_dir}")

    for xml_file in tqdm(xml_files, desc="Loading XML files"):
        text = extract_text_from_xml(xml_file)
        if text:
            documents.append(text)
            logger.debug(f"Loaded {xml_file.name}: {len(text)} characters")

    logger.info(f"Successfully loaded {len(documents)} documents")
    return documents


def build_rag_pipeline(
    data_dir: str,
    persist_dir: str = RAG_PERSIST_DIR,
    model_id: str = RAG_EMBEDDING_MODEL,
    chunk_size: int = RAG_MAX_SEQ_LENGTH,
    chunk_overlap: int = 20,
    device: str = "cpu",
    batch_size: str = "auto",
    max_batch_size: int = 64,
):
    """
    Build the RAG pipeline by loading documents, creating embeddings, and storing in vector DB.

    Args:
        data_dir: Directory containing XML documents
        persist_dir: Directory to persist the vector database
        model_id: Embedding model to use
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        device: Device to use for embeddings ("cpu" or "cuda")
        batch_size: Batch size for embedding generation ("auto" or integer)
        max_batch_size: Maximum batch size to try
    """
    logger.info("=" * 80)
    logger.info("Building RAG Pipeline")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Persist directory: {persist_dir}")
    logger.info(f"Embedding model: {model_id}")
    logger.info(f"Chunk size: {chunk_size}")
    logger.info(f"Chunk overlap: {chunk_overlap}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {batch_size}")
    logger.info("=" * 80)

    # Step 1: Load documents
    logger.info("\n[1/3] Loading XML documents...")
    documents = load_xml_documents(data_dir)

    if not documents:
        logger.error("No documents loaded. Exiting.")
        return

    total_chars = sum(len(doc) for doc in documents)
    logger.info(f"Total characters loaded: {total_chars:,}")

    # Step 2: Create embedding function
    logger.info("\n[2/3] Initializing embedding model...")
    embedding_fn = Embedding(
        model_id=model_id,
        tokenizer_id=model_id,
        max_seq_length=chunk_size,
        device=device,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        trust_remote_code=True,
    )
    logger.info(f"Embedding model initialized: {model_id}")

    # Step 3: Build vector store
    logger.info("\n[3/3] Building vector database...")
    logger.info("This may take a while depending on the size of your dataset...")

    # Remove old database if it exists
    if os.path.exists(persist_dir):
        logger.warning(f"Removing existing database at {persist_dir}")
        import shutil
        shutil.rmtree(persist_dir)

    # Create the vector store
    db = Golden_Retriever.from_texts(
        texts=documents,
        embedding=embedding_fn,
        persist_directory=persist_dir,
        collection_name="rag_xml_documents",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        device=device,
        do_chunking=True,
        language="",  # Empty string for generic text splitter
    )

    # Get statistics
    doc_count = db._collection.count()
    logger.info(f"Vector database created successfully!")
    logger.info(f"Total chunks stored: {doc_count:,}")
    logger.info(f"Database location: {persist_dir}")

    # Test the retriever
    logger.info("\n[Test] Running sample query...")
    test_query = "procurement"
    results = db.similarity_search(test_query, k=3)
    logger.info(f"Query: '{test_query}'")
    logger.info(f"Found {len(results)} results")
    if results:
        logger.info(f"Top result preview: {results[0].page_content[:200]}...")

    logger.info("\n" + "=" * 80)
    logger.info("RAG Pipeline build complete!")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Build RAG pipeline from XML documents"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/ec2-user/git/data_sanitizer/data",
        help="Directory containing XML documents"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=RAG_PERSIST_DIR,
        help="Directory to persist vector database"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=RAG_EMBEDDING_MODEL,
        help="Embedding model ID"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=RAG_MAX_SEQ_LENGTH,
        help="Size of text chunks"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=20,
        help="Overlap between chunks"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for embeddings"
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default="auto",
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=64,
        help="Maximum batch size to try"
    )

    args = parser.parse_args()

    build_rag_pipeline(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir,
        model_id=args.model_id,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        device=args.device,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
    )


if __name__ == "__main__":
    main()
