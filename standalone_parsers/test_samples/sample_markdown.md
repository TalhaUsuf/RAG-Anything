# RAG-Anything Architecture Guide

## Overview

RAG-Anything is a **multimodal RAG framework** built on top of LightRAG.

## Key Features

- Multimodal document processing
- Knowledge graph construction
- Vector similarity search
- Multiple parser support (MinerU, Docling, PaddleOCR)

## Architecture

### Parser Layer

The parser layer converts documents into structured content lists. Each parser implements four abstract methods:

1. `parse_pdf()` - Handle PDF documents
2. `parse_image()` - Handle image files
3. `parse_document()` - Route by file type
4. `check_installation()` - Verify dependencies

### Processing Pipeline

Documents flow through: Parser -> Content Separation -> Text Insertion + Modal Processing -> Knowledge Graph

## Configuration

Set `PARSER=mineru` in your environment to select the default parser.
