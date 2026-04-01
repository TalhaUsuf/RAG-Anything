# RAG-Anything System Architecture

## Overview

RAG-Anything is a multimodal RAG framework built on LightRAG. This document
describes the complete system architecture with diagrams and technical details.

## Processing Pipeline

![RAG-Anything Processing Pipeline](pipeline_diagram.png)

The pipeline processes documents through three main stages:
1. Parser extracts structured content from raw documents
2. Processor separates text from multimodal content and generates descriptions
3. LightRAG stores everything in the knowledge graph and vector database

## Supported Content Types

| Content Type | Processor | LLM Used | Input Format |
|-------------|-----------|----------|-------------|
| Text | None (direct insert) | N/A | Raw text string |
| Images | ImageModalProcessor | Vision LLM | Base64 encoded PNG/JPG |
| Tables | TableModalProcessor | Text LLM | Markdown table body |
| Equations | EquationModalProcessor | Text LLM | LaTeX string |
| Lists | GenericModalProcessor | Text LLM | JSON list items |

## Similarity Search

The vector similarity between a query $q$ and document chunk $d$ is computed using cosine similarity:

$$\text{sim}(q, d) = \frac{q \cdot d}{\|q\| \cdot \|d\|} = \frac{\sum_{i=1}^{n} q_i d_i}{\sqrt{\sum_{i=1}^{n} q_i^2} \cdot \sqrt{\sum_{i=1}^{n} d_i^2}}$$

Chunks with similarity above the threshold $\tau = 0.2$ are retrieved for context.

## Image Analysis Example

Below is an example of data captured by the system's monitoring component:

![Measurement Dashboard Example](measurement.png)

The Vision LLM (InternVL3-38B) analyzes images like this and generates
structured descriptions including identified objects, text, colors, and
relationships between visual elements.

## Knowledge Graph

The knowledge graph stores entities and relationships extracted from all
content types. The graph traversal algorithm uses BFS with depth limit:

$$\text{score}(e) = \sum_{p \in \text{paths}(q, e)} \prod_{(u,v) \in p} w(u,v) \cdot \alpha^{|p|}$$

where $\alpha = 0.85$ is the damping factor and $w(u,v)$ is the edge weight.

## Configuration

Set the following environment variables to configure the system:

| Variable | Default | Description |
|----------|---------|-------------|
| MINERU_API_URL | (see .env) | Remote MinerU endpoint |
| VLM_BASE_URL | (see .env) | Vision LLM endpoint |
| LLM_BASE_URL | (see .env) | Text LLM endpoint |
| EMBED_BASE_URL | (see .env) | Embedding endpoint |
| PARSER | mineru | Document parser selection |
