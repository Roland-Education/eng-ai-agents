# Architectural Analysis of RAGFlow

**Roland Chi**
**Intro to AI - Spring 2026**

---

## 1. Deep Document Understanding vs Naive Chunking

Deep document understanding outperforms fixed-size chunking because documents encode meaning through structure, not just token sequences. A financial table split mid-row by a 512-token window loses its relational semantics entirely - the numbers become unmoored from their column headers, rendering retrieved chunks misleading rather than merely incomplete.

**Retrieval fidelity.** Layout-aware parsing preserves semantic units: a table remains a table, a list item retains its parent context, and a section heading stays attached to the paragraph it governs. This means the retrieval system returns chunks that are self-contained and interpretable by the downstream LLM, reducing hallucination caused by decontextualized fragments.

**Index design.** Structure-aware chunks carry richer metadata (document type, section hierarchy, table coordinates) that can be stored as filterable fields in the index. This enables hybrid queries that combine semantic similarity with structural constraints - for example, "find clauses in Section 4 of the lease agreement" - which is impossible with flat, position-indexed chunks.

**Preprocessing cost.** Deep parsing is significantly more expensive. OCR, layout detection (often via vision models or heuristic engines like DeepDoc), and table extraction add latency and compute. The trade-off is front-loaded: you pay once at ingestion to avoid repeated retrieval failures at query time. For corpora that are queried frequently, this amortization is strongly favorable. For rarely-queried or ephemeral documents, naive chunking may be rational.

---

## 2. Chunking Strategy: Template vs Semantic

**Template-based chunking** applies deterministic rules - split on headings, page breaks, or regex patterns - producing chunks whose boundaries align with the document's explicit structure. **Semantic segmentation** uses embedding similarity between adjacent sentences to detect topic shifts, producing chunks whose boundaries align with meaning transitions regardless of formatting.

**Failure on highly structured documents (financial reports).** Semantic segmentation struggles here. Financial reports have dense, repetitive language across sections (e.g., every quarter's revenue discussion uses similar vocabulary). Embedding-based boundary detection may fail to split between sections that are semantically similar but structurally distinct. Template-based chunking, keyed to heading hierarchy and table boundaries, preserves the logical organization that users expect to query against.

**Failure on loosely structured corpora (chat logs).** Template-based chunking fails because chat logs lack consistent structural markers. Messages are short, interleaved across topics, and formatting is informal. Semantic segmentation can detect topic drift across message sequences and group related exchanges into coherent chunks, even without explicit section headers.

The practical conclusion is that chunking strategy must be configured per-corpus type, which is exactly why RAGFlow makes it configurable rather than prescribing a single method. A production system should maintain a registry of chunking strategies mapped to document classifiers.

---

## 3. Hybrid Retrieval Architecture

Hybrid retrieval combines lexical matching (BM25) and dense vector similarity, typically followed by a re-ranker, to cover failure modes that neither approach handles alone.

**Lexical-only failure.** BM25 relies on exact term overlap. A query for "car accident injuries" will miss a chunk discussing "vehicular collision trauma" because the vocabulary is disjoint. This is the vocabulary mismatch problem - synonymy and paraphrase defeat term-frequency methods. BM25 also cannot capture semantic relationships: "Python is slower than C++" will not match a query about "programming language performance comparison."

**Vector-only failure.** Dense retrieval captures semantic similarity but loses lexical precision. A query for "HIPAA Section 164.512(e)" - a specific regulatory citation - may retrieve chunks about healthcare privacy in general rather than the exact subsection, because the embedding model compresses the specific identifier into a neighborhood of semantically related but legally distinct content. Rare terms, proper nouns, and identifiers are systematically underweighted in embedding spaces.

**Hybrid edge case.** Hybrid retrieval can still fail when the relevant chunk is semantically distant from the query and contains none of the query terms. Consider a query "Why did the project fail?" where the answer is buried in a meeting transcript discussing "timeline adjustments" and "resource reallocation" - neither lexically nor semantically close to "failure." This requires either query decomposition or multi-hop reasoning, not better retrieval fusion.

Formally, hybrid retrieval improves recall because the union of BM25's term-match candidates and the vector model's semantic candidates is a superset of either alone. Precision is recovered by the re-ranker, which scores the merged candidate set using a cross-encoder that attends jointly to query and passage.

---

## 4. Multi-Stage Retrieval Pipeline

A multi-stage pipeline (candidate generation → re-ranking → query refinement) decomposes the retrieval problem into phases with different precision/recall operating points.

**Recall vs latency trade-off.** The first stage (ANN search or BM25) operates over the full corpus and must be fast, so it uses approximate methods that maximize recall at the cost of precision. A single-pass ANN search with k=10 may miss relevant documents that sit just outside the approximate nearest-neighbor boundary. The re-ranking stage applies a more expensive cross-encoder to a small candidate set (e.g., top 100), recovering precision without paying the cost of running the cross-encoder over millions of documents.

**Cascading error propagation.** The critical risk is that if the first stage fails to retrieve a relevant document, no subsequent stage can recover it. This is the recall ceiling problem: re-ranking cannot promote a document that was never retrieved. Mitigation strategies include over-retrieving at the first stage (high k), using multiple retrieval signals (hybrid), and iterative query refinement - where the system rewrites the query based on initial results and re-retrieves.

The multi-stage design mirrors classical information retrieval cascades (e.g., Matveeva et al., 2006) and reflects a fundamental systems principle: use cheap, high-recall filters early and expensive, high-precision scorers late.

---

## 5. Indexing Strategy and Storage Backends

The choice of storage backend should be driven by the dominant query pattern, consistency requirements, and operational complexity budget.

**Elasticsearch-like hybrid store.** Supports both inverted indexes (BM25) and dense vector search (kNN plugin). Favored for workloads that require full-text search combined with metadata filtering and faceting - e.g., enterprise knowledge bases where users query by keyword, date range, and department simultaneously. Weakness: vector search performance degrades at scale compared to purpose-built vector databases, and operational overhead is significant.

**Vector-native database (e.g., Milvus, Qdrant, Weaviate).** Optimized for high-throughput ANN queries with HNSW or IVF indexes. Favored for workloads dominated by semantic similarity search over large embedding collections - e.g., image retrieval, recommendation systems, or RAG systems where lexical search is not needed. Weakness: limited support for complex filtering, joins, or full-text queries.

**Graph-augmented store (e.g., Neo4j with vector extensions).** Favored for workloads requiring multi-hop traversal or relationship-aware retrieval - e.g., "find all subsidiaries of companies mentioned in this contract" or biomedical knowledge graphs connecting genes, proteins, and diseases. Graph stores excel at compositional queries that vector similarity cannot express. Weakness: graph construction is expensive, query languages (Cypher, SPARQL) add complexity, and scaling graph traversal is non-trivial.

Design criteria: query latency requirements, dominant query type (keyword/semantic/relational), corpus update frequency, and operational team expertise.

---

## 6. Query Understanding and Reformulation

The semantic gap between a user's natural language query and the retrieval index is often the primary bottleneck in RAG systems - not the retrieval algorithm itself.

**Static query to retrieval.** The user's query is embedded or tokenized and sent directly to the index. This fails when the query is ambiguous ("Tell me about the merger"), underspecified ("What happened last quarter?"), or uses vocabulary mismatched with the indexed content. There is no mechanism to recover from a poorly formulated query.

**Iterative query refinement (agent-driven).** An LLM agent inspects initial retrieval results, identifies gaps, and reformulates the query - decomposing complex questions into sub-queries, expanding terms, or constraining scope. For example, "Compare our 2024 and 2025 revenue" can be decomposed into two targeted retrievals. This is analogous to how a human researcher iteratively refines search terms based on initial results.

Query transformation is critical because it shifts the burden of bridging the semantic gap from the user to the system. In production, most users cannot be expected to formulate retrieval-optimal queries. Techniques include HyDE (hypothetical document embeddings, where the LLM generates a hypothetical answer and uses its embedding as the query), step-back prompting (abstracting the query to a higher level before retrieval), and multi-turn decomposition.

---

## 7. Knowledge Representation Layer

**Dense vector space.** Documents and queries are mapped to a shared embedding space where proximity encodes semantic similarity. Strengths: captures fuzzy semantic relationships, scales to large corpora, and requires no manual schema. Weakness: vectors are opaque - you cannot inspect why two items are similar, making debugging and explainability difficult. Compositional reasoning (A relates to B, B relates to C, therefore A relates to C) is poorly supported because transitive relationships are not preserved in cosine similarity.

**Relational schema.** Structured data in normalized tables with foreign keys. Strengths: precise queries via SQL, strong consistency guarantees, and full explainability (you can trace exactly which join produced a result). Weakness: requires upfront schema design, brittle to schema changes, and cannot represent fuzzy or probabilistic relationships. Compositional reasoning is supported through joins but only over pre-defined relationships.

**Knowledge graph.** Entities and relationships represented as typed edges in a graph. Strengths: naturally supports compositional and multi-hop reasoning ("Who founded the company that acquired X?"), and provides full explainability through traceable paths. Weakness: construction is expensive (entity extraction, relation extraction, coreference resolution), coverage is often incomplete, and maintaining consistency as new documents are ingested is an open problem.

The optimal representation depends on the reasoning task. For single-hop factual retrieval, vectors suffice. For structured analytical queries, relational schemas are superior. For complex reasoning over interconnected entities, knowledge graphs are necessary. Production systems increasingly combine all three.

---

## 8. Data Ingestion Pipeline Architecture

A robust ingestion pipeline must handle heterogeneous sources (PDFs, web pages, databases, APIs) and produce consistently indexed knowledge.

**Schema normalization.** Each source produces different raw formats. The pipeline should define a canonical intermediate representation - e.g., a document object with fields for text, metadata, source URI, timestamp, and structural annotations. Source-specific adapters convert raw input to this canonical form. This decouples parsing logic from indexing logic and allows new sources to be added without modifying downstream components.

**Incremental indexing.** Re-indexing the entire corpus on every update is prohibitive at scale. The pipeline should track document versions (via content hashes or modification timestamps) and only re-process changed documents. This requires a metadata store that maps source documents to their indexed chunks, enabling targeted deletion and re-insertion. Change data capture (CDC) patterns from database replication apply directly here.

**Consistency vs throughput trade-offs.** Synchronous indexing (write-through) guarantees that queries immediately reflect new documents but limits ingestion throughput. Asynchronous indexing (write-behind via a message queue) maximizes throughput but introduces a visibility lag - newly ingested documents are not immediately queryable. The appropriate choice depends on the application: a legal discovery system may require strong consistency, while a general knowledge base can tolerate eventual consistency.

A well-designed pipeline includes: source adapters → schema normalization → chunking → embedding generation → index writing, with dead-letter queues for failed documents, idempotent processing for retry safety, and monitoring for ingestion lag.

---

## 9. Memory Design in RAG Systems

Memory in RAG systems enables context persistence across interactions, moving beyond stateless query-response patterns.

**Vector memory (semantic recall).** Past interactions are embedded and stored in a vector index. At query time, semantically similar past exchanges are retrieved and injected into the prompt. Strengths: captures thematic continuity ("we were discussing the marketing budget") without requiring structured schemas. Weakness: no temporal ordering - the system cannot distinguish what was said five minutes ago from five days ago, and similar but contradictory statements (e.g., a revised budget figure) may both be retrieved.

**Structured memory (SQL/graph).** Key facts extracted from interactions are stored in a schema - user preferences, decisions, entity states. Strengths: precise recall of specific facts, supports updates (the latest budget figure overwrites the old one), and enables relational queries. Weakness: requires extraction logic to identify what constitutes a "fact" worth storing, which is error-prone and domain-dependent.

**Episodic logs (temporal traces).** Full interaction history stored chronologically, typically summarized at session boundaries. Strengths: preserves temporal ordering and conversational arc, enabling the system to reference "what we discussed earlier" with fidelity. Weakness: grows linearly with interaction count, and long histories must be compressed or summarized, introducing information loss.

Production memory systems should layer these approaches: episodic logs for short-term context within a session, structured memory for persistent facts across sessions, and vector memory for thematic retrieval across a large interaction history.

---

## 10. End-to-End System Decomposition

A microservices architecture for RAGFlow should decompose along functional boundaries that align with independent scaling and failure isolation needs.

**Service decomposition:**

- **Ingestion Service** (stateless): Accepts documents, applies parsing and chunking, produces canonical document objects. Scales horizontally with document volume. Isolated failures (a malformed PDF) do not affect other services.
- **Embedding Service** (stateless): Generates vector embeddings for chunks. GPU-bound, scales independently based on embedding throughput requirements. Can be backed by a model serving framework (e.g., Triton, vLLM).
- **Index Service** (stateful): Manages the retrieval index (Elasticsearch, vector DB). Stateful by nature - data persistence and replication are critical. Scaling strategy depends on the backend: Elasticsearch scales via sharding, vector DBs via partitioning.
- **Retrieval Service** (stateless): Accepts queries, performs hybrid retrieval against the Index Service, applies re-ranking. Stateless because all state lives in the index. Scales horizontally with query volume.
- **Reasoning Service** (stateless): Orchestrates LLM calls, manages agent loops, applies query reformulation. Stateless per-request but may hold ephemeral session state for multi-turn interactions. Scales with LLM API throughput.
- **Memory Service** (stateful): Manages user interaction history and extracted knowledge. Stateful - requires durable storage. Scales based on user count and interaction depth.
- **API Gateway / Serving Layer** (stateless): Handles authentication, rate limiting, request routing. Standard stateless gateway pattern.

**Failure isolation boundaries.** The most critical boundary is between Ingestion and Retrieval: ingestion failures (parsing errors, embedding model downtime) must not degrade query serving. This is achieved by making ingestion asynchronous - documents are queued, and the retrieval index serves from its last consistent state. Similarly, Reasoning Service failures (LLM API timeouts) should return graceful degradation (e.g., raw retrieved chunks) rather than cascading into retrieval failures.

**Scaling strategy.** Ingestion and Embedding scale with document volume (batch-oriented, can use spot instances). Retrieval and Reasoning scale with query volume (latency-sensitive, require reserved compute). Index and Memory scale with data volume (storage-bound, require careful capacity planning).

```mermaid
graph TD
    A[API Gateway] --> B[Retrieval Service]
    A --> C[Reasoning Service]
    A --> F[Ingestion Service]
    F --> G[Embedding Service]
    G --> D[Index Service]
    F --> D
    B --> D
    C --> B
    C --> E[Memory Service]
    C --> H[LLM API]
```
