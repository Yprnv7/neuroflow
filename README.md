# NeuroFlow (v1 — Deprecated & Superseded)

> **Notice:** This repository contains the initial prototype of NeuroFlow. It has been superseded by **[NeuroFlow Hybrid RAG](https://github.com/Yprnv7/neuroflow-hybrid-rag)**.

---

## Migration to `neuroflow-hybrid-rag`

Active development, production upgrades, and empirical evaluations have moved to the flagship repository:

👉 **[https://github.com/Yprnv7/neuroflow-hybrid-rag](https://github.com/Yprnv7/neuroflow-hybrid-rag)**

### Key Improvements in the Upgraded Repository:
1. **Real-World Empirical Evaluation:** Full BEIR SciFact benchmark (5,183 passages, 150 test queries) replacing synthetic mocks, measuring **0.7392 NDCG@10** on Hybrid RRF fusion and **0.9767 Recall@100**.
2. **Dual-Stage Hybrid Retrieval:** Combines BM25Okapi lexical tokenization with dense `BAAI/bge-small-en-v1.5` embeddings and reciprocal rank fusion ($k=60$).
3. **Cross-Encoder Reranker:** Integrates `ms-marco-MiniLM-L-6-v2` with candidate truncation (top-25) to balance contextual attention against interactive serving latencies.
4. **Calibrated Semantic Cache & GDPR Erasure:** Cosine similarity threshold calibrated at $\tau = 0.92$ to prevent false-positive semantic collisions, with atomic `DELETE /v1/tenants/{tenant_id}/forget` compliance sweeps.

---

## Archival Status
This repository is preserved as a historical snapshot. For implementation details, benchmarks, and production usage, please refer to [neuroflow-hybrid-rag](https://github.com/Yprnv7/neuroflow-hybrid-rag).

---

## Author
**Pranav Yeleswarapu**  
- GitHub: [@Yprnv7](https://github.com/Yprnv7)  
- LinkedIn: [linkedin.com/in/pranavyeleswarapu](https://www.linkedin.com/in/pranavyeleswarapu)
