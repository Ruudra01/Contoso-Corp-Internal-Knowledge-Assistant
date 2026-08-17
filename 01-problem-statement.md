# Problem Statement — Contoso Corp Internal Knowledge Assistant

**Use Case 1 · Week 1 · RAG Knowledge Chatbot**
**Author:** Ruudra Patel · **Date:** 2026-08-17 · **Status:** Draft for Monday design review

---

## 1. Business framing

Contoso Corp (mid-size enterprise, ~5,000 employees) keeps its HR policies, benefits guides, IT security standards, expense rules, and onboarding procedures spread across a document library of PDFs, Word files, and intranet pages. The documents are authoritative but not findable: keyword search on the intranet returns the wrong document as often as the right one, and employees route around it by asking HR and IT Helpdesk directly.

The observable cost is ticket volume and answer latency. HR and Helpdesk absorb a steady stream of questions whose answers already exist in writing ("how many PTO days carry over?", "what's the reimbursement limit for a hotel in a Tier 1 city?", "when does dental coverage start for a new hire?"). Each one costs a support agent a few minutes and costs the employee hours-to-days of waiting. Worse, answers given verbally drift from the written policy, which creates compliance exposure on anything touching leave, pay, or data handling.

An internal knowledge assistant that answers in natural language **and cites the policy it is quoting** converts this from a routing problem into a self-service one, and keeps the written document as the single source of truth.

## 2. Users

| User | What they need | How success feels to them |
| --- | --- | --- |
| **Employee** (primary) | A fast, plain-language answer to a policy question, with a link to the exact section so they can trust it | "I got the answer in 15 seconds and I can see where it came from" |
| **HR / IT Helpdesk agent** (secondary) | Fewer repeat tier-1 questions; a fast lookup tool when handling the ones that remain | "The easy tickets stopped arriving" |
| **Knowledge admin / Policy owner** (secondary) | Confidence that the assistant is answering from the current document set, and a way to re-index after a policy update | "I published v3 of the travel policy and the assistant picked it up" |

## 3. Scope

**In scope for Week 1:** a working assistant over a 24-document Contoso Corp corpus (10 PDF, 8 DOCX, 2 HTML, 4 Markdown), covering ingestion, retrieval, grounded generation with clickable citations, honest refusal, prompt-injection guardrails, multi-turn chat UI, and an admin document/re-index view.

## 4. Success criteria

These are the numbers the demo is measured against. Baseline is the 25-question evaluation set (`eval/questions.json`), each question labelled with its expected source document ID.

| # | Criterion | Target | How measured |
| --- | --- | --- | --- |
| S1 | **Retrieval hit rate** — expected source document appears in the retrieved context | ≥ 90% (23/25) | Automated run of the eval set; recall@6 against labelled `source_doc_id` |
| S2 | **Answer groundedness** — every factual claim traceable to a cited chunk | ≥ 90% of answered questions | Manual review of 25 answers, scored pass/fail against cited passages |
| S3 | **Citation correctness** — cited section actually contains the claim | 100% of citations resolve to a real chunk; ≥ 90% substantively support the claim | Click-through check on all citations in the eval run |
| S4 | **Honest refusal** — out-of-corpus questions produce a refusal, not a guess | 5/5 on the out-of-corpus probe set | Dedicated probe questions (e.g. "what is Contoso's stock price?") |
| S5 | **Prompt-injection resistance** — instruction-override attempts do not change system behaviour | 3/3 injection probes contained | Direct-input and corpus-embedded injection tests |
| S6 | **Latency** — time to first streamed token | p50 ≤ 2.0 s, p95 ≤ 4.0 s | Instrumented in App Insights over the eval run |
| S7 | **Cost per query** | ≤ $0.03 at pilot scale | Token accounting per request, logged per message |

## 5. Out of scope (Week 1)

- **Per-document access control.** Every document in the corpus is treated as readable by every authenticated employee. Row/document-level ACL trimming is designed for (see Solution Design §9) but not implemented.
- **Write actions.** The assistant answers questions; it does not file tickets, submit PTO requests, or update records.
- **Live connectors** to SharePoint, Confluence, or Workday. Ingestion runs from a Blob Storage container that is populated manually.
- **Non-English content**, OCR of scanned/image-only PDFs, and audio/video sources.
- **Personalisation** by employee record (tenure, location, band). Answers are corpus-general; where a policy varies by region, the assistant surfaces the distinction rather than resolving it for the user.
- **Production hardening**: multi-region failover, DR runbooks, formal pen-test, SOC 2 evidence.

## 6. Key assumptions and risks

| Assumption / risk | Impact if wrong | Mitigation |
| --- | --- | --- |
| The corpus is authoritative and internally consistent | Contradictory chunks produce confident-but-wrong answers | Show all conflicting citations rather than picking one; log conflicts for the policy owner |
| 24 documents is representative enough to tune retrieval | Chunking/top-k tuned to a small corpus may not hold at 10k docs | Scalability section states which parameters are corpus-size-sensitive; re-tune before pilot |
| Employees will trust a cited answer | Low adoption; tickets continue | Citations are clickable to the source passage; refusal is explicit rather than hedged |
| Semantic ranker + hybrid search is sufficient without fine-tuning | Retrieval hit rate below S1 | Fallback plan: query rewriting + increased candidate `k` before reranking |
