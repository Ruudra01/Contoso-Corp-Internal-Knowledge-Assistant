# Contoso Corp Internal Knowledge Assistant Corpus

This is a fictional, internally consistent enterprise HR and operations corpus created for RAG evaluation. Contoso is used here as a fictional sample company; Microsoft also uses Contoso as a fictional company in its documentation. This corpus is independently authored and should not be treated as real Contoso policy.

## Contents

- 10 PDF documents
- 8 DOCX documents
- 2 HTML documents
- 4 Markdown documents
- 25 evaluation questions
- corpus_manifest.json

## RAG requirements

The assistant should answer only from retrieved corpus content. If evidence is absent, it should explicitly say that the information was not found in the knowledge base.

Several documents intentionally overlap. Examples include PTO entitlement across the handbook and PTO policy, remote work rules across the remote policy and flexible-work procedure, and benefits rules across the benefits guide, health policy, and FAQ.

The evaluation set includes direct retrieval, multi-document reasoning, conditional questions, procedural questions, and an unanswerable question.
