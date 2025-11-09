# Final Technical Report

## 1. Architecture Overview
The prototype is organized around a central `Controller` which receives user input and routes it to specialized agents:
- `Memory` (limited conversational memory)
- `RAGAgent` (document retriever + simple synthesis)
- `WeatherAgent` (external API optional; mocked fallback)
- `ImageAgent` (text-to-image placeholder with hook for external APIs)
- `SQLAgent` (sqlite demo database)
- `RecommenderAgent` (content-based recommendations)

The system is intentionally modular: each agent has a clear interface and can be replaced by a more powerful implementation (e.g., replace TF-IDF with dense embeddings).

## 2. Key Implementation Decisions
- **Rule-based controller**: Simple, predictable routing for a prototype. Replace with NLU/LLM intent detection for production.
- **RAG via TF-IDF**: Chosen for low dependency and explainability. If `scikit-learn` is not available, a substring fallback is used.
- **Memory design**: Fixed-size sliding window stored on disk as JSON to survive runs.
- **Image generation**: Placeholder using PIL so the prototype is runnable offline. Hooks and comments point to where to add real API calls (OpenAI, Stability/AWS).
- **Weather agent**: Uses OpenWeatherMap if `OPENWEATHER_API_KEY` is set; otherwise returns deterministic mock to make demos reproducible.
- **SQL agent**: Uses sqlite for a self-contained demo DB with sample `products` table.

## 3. Debugging and Testing Process
- Unit testing was performed manually by running the CLI and exercising each agent:
  - RAG: verified that queries return document snippets from `docs/`.
  - Memory: confirmed last N turns are persisted in `state/memory.json`.
  - Weather: tested both mocked and API modes.
  - Image: generated placeholder images with different prompts.
  - SQL: ran SELECT queries and verified results.
- Common issues:
  - Missing dependencies (scikit-learn): handled with fallbacks.
  - Network errors for weather/image APIs: exceptions are caught and returned as user-friendly messages.

## 4. Challenges and Resolutions
- **Dependency availability**: Not all environments will have ML libraries installed. Resolved by implementing fallbacks and clear `requirements.txt`.
- **Intent detection complexity**: For a prototype, rule-based matching is used to avoid brittle LLM-based intent parsing. Documented how to upgrade to ML-based intent detection.
- **No live image-inference in offline mode**: Created a robust placeholder image generation and documented how to plug in external services.

## 5. Improvements & Future Work
- Replace TF-IDF with dense embeddings (e.g., SentenceTransformers) and a vector DB (FAISS, Milvus).
- Add an NLU/intent classifier (small fine-tuned model) or use an LLM for routing.
- Support multi-turn RAG synthesis via an LLM that ingests context + retrieved docs.
- Add automated unit tests (pytest) and CI.
- Provide a web UI (Flask/FastAPI + simple React front-end).

## 6. How to evaluate
1. Install requirements.
2. Run `python main.py` and try commands shown in the README.
3. Inspect `state/memory.json`, `outputs/` images, and `data/sample.db`.

-- End of Report
