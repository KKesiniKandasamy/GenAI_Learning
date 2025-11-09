# Integrated Multi-Agent Prototype

This prototype demonstrates:
- Conversational interface with limited memory (CLI)
- Document-based QA using RAG (TF-IDF retrieval + local answer fusion)
- Text-to-image generation (placeholder + hooks for real APIs)
- Multi-agent task handling via a central controller: Weather, SQL, Recommender
- Final technical report included

## How to run
1. Create a virtual environment and install requirements:
   ```
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Run the CLI:
   ```
   python main.py
   ```
3. Optional: set environment variables for real Weather or Image APIs:
   - `OPENWEATHER_API_KEY` for weather
   - `IMAGE_API_KEY` and modify `agents/image_agent.py` to call your preferred image API.

## Prototype behavior
- The controller interprets user intents via simple rule-based matching and routes to agents.
- Conversation memory stores the last N messages (configurable).
- RAG uses TF-IDF to find top documents in `docs/` and returns a synthesized answer.
- SQL agent uses `data/sample.db` (sqlite) with a sample `products` table for demo queries.
- Image agent creates a placeholder image (PIL) if no API configured.

## Files
- `main.py` - entrypoint CLI demonstrating integrated flows.
- `controller.py` - central router.
- `agents/` - weather, rag, image, sql, recommender agents.
- `docs/` - sample documents used by RAG.
- `report/final_report.md` - technical report (architecture, decisions, debugging).
- `prototype_package.zip` - this package (created for convenience).

## Notes
This is a lightweight, educational prototype. Replace API hooks with real service calls in production.
