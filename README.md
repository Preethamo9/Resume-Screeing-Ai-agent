

# Resume Screening Assistant

Streamlit app that uses a multi-agent pipeline to match a candidate’s resume with a job description. Powered by **LangGraph**, **LangChain**, and **Groq** LLMs.

---

## Features

- Upload a **resume (PDF)** and a **job description (TXT or plain text)**.
- Click **Match Resume** to run the agents.
- View **agent outputs** and a **final verdict** with score breakdown.
- Mesmerizing UI with gradient styling; simple sidebar greeting and inputs.


---

## Scoring

The **Recruiter Agent** evaluates and returns:
- Total score (0–100)
- Breakdown: Skills (30), Experience (50), Education (10), Extras (10)
- Short summary and recommendation

---

## Agents

Roles in the pipeline:

### Resume Agent
- Extracts candidate name and contact from PDF (`multi_agents.py:29–45`).

### JD Agent
- Extracts exact job requirements from `JD.txt` (`multi_agents.py:49–63`).

### Redflag Agent
- Detects gaps, hopping, irrelevant claims, missing education, issues (`multi_agents.py:67–101`).


### Recruiter Agent
- Scores and returns summary + recommendation (`multi_agents.py:106–168`).

---

## Tech

- Python, Streamlit
- LangGraph, LangChain
- Groq LLMs (via `langchain-groq`)
- PyPDFLoader, Pillow

---

## Visuals

Workflow diagram generation and display have been **removed** by design.


## Usage

1. Create `.env` with:
   - `GROQ_API_KEY=gsk_...` (required)
   - `GROQ_MODEL=llama-3.3-70b-versatile` (optional; default applied)
2. Install deps: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
4. Upload resume PDF and JD (TXT or paste), then click **Match Resume**.

Notes:
- Valid Groq key must start with `gsk_`.
- Default model follows Groq deprecation guidance; override via `GROQ_MODEL` if needed.

---

## Troubleshooting

- Invalid API key (401): Ensure `.env` contains `GROQ_API_KEY` starting with `gsk_` and restart.
- Model decommissioned (400): Set `GROQ_MODEL` to a supported model (e.g., `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`).
- No results: Confirm `Resume.pdf` and `JD.txt` exist in the project root.

---

Built for fast, transparent resume-to-JD matching.
