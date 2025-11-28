import os
import re
import time
import streamlit as st
from multi_agents import *
from langgraph.graph import StateGraph, END
from PIL import Image

def load_image(image_file):
    return Image.open(image_file)

def main():
    st.set_page_config(page_title="Resume Screening Assistant", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        .block-container{padding-top:2rem;padding-bottom:2rem}
        body{background:radial-gradient(1200px 600px at 10% 10%,#0f172a 0,#0b1220 40%,#05060a 100%)}
        .mesmerize-card{background:rgba(255,255,255,0.06);backdrop-filter:blur(10px);border-radius:18px;border:1px solid rgba(255,255,255,0.08);box-shadow:0 20px 40px rgba(0,0,0,.35);padding:24px}
        .stButton>button{background:linear-gradient(135deg,#7c3aed,#06b6d4);color:#fff;border:none;border-radius:12px;padding:12px 18px;font-weight:600}
        .stButton>button:hover{filter:brightness(1.08)}
        </style>
        """
        ,
        unsafe_allow_html=True,
    )

    def valid_key():
        k = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
        return bool(k) and k.startswith("gsk_")
    api_ok = valid_key()

    with st.sidebar:
        st.markdown("<div class='mesmerize-card'>", unsafe_allow_html=True)
        st.title("Upload")
        pdf_file = st.file_uploader("Resume (PDF)", type=["pdf"])
        if pdf_file is not None:
            with open("Resume.pdf", "wb") as f:
                f.write(pdf_file.read())
        text_file = st.file_uploader("Job Description (TXT)", type=["txt"]) 
        jd_text = ""
        if text_file is not None:
            jd_text = text_file.read().decode("utf-8", errors="ignore")
        else:
            jd_text = st.text_area("Or paste JD", height=150)
        if jd_text.strip() != "":
            with open("JD.txt", "w", encoding="utf-8") as f:
                f.write(jd_text)
        if api_ok:
            st.success("Groq API key detected")
        else:
            st.error("Invalid or missing GROQ_API_KEY. Set in .env (starts with gsk_) and restart")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='mesmerize-card'>", unsafe_allow_html=True)
    st.title("Resume Screening and Matching Assistant")
    st.caption("Welcome!")
    tab_upload, tab_results = st.tabs(["Upload", "Results"])

    with tab_upload:
        st.write("Use the sidebar to upload files and configure the job description.")

    ready = pdf_file is not None and (text_file is not None or jd_text.strip() != "") and api_ok
    run_disabled = not ready

    trigger = st.button("Match Resume", disabled=run_disabled)
    if trigger:
        inputs = {"messages": ["You are a recruitment expert and your role is to match a candidate's profile with a given job description."]}

        workflow = StateGraph(AgentState)
        workflow.add_node("Resume_agent", agent)
        workflow.add_node("JD_agent", JD_agent)
        workflow.add_node("Redflag_agent", redflag_agent)
        workflow.add_node("Recruiter_agent", recruit_agent)
        workflow.set_entry_point("Resume_agent")
        workflow.add_edge("Resume_agent", "JD_agent")
        workflow.add_edge("Resume_agent", "Redflag_agent")
        workflow.add_edge("JD_agent", "Recruiter_agent")
        workflow.add_edge("Redflag_agent", "Recruiter_agent")
        workflow.add_edge("Recruiter_agent", END)
        app_graph = workflow.compile()

        pass

        steps = ["Resume_agent", "JD_agent", "Redflag_agent", "Recruiter_agent"]
        progress = st.progress(0.0)
        outputs = app_graph.stream(inputs)
        agent_outputs = {}
        with st.spinner("Running agents"):
            for output in outputs:
                for key, value in output.items():
                    messages = value.get("messages", [])
                    agent_outputs.setdefault(key, [])
                    for msg in messages:
                        agent_outputs[key].append(str(msg))
                    if key in steps:
                        idx = steps.index(key)
                        progress.progress((idx + 1) / len(steps))

        st.session_state["agent_outputs"] = agent_outputs

    

    def parse_recruiter(text: str):
        total = None
        skills = None
        experience = None
        education = None
        extras = None
        total_m = re.search(r"(?i)total\s*score.*?(\d{1,3})", text)
        if total_m:
            try:
                total = int(total_m.group(1))
            except:
                total = None
        skills_m = re.search(r"(?i)skills\s*:\s*(\d{1,3})\s*/\s*30", text)
        exp_m = re.search(r"(?i)experience\s*:\s*(\d{1,3})\s*/\s*50", text)
        edu_m = re.search(r"(?i)education\s*:\s*(\d{1,3})\s*/\s*10", text)
        extra_m = re.search(r"(?i)extras.*?:\s*(\d{1,3})\s*/\s*10", text)
        try:
            skills = int(skills_m.group(1)) if skills_m else None
            experience = int(exp_m.group(1)) if exp_m else None
            education = int(edu_m.group(1)) if edu_m else None
            extras = int(extra_m.group(1)) if extra_m else None
        except:
            pass
        return {"total": total, "skills": skills, "experience": experience, "education": education, "extras": extras}

    with tab_results:
        if "agent_outputs" in st.session_state:
            outputs = st.session_state["agent_outputs"]
            st.subheader("Agent Outputs")
            for name in ["Resume_agent", "JD_agent", "Redflag_agent"]:
                if name in outputs:
                    with st.expander(name.replace("_", " ").title(), expanded=True):
                        for msg in outputs[name]:
                            st.write(msg)

            rec_text = None
            if "Recruiter_agent" in outputs:
                rec_text = "\n\n".join(outputs["Recruiter_agent"]) 
                stats = parse_recruiter(rec_text)
                st.subheader("Score")
                cols = st.columns(5)
                cols[0].metric("Total", stats.get("total") if stats.get("total") is not None else "—")
                cols[1].metric("Skills", stats.get("skills") if stats.get("skills") is not None else "—")
                cols[2].metric("Experience", stats.get("experience") if stats.get("experience") is not None else "—")
                cols[3].metric("Education", stats.get("education") if stats.get("education") is not None else "—")
                cols[4].metric("Extras", stats.get("extras") if stats.get("extras") is not None else "—")
                st.markdown("### Verdict")
                st.success(rec_text)
            else:
                st.info("Run a match to see results")
        else:
            st.info("Run a match to see results")
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
