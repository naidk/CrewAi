# streamlit_app.py
from __future__ import annotations
import os
import json
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from crewai import Crew, Process

# Your project modules
from agents import blog_researcher, blog_writer
from tasks import research_task, write_task

# --- Env/Secrets ---
# Priority: Streamlit secrets -> .env -> OS env
def _load_env():
    # From Streamlit secrets
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    if "OPENAI_MODEL" in st.secrets:
        os.environ["OPENAI_MODEL"] = st.secrets["OPENAI_MODEL"]
    if "YOUTUBE_CHANNELS" in st.secrets:
        os.environ["YOUTUBE_CHANNELS"] = st.secrets["YOUTUBE_CHANNELS"]
    if "YOUTUBE_MAX_RESULTS" in st.secrets:
        os.environ["YOUTUBE_MAX_RESULTS"] = str(st.secrets["YOUTUBE_MAX_RESULTS"])
    # From .env (local dev)
    load_dotenv(override=False)

_load_env()

# --- Page config ---
st.set_page_config(page_title="CrewAI YouTube → Blog", page_icon="🧩", layout="wide")

# --- Sidebar: Config ---
with st.sidebar:
    st.title("⚙️ Settings")
    topic = st.text_input("Topic", value="AI vs ML vs DL vs Data Science")
    process_mode = st.selectbox("Process", ["sequential", "hierarchical"], index=0)
    memory_on = st.toggle("Enable Crew memory", value=True)
    cache_on = st.toggle("Enable Crew cache", value=True)
    max_rpm = st.slider("Max RPM", min_value=10, max_value=200, value=60, step=10)
    output_dir = st.text_input("Output directory", value="outputs")
    output_name = st.text_input("Output filename (.md)", value="new-blog-post.md")
    show_debug = st.toggle("Show research debug (JSON)", value=False)

    st.markdown("---")
    st.caption("Secrets used:")
    st.code(
        "OPENAI_MODEL="
        + os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        + "\nYOUTUBE_CHANNELS="
        + os.getenv("YOUTUBE_CHANNELS", "@krishnaik06"),
        language="bash",
    )

# --- Helpers ---
@st.cache_resource(show_spinner=False)
def build_crew(process: str, memory: bool, cache: bool, rpm: int) -> Crew:
    return Crew(
        agents=[blog_researcher, blog_writer],
        tasks=[research_task, write_task],
        process=Process.hierarchical if process == "hierarchical" else Process.sequential,
        memory=memory,
        cache=cache,
        max_rpm=rpm,
        share_crew=True,
    )

def save_outputs(md_text: str, meta: dict, out_dir: str, out_name: str) -> Path:
    pdir = Path(out_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    md_path = pdir / out_name
    meta_path = pdir / (Path(out_name).stem + "_meta.json")
    md_path.write_text(md_text, encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return md_path

# --- Main UI ---
st.title("🧩 CrewAI: YouTube Research → Blog Generator")
st.write(
    "This app uses your CrewAI agents & tasks to research YouTube videos and write a ready-to-publish blog post."
)

col_run, col_status = st.columns([1, 3])
with col_run:
    run = st.button("🚀 Run Crew", type="primary")

status_box = col_status.empty()

# --- Execution ---
if run:
    if not topic.strip():
        st.error("Please enter a topic.")
        st.stop()

    # Sanity for OpenAI key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is missing. Add it in Streamlit → Settings → Secrets.")
        st.stop()

    crew = build_crew(process_mode, memory_on, cache_on, max_rpm)

    status_box.info("⏳ Running crew… this can take a minute depending on transcripts and model speed.")

    t0 = time.time()
    try:
        with st.spinner("Thinking hard, watching videos, and writing…"):
            result = crew.kickoff(inputs={"topic": topic})
    except Exception as e:
        st.exception(e)
        st.stop()

    elapsed = time.time() - t0
    status_box.success(f"✅ Done in {elapsed:.1f}s")

    # Normalize result to string
    md = result if isinstance(result, str) else str(result)
    meta = {
        "topic": topic,
        "elapsed_sec": elapsed,
        "process": process_mode,
        "memory": memory_on,
        "cache": cache_on,
        "max_rpm": max_rpm,
    }

    # Save & offer download
    md_path = save_outputs(md, meta, output_dir, output_name)
    st.download_button(
        label="⬇️ Download Markdown",
        data=md.encode("utf-8"),
        file_name=output_name,
        mime="text/markdown",
        use_container_width=True,
    )

    # Preview
    st.markdown("## 📝 Blog Preview")
    st.markdown(md, unsafe_allow_html=False)

    # Debug (optional): show the research task output if your framework exposes it
    if show_debug:
        st.markdown("---")
        st.markdown("### 🔍 Research Debug (if available)")
        # Depending on CrewAI version, you might be able to access task outputs.
        # If not directly available, you can add custom returns in your tasks.
        st.info(
            "Tip: Return structured JSON from research_task and attach as context. "
            "Then render it here for transparency."
        )
