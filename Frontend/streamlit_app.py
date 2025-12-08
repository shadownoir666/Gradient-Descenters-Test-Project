import streamlit as st
import requests
import json
import time
import pandas as pd
import altair as alt
import html
import base64

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/api/upload"

# --- Page Config ---
st.set_page_config(
    page_title="SonicTrace",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .segment-container {
        border-left: 4px solid #38bdf8;
        background-color: #1e293b;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-radius: 0 12px 12px 0;
        transition: transform 0.2s;
    }
    .segment-container:hover {
        background-color: #253347;
    }
    .speaker-id {
        font-weight: 700;
        color: #38bdf8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    .text-content {
        font-size: 1.1rem;
        color: #e2e8f0;
        line-height: 1.6;
    }
    .meta-tags {
        display: flex;
        gap: 10px;
        margin-top: 1rem;
        font-size: 0.8rem;
        color: #94a3b8;
        align-items: center;
    }
    .emotion-badge {
        background-color: rgba(56, 189, 248, 0.1);
        color: #38bdf8;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/audio-wave.png", width=64)
    st.title("SonicTrace")
    st.markdown("---")
    st.markdown("### Settings")
    st.info("Additional settings coming soon.")
    st.markdown("---")
    st.markdown("Advanced Audio Analysis\n- Speaker Diarization\n- Whisper Transcription\n- Emotion Recognition")

# --- Main Content ---
st.markdown('<div class="main-header">SonicTrace Analysis</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Audio (WAV, MP3, M4A, FLAC)", type=['wav', 'mp3', 'm4a', 'flac'])

if uploaded_file:
    # Two columns for player and status
    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        st.audio(uploaded_file)
    with col2:
        analyze_btn = st.button("� Run Deep Analysis", type="primary", use_container_width=True)

    if analyze_btn:
        with st.status("Processing Audio...", expanded=True) as status:
            st.write("📤 Uploading file...")
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            
            try:
                st.write("⚙️ Running AI Models (Diarization & ASR)...")
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                    data = response.json()
                    segments = data.get("segments", [])
                    st.session_state['segments'] = segments
                else:
                    status.update(label="Analysis Failed", state="error")
                    st.error(f"Error {response.status_code}: {response.text}")
                    st.stop()
            except Exception as e:
                status.update(label="Connection Failed", state="error")
                st.error(f"Could not connect to backend: {e}")
                st.stop()

# --- Results View ---
if 'segments' in st.session_state:
    segments = st.session_state['segments']
    
    if not segments:
        st.warning("No speech segments detected.")
    else:
        # Metrics Row
        m1, m2, m3 = st.columns(3)
        df_seg = pd.DataFrame(segments)
        
        with m1:
            st.metric("Total Segments", len(segments))
        with m2:
            num_speakers = df_seg['speaker'].nunique() if not df_seg.empty else 0
            st.metric("Speakers Detected", num_speakers)
        with m3:
            dominant_emotion = df_seg['emotion'].mode()[0] if not df_seg.empty else "N/A"
            st.metric("Dominant Emotion", dominant_emotion.title())

        st.markdown("---")

        # Tabs for layouts
        tab1, tab2, tab3 = st.tabs(["📝 Transcript Flow", "📊 Analytics Dashboard", "🧬 Clustering Internals"])

        with tab1:
            for seg in segments:
                start_fmt = time.strftime('%M:%S', time.gmtime(seg['start']))
                end_fmt = time.strftime('%M:%S', time.gmtime(seg['end']))
                confidence = int(seg.get('emotion_score', 0) * 100)
                
                # Escape content to prevent HTML/Markdown breaking
                speaker_safe = html.escape(seg.get('speaker', 'Unknown'))
                # Replace newlines with space to prevent breaking out of HTML block
                text_safe = html.escape(seg.get('text', '')).replace('\n', ' ')
                emotion_safe = html.escape(seg.get('emotion', 'unknown').title())
                
                # Construct HTML without indentation to be 100% safe
                html_content = (
                    f'<div class="segment-container">'
                    f'<div class="speaker-id">{speaker_safe}</div>'
                    f'<div class="text-content">{text_safe}</div>'
                    f'<div class="meta-tags">'
                    f'<span>⏱️ {start_fmt} - {end_fmt}</span>'
                    f'<span>•</span>'
                    f'<span class="emotion-badge">{emotion_safe} ({confidence}%)</span>'
                    f'</div></div>'
                )
                
                st.markdown(html_content, unsafe_allow_html=True)

        with tab2:
            if not df_seg.empty:
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.subheader("Emotion Distribution")
                    chart_emotion = alt.Chart(df_seg).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta("count()", stack=True),
                        color=alt.Color("emotion", legend=alt.Legend(title="Emotion")),
                        tooltip=["emotion", "count()"]
                    )
                    st.altair_chart(chart_emotion, use_container_width=True)

                with col_b:
                    st.subheader("Speaker Participation (Segments)")
                    chart_speaker = alt.Chart(df_seg).mark_bar().encode(
                        x=alt.X("speaker", sort="-y"),
                        y="count()",
                        color="speaker",
                        tooltip=["speaker", "count()"]
                    )
                    st.altair_chart(chart_speaker, use_container_width=True)
                
                st.subheader("Timeline Analysis")
                chart_timeline = alt.Chart(df_seg).mark_circle().encode(
                    x=alt.X('start', title='Time (seconds)'),
                    y=alt.Y('speaker', title='Speaker'),
                    size='emotion_score',
                    color='emotion',
                    tooltip=['text', 'emotion', 'start', 'end']
                ).interactive()
                st.altair_chart(chart_timeline, use_container_width=True)

        with tab3:
            st.markdown("### 🧬 Adaptive Clustering Decisions")
            st.info("These dendrograms show how the system decided on the number of speakers. The red line represents the threshold used to cut the tree.")
            
            dendrograms = data.get("dendrograms", [])
            if dendrograms:
                for d in dendrograms:
                    score_val = d.get('score', 0.0)
                    st.markdown(f"#### Candidate: {d['name'].title()} (Threshold: {d['threshold']:.3f}, Score: {score_val:.3f})")
                    if d.get("image"):
                        st.image(base64.b64decode(d['image']), use_container_width=True)
            else:
                st.warning("No clustering visualization data available.")
