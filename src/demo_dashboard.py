"""Optional local viewer for the fictional-data demo."""
import json
from pathlib import Path
import sys
import tempfile
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo import run_demo

st.set_page_config(page_title="SocialCompass demo", page_icon="🧭", layout="wide")
st.title("SocialCompass — fictional team demo")
st.caption("Invented conversations and descriptive features. No personality or role predictions.")


@st.cache_data
def load_demo():
    with tempfile.TemporaryDirectory() as temp:
        summary = run_demo(temp)
        features = json.loads((Path(temp) / "features.json").read_text())
        messages = [json.loads(line) for line in
                    (Path(temp) / "synthetic_messages.jsonl").read_text().splitlines()]
    return summary, features, messages


summary, features, messages = load_demo()
team = st.selectbox("Choose a fictional team", sorted({r["team"] for r in features}))
rows = [r for r in features if r["team"] == team]
selected = [r for r in messages if r["team"] == team]
a, b = st.columns(2)
a.metric("Fictional members", len(rows))
b.metric("Messages", len(selected))
st.bar_chart({"member": [r["author"] for r in rows],
              "question_rate": [r["question_ratio"] for r in rows]},
             x="member", y="question_rate")
st.dataframe(rows, hide_index=True)
with st.expander("Read the invented conversation"):
    st.dataframe([{"member": r["author"], "message": r["body"]} for r in selected], hide_index=True)
st.info("Message features describe this chat only; they do not measure a person's value or work quality.")
