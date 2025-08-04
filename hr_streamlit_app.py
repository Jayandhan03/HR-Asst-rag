import streamlit as st
from datetime import datetime
import traceback

st.set_page_config(page_title="HR Policy Assistant", layout="wide")

# === 🧠 Title & Description ===
st.title("🤖 HR Policy Assistant")
st.markdown("Ask any HR policy-related question below:")

# === 🚨 Import with error catch ===
try:
    from interface.query_handler import handle_query
    from render_to_docx import render_to_docx
    backend_ready = True
except Exception as e:
    st.error("❌ Failed to load backend modules.")
    st.code(traceback.format_exc())
    backend_ready = False

# === 🧠 Initialize session ===
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# === 👤 User Input ===
query = st.text_input("👤 Your Question:", key="input_text")

if st.button("Ask") and backend_ready:
    if not query:
        st.warning("⚠️ Please enter a question before submitting.")
    else:
        try:
            with st.spinner("Thinking... 🤔"):
                response = handle_query(query)
                st.session_state.conversation.append((query, response))
                st.success("✅ Answer generated!")

        except Exception as e:
            st.error("❌ Failed to generate response.")
            st.code(traceback.format_exc())

# === 💬 Show Latest Response ===
if st.session_state.conversation:
    latest_query, latest_response = st.session_state.conversation[-1]
    st.text_area("🤖 HR Assistant Response:", value=latest_response, height=200)

# === 📜 Full Chat Log ===
if st.session_state.conversation:
    st.markdown("### 🧾 Conversation Log")
    for idx, (q, a) in enumerate(st.session_state.conversation):
        st.markdown(f"**👤 You:** {q}")
        st.markdown(f"**🤖 Assistant:** {a}")

    if st.button("📄 Save Conversation as DOCX"):
        full_convo = "\n\n".join(
            f"👤 You: {q}\n🤖 HR Assistant: {a}" for q, a in st.session_state.conversation
        )
        filename = f"HR_Conversation_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.docx"
        try:
            render_to_docx(query="HR Conversation", response=full_convo, filename=filename)
            with open(filename, "rb") as f:
                st.download_button("⬇️ Download DOCX", f, file_name=filename, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        except Exception as e:
            st.error("❌ Error saving DOCX file.")
            st.code(traceback.format_exc())

# === 🛠 Footer ===
st.markdown("---")
st.caption("Built with 💼 LangChain + Streamlit + RAG 🧠")
