import streamlit as st
import requests

# --- Configuration ---
FASTAPI_BASE_URL = "http://127.0.0.1:8000/api/v1"

st.set_page_config(
    page_title="Knowledge Copilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Main UI ---
st.title("🧠 Knowledge Copilot")
st.markdown("Your locally-hosted, privacy-first Hybrid RAG AI.")

# Create clean navigation tabs
tab_chat, tab_manage = st.tabs(["💬 Chat Interface", "🗂️ Manage Knowledge Base"])

# ==========================================
# TAB 1: CHAT INTERFACE
# ==========================================
with tab_chat:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # If it's an AI message with citations, show the metadata cleanly
            if msg["role"] == "assistant" and "citations" in msg:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric(label="AI Confidence", value=f"{msg['confidence'] * 100:.1f}%")
                with col2:
                    if msg["citations"]:
                        with st.expander("📚 View Verified Sources"):
                            for cite in msg["citations"]:
                                st.caption(f"**File:** {cite['filename']} (Page {cite['page']}) | **Match Score:** {cite['cross_encoder_score']:.2f}")

    # Chat Input Box
    if prompt := st.chat_input("Ask a question about your documents..."):
        # 1. Add user message to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Call FastAPI Backend
        with st.chat_message("assistant"):
            with st.spinner("Searching vectors and analyzing text..."):
                try:
                    response = requests.post(
                        f"{FASTAPI_BASE_URL}/chat/ask",
                        json={"query": prompt}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data.get("answer", "No answer generated.")
                        confidence = data.get("confidence_score", 0.0)
                        citations = data.get("citations", [])
                        
                        # Display Answer
                        st.markdown(answer)
                        
                        # Display Metadata
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric(label="AI Confidence", value=f"{confidence * 100:.1f}%")
                        with col2:
                            if citations:
                                with st.expander("📚 View Verified Sources"):
                                    for cite in citations:
                                        st.caption(f"**File:** {cite['filename']} (Page {cite['page']}) | **Match Score:** {cite['cross_encoder_score']:.2f}")
                        
                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "confidence": confidence,
                            "citations": citations
                        })
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("🚨 Cannot connect to Backend! Is your FastAPI server running on port 8000?")

# ==========================================
# TAB 2: MANAGE KNOWLEDGE BASE
# ==========================================
with tab_manage:
    st.header("Upload & Manage Documents")
    
    col_upload, col_list = st.columns([1, 1], gap="large")
    
    # --- Upload Section ---
    with col_upload:
        st.subheader("1. Add New Knowledge")
        uploaded_file = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"])
        
        if st.button("🚀 Process & Ingest Document", use_container_width=True, type="primary"):
            if uploaded_file:
                with st.spinner("Chunking text, generating embeddings, and updating BM25..."):
                    try:
                        # Send the physical file to FastAPI
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        res = requests.post(f"{FASTAPI_BASE_URL}/documents/upload", files=files)
                        
                        if res.status_code == 200:
                            st.success(f"✅ Successfully ingested: {uploaded_file.name}")
                        else:
                            st.error(f"Failed to upload: {res.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("🚨 Cannot connect to Backend!")
            else:
                st.warning("Please select a file first.")

    # --- List & Delete Section ---
    with col_list:
        st.subheader("2. Current Knowledge Base")
        
        # Add a refresh button just in case
        if st.button("🔄 Refresh List"):
            pass 
            
        try:
            res = requests.get(f"{FASTAPI_BASE_URL}/documents/")
            if res.status_code == 200:
                documents = res.json()
                
                if not documents:
                    st.info("Your knowledge base is empty. Upload a file to get started!")
                else:
                    # Display each document in a clean visual card
                    for doc in documents:
                        with st.container():
                            doc_col, btn_col = st.columns([4, 1])
                            with doc_col:
                                st.markdown(f"📄 **{doc['filename']}**")
                                st.caption(f"ID: {doc['document_uuid'][:8]}... | Uploaded: {doc['upload_time'][:10]}")
                            
                            with btn_col:
                                # Unique key is required for Streamlit buttons in loops
                                if st.button("🗑️ Delete", key=f"del_{doc['document_uuid']}"):
                                    with st.spinner("Deleting vectors & rebuilding BM25..."):
                                        del_res = requests.delete(f"{FASTAPI_BASE_URL}/documents/{doc['document_uuid']}")
                                        if del_res.status_code == 200:
                                            st.success("Deleted!")
                                            st.rerun() # Refresh the page instantly
                                        else:
                                            st.error("Failed to delete.")
                            st.divider()
            else:
                st.error("Failed to fetch documents.")
        except requests.exceptions.ConnectionError:
            st.error("🚨 Cannot connect to Backend!")