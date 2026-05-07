import os
import tempfile

from flask import Flask, render_template, request
from rag_core import (
    load_pdf,
    chunking,
    create_emb,
    create_index,
    search,
    generate_ans,
    extract_profile_answer,
    extract_capability_answer,
    extract_certificate_answer,
    is_capability_query,
    is_certificate_query,
    is_profile_query,
    is_resume_document,
    calculate_ats_score   # 
)

app = Flask(__name__)

index = None
chunks = []
vectorizer = None
chat_history = []
document_text = ""
uploaded_filename = ""


# FORMAT SOURCES

def format_sources(source_chunks, max_items=3, max_chars=140):
    snippets = []
    for chunk in source_chunks[:max_items]:
        text = " ".join(chunk.split())
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        snippets.append(text)
    return snippets



# MAIN ROUTE

@app.route("/", methods=["GET", "POST"])
def home():
    global index, chunks, vectorizer, chat_history, document_text, uploaded_filename

    answer = ""
    status_type = "info"
    sources = []
    ats_result = None 

    if request.method == "POST":

       
        # 1. PDF UPLOAD
   
        if "pdf" in request.files:
            file = request.files["pdf"]

            if file and file.filename:
                uploaded_filename = os.path.basename(file.filename)
                tmp_path = None

                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp_path = tmp.name

                    file.save(tmp_path)

                    try:
                        text = load_pdf(tmp_path)
                    except Exception:
                        text = ""

                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)

                if not text.strip():
                    answer = "The uploaded PDF is empty or has no extractable text."
                    status_type = "error"
                    index = None
                    chunks = []
                    vectorizer = None
                    document_text = ""
                    chat_history = []
                elif not is_resume_document(text):
                    answer = "This does not look like a resume. Please upload a resume PDF."
                    status_type = "error"
                    index = None
                    chunks = []
                    vectorizer = None
                    document_text = ""
                    chat_history = []
                else:
                    document_text = text

                    chunks = chunking(text, chunk_size=600, overlap=80)
                    embeddings, vectorizer = create_emb(chunks)
                    index = create_index(embeddings)

                    chat_history = []
                    answer = "PDF processed successfully."
                    status_type = "success"

        # 2. USER QUERY (CHAT)

        query = request.form.get("query", "").strip()

        if query:
            if index is None or not chunks:
                answer = "Please upload and process a PDF first."

            else:
                # Profile extraction
                profile_answer = extract_profile_answer(document_text, query)
                if profile_answer:
                    answer = profile_answer
                    sources = []

                # Capability extraction
                else:
                    capability_answer = extract_capability_answer(document_text, query)
                    certificate_answer = extract_certificate_answer(document_text, query)

                    if capability_answer:
                        answer = capability_answer
                        sources = []

                    elif certificate_answer:
                        answer = certificate_answer
                        sources = []

                    elif is_capability_query(query):
                        answer = "Not found in document"
                        sources = []

                    elif is_certificate_query(query):
                        answer = "Not found in document"
                        sources = []

                    elif is_profile_query(query):
                        answer = "Not found in document"
                        sources = []

                    else:
                        # 🔥 Improved retrieval (k=5)
                        results = search(query, index, chunks, vectorizer=vectorizer, k=5)

                        context = "\n\n".join(results)
                        answer = generate_ans(context, query)
                        sources = format_sources(results) if answer != "Not found in document" else []

                chat_history.append({"q": query, "a": answer})
                answer = ""

        # 3. ATS SCORING (NEW FEATURE)

        job_desc = request.form.get("job_desc", "").strip()

        if job_desc and document_text:
            score, matched, missing = calculate_ats_score(document_text, job_desc)

            ats_result = {
                "score": score,
                "matched": matched,
                "missing": missing
            }

    # RETURN RESPONSE

    return render_template(
        "index.html",
        answer=answer,
        status_type=status_type,
        uploaded_filename=uploaded_filename,
        chat=chat_history,
        sources=sources,
        ats=ats_result  
    )

# RUN APP

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
