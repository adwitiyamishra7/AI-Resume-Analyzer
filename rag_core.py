import re
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import requests


# 1. LOAD PDF

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text


def is_resume_document(text):
    normalized = re.sub(r"\s+", " ", text.lower())
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    section_keywords = [
        "experience", "work experience", "employment", "education", "skills",
        "technical skills", "projects", "certifications", "summary", "objective",
    ]
    profile_patterns = [
        r"\S+@\S+\.\S+",
        r"\+?\d[\d\s\-]{8,}\d",
        r"linkedin\.com",
        r"github\.com",
    ]
    role_keywords = [
        "developer", "engineer", "analyst", "intern", "manager", "specialist",
        "consultant", "designer", "architect", "student", "candidate",
    ]

    section_hits = sum(1 for keyword in section_keywords if keyword in normalized)
    profile_hits = sum(1 for pattern in profile_patterns if re.search(pattern, normalized))
    role_hits = sum(1 for keyword in role_keywords if keyword in normalized)

    likely_name_at_top = False
    for line in lines[:5]:
        if re.search(r"[@\d:/]", line):
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(word.replace(".", "").isalpha() for word in words):
            likely_name_at_top = True
            break

    return (
        section_hits >= 3
        or (section_hits >= 2 and profile_hits >= 1)
        or (section_hits >= 2 and role_hits >= 1 and likely_name_at_top)
    )



# 2. CHUNKING

def chunking(text, chunk_size=600, overlap=80):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# 3. EMBEDDINGS (TF-IDF + MiniLM optional)

from sklearn.feature_extraction.text import TfidfVectorizer

_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def create_emb(chunks):
    # TF-IDF (lightweight, stable)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunks).toarray()

    # MiniLM (semantic)
    sem = _embed_model.encode(chunks, show_progress_bar=False)

    # concat (hybrid)
    embeddings = np.hstack([tfidf_matrix, sem])
    return embeddings.astype("float32"), vectorizer


# 4. FAISS INDEX

def create_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


# 5. SEARCH

def search(query, index, chunks, vectorizer=None, k=5):
    # TF-IDF part
    q_tfidf = vectorizer.transform([query]).toarray()

    # semantic part
    q_sem = _embed_model.encode([query], show_progress_bar=False)

    q = np.hstack([q_tfidf, q_sem]).astype("float32")

    distances, indices = index.search(q, k)
    return [chunks[i] for i in indices[0]]


# 6. LLM (OpenRouter)

OPENROUTER_API_KEY = "Bearer sk-or-v1-REPLACE_WITH_YOUR_KEY"

def generate_ans(context, query):
    prompt = f"""
You are an assistant. Answer ONLY from the given context.
If not present, say "Not found in document".

Context:
{context}

Question:
{query}

Answer:
"""
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": OPENROUTER_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "model": "google/gemma-3-4b-it:free",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        data = resp.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        return "Not found in document"
    except Exception:
        return "Not found in document"


# 7. PROFILE / CAPABILITY (simple rules)

def is_profile_query(q):
    q = q.lower()
    return any(x in q for x in [
        "name", "email", "phone", "location", "address",
        "linkedin", "linked in", "github", "git hub", "githu",
    ])

def is_capability_query(q):
    q = q.lower()
    return any(x in q for x in ["skill", "experience", "know", "expert", "technology", "tech stack"])

def is_certificate_query(q):
    q = q.lower()
    return any(x in q for x in ["certificate", "certificates", "certification", "certifications"])

def _clean_line(line):
    return " ".join(line.replace("|", " ").split()).strip(" -:\t")

def _normalize_url_spacing(text):
    return re.sub(r"https?\s*:\s*/\s*/", "https://", text)

def _looks_like_heading(line):
    if not line:
        return False
    raw = line.strip()
    lowered = raw.lower()
    known_headings = [
        "skills", "technical skills", "certifications", "certification",
        "certificates", "projects", "education", "experience",
        "extra curricular activities", "achievements", "summary"
    ]
    if lowered in known_headings:
        return True
    words = lowered.split()
    return 0 < len(words) <= 6 and raw == raw.upper() and not re.search(r"[@\d]", raw)

def _extract_section_lines(text, headings):
    lines = [_clean_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]

    for idx, line in enumerate(lines):
        lowered = line.lower()
        if any(h in lowered for h in headings):
            collected = []

            inline = re.split(r"[:\-]", line, maxsplit=1)
            if len(inline) == 2 and inline[1].strip():
                collected.append(inline[1].strip())

            for next_line in lines[idx + 1:]:
                if _looks_like_heading(next_line):
                    break
                collected.append(next_line)
            return [line for line in collected if line]
    return []

def _split_items(lines):
    items = []
    for line in lines:
        pieces = re.split(r"[,\u2022]|(?:\s{2,})", line)
        for piece in pieces:
            item = _clean_line(piece)
            if item and len(item) > 1:
                items.append(item)
    return items

def _dedupe_keep_order(items):
    seen = set()
    result = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result

def _format_list_answer(items):
    cleaned = _dedupe_keep_order([_clean_line(item) for item in items if _clean_line(item)])
    return ", ".join(cleaned)

def _format_multiline_answer(items):
    cleaned = _dedupe_keep_order([_normalize_url_spacing(_clean_line(item)) for item in items if _clean_line(item)])
    return "\n".join(cleaned)

def _extract_certificate_entries(lines):
    normalized_lines = [_normalize_url_spacing(_clean_line(line)) for line in lines if _clean_line(line)]
    if not normalized_lines:
        return []

    if len(normalized_lines) > 1:
        return _dedupe_keep_order(normalized_lines)

    blob = normalized_lines[0]
    month_pattern = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    matches = re.findall(
        rf"([A-Za-z][A-Za-z0-9&.+\-\s]*?https?://\S+\s+{month_pattern}\s+\d{{4}}(?:,\s*[^,]+(?:\s*[.?\u00b7-]\s*[^,]+)?)?)",
        blob
    )

    if matches:
        return _dedupe_keep_order([match.strip(" ,") for match in matches])

    return normalized_lines

def extract_name(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    for line in lines[:10]:
        match = re.match(r"^name\s*[:\-]\s*(.+)$", line, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    for line in lines[:10]:
        if re.search(r"[@\d]", line):
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(word.replace(".", "").isalpha() for word in words):
            return line

    return lines[0]

def extract_profile_answer(text, query):
    query_l = query.lower()
    normalized_text = _normalize_url_spacing(text)

    if "linkedin" in query_l or "linked in" in query_l:
        m = re.search(r"(?:https?://)?(?:www\.)?linkedin\.com/[^\s,;]+", normalized_text, flags=re.IGNORECASE)
        if m:
            link = m.group(0).rstrip(".,)")
            if not link.lower().startswith(("http://", "https://")):
                link = "https://" + link
            return link
        return ""

    if "github" in query_l or "git hub" in query_l or "githu" in query_l:
        m = re.search(r"(?:https?://)?(?:www\.)?github\.com/[^\s,;]+", normalized_text, flags=re.IGNORECASE)
        if m:
            link = m.group(0).rstrip(".,)")
            if not link.lower().startswith(("http://", "https://")):
                link = "https://" + link
            return link
        return ""

    if "email" in query_l:
        m = re.search(r"\S+@\S+\.\S+", text)
        return m.group(0) if m else ""
    if "phone" in query_l:
        m = re.search(r"\+?\d[\d\s\-]{8,}\d", text)
        return m.group(0) if m else ""
    if "name" in query_l:
        name = extract_name(text)
        return f"Name: {name}" if name else ""
    return ""

def extract_capability_answer(text, query):
    query_l = query.lower()

    if "skill" in query_l:
        skills = extract_skills(text)
        if skills:
            return ", ".join(skills)
        return ""

    # simple heuristic
    if "python" in query_l and "python" in text.lower():
        return "Yes, Python is mentioned in the resume."
    return ""

def extract_certificate_answer(text, query):
    if not is_certificate_query(query):
        return ""

    lines = _extract_section_lines(
        text,
        [
            "certifications", "certification", "certificates", "certificate",
            "licenses", "extra curricular activities", "extra-curricular activities",
            "achievements"
        ]
    )

    items = _extract_certificate_entries(lines)

    if not items:
        url_matches = re.findall(r"https?://\S+", text)
        certificate_links = [url.rstrip(".,)") for url in url_matches if "forage" in url.lower() or "coursera" in url.lower() or "udemy" in url.lower()]
        items = certificate_links

    return _format_multiline_answer(items)


# 8. ATS (HYBRID SKILL EXTRACTION)

STATIC_SKILLS = [
    # programming
    "python","java","c++","javascript",
    # web
    "html","css","react","node","flask","django",
    # ai/ml
    "machine learning","deep learning","nlp","tensorflow","pytorch",
    # cybersec
    "penetration testing","ethical hacking","network security",
    "cryptography","kali linux","wireshark","metasploit",
    # devops/cloud
    "docker","kubernetes","aws","azure","ci/cd"
]

def extract_skills_static(text):
    t = text.lower()
    return list({s for s in STATIC_SKILLS if s in t})

def extract_skills_section(text):
    lines = _extract_section_lines(text, ["technical skills", "skills", "tech stack"])
    items = _split_items(lines)
    cleaned = []
    for item in items:
        if re.search(r"[A-Za-z]", item) and len(item) <= 40:
            cleaned.append(item.lower())
    return _dedupe_keep_order(cleaned)

def extract_skills_llm(text):
    prompt = "Extract technical skills as comma-separated list."
    ans = generate_ans(text, prompt)
    parts = [p.strip().lower() for p in ans.split(",")]
    return [p for p in parts if len(p) > 2]

def extract_skills(text):
    section_skills = extract_skills_section(text)
    static_skills = sorted(extract_skills_static(text))

    skills = _dedupe_keep_order(section_skills + static_skills)
    if len(skills) >= 3:
        return skills

    llm_skills = extract_skills_llm(text)
    return _dedupe_keep_order(skills + llm_skills)

def calculate_ats_score(resume_text, job_text):
    rs = extract_skills(resume_text)
    js = extract_skills(job_text)

    matched = [s for s in js if s in rs]
    missing = [s for s in js if s not in rs]

    score = int((len(matched) / len(js)) * 100) if js else 0
    return score, matched, missing
