"""
Semantic resume extractor module.
Extracts candidate metadata, contact details, skills, projects, and work history.
Uses LLM adapter for semantic parsing when available, with a resilient rule-based fallback.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

from src.llm import BaseLLMProvider
from src.models import ExtractedProject, ExtractedResumeData
from src.utils import clean_text, extract_github_username, setup_logger

logger = setup_logger("extractor")

# Curated skill dictionary with standard casing
SKILL_TAXONOMY = {
    # Programming Languages
    "python": "Python",
    "python3": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "java": "Java",
    "c++": "C++",
    "c#": "C#",
    "golang": "Go",
    "rust": "Rust",
    "sql": "SQL",
    "html": "HTML",
    "css": "CSS",
    # AI / LLM / Agentic Frameworks & Concepts
    "langchain": "LangChain",
    "langgraph": "LangGraph",
    "llamaindex": "LlamaIndex",
    "llama_index": "LlamaIndex",
    "llama index": "LlamaIndex",
    "crewai": "CrewAI",
    "autogen": "AutoGen",
    "google adk": "Google ADK",
    "agent development kit": "Google ADK",
    "rag": "RAG",
    "retrieval augmented generation": "RAG",
    "retrieval-augmented generation": "RAG",
    "vector search": "Vector Search",
    "vector database": "Vector Database",
    "vector db": "Vector DB",
    "embeddings": "Embeddings",
    "embedding": "Embeddings",
    "faiss": "FAISS",
    "chroma": "ChromaDB",
    "chromadb": "ChromaDB",
    "pinecone": "Pinecone",
    "qdrant": "Qdrant",
    "weaviate": "Weaviate",
    "milvus": "Milvus",
    "ragas": "Ragas",
    "trulens": "TruLens",
    "tool calling": "Tool Calling",
    "tool-calling": "Tool Calling",
    "function calling": "Function Calling",
    "agentic ai": "Agentic AI",
    "ai agents": "AI Agents",
    "multi-agent": "Multi-Agent Workflows",
    "multi agent": "Multi-Agent Workflows",
    "prompt engineering": "Prompt Engineering",
    "hugging face": "Hugging Face",
    "huggingface": "Hugging Face",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "scikit-learn": "Scikit-Learn",
    "ollama": "Ollama",
    "vllm": "vLLM",
    # Backend & Architecture
    "fastapi": "FastAPI",
    "flask": "Flask",
    "django": "Django",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "express": "Express.js",
    "express.js": "Express.js",
    "rest api": "REST APIs",
    "restful": "REST APIs",
    "rest apis": "REST APIs",
    "graphql": "GraphQL",
    "grpc": "gRPC",
    "celery": "Celery",
    "redis": "Redis",
    "rabbitmq": "RabbitMQ",
    "kafka": "Kafka",
    "microservices": "Microservices",
    "system design": "System Design",
    # Databases
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "sqlite": "SQLite",
    "supabase": "Supabase",
    "firebase": "Firebase",
    # Cloud & DevOps
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "aws": "AWS",
    "gcp": "GCP",
    "google cloud": "GCP",
    "azure": "Azure",
    "ci/cd": "CI/CD",
    "github actions": "GitHub Actions",
    "linux": "Linux",
    "nginx": "Nginx",
    # Frontend & Fullstack
    "react": "React",
    "react.js": "React",
    "next.js": "Next.js",
    "nextjs": "Next.js",
    "tailwind": "Tailwind CSS",
    "redux": "Redux",
    "zustand": "Zustand",
    # Engineering Depth & Testing
    "pytest": "Pytest",
    "unit testing": "Unit Testing",
    "unittest": "Unittest",
}


def extract_candidate_name(text: str, file_path: str) -> str:
    """
    Extracts the candidate's name using multi-pass heuristic filtering.
    1. Evaluates candidate lines near the top of the resume.
    2. Falls back to scanning blocks adjacent to contact details (email/phone).
    3. Guarantees a deterministic, non-empty candidate name.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return Path(file_path).stem.replace("_", " ").title()

    blacklist_words = {
        "summary", "curriculum", "resume", "developer", "engineer", "phone",
        "email", "http", "https", "github", "linkedin", "skills", "projects",
        "education", "experience", "motivated", "detail-oriented", "student",
        "b.tech", "mca", "b.e", "m.tech", "certifications", "india", "bengaluru",
        "delhi", "hyderabad", "pune", "mumbai", "contact", "address", "profile"
    }

    # Pass 1: Check top 4 non-empty lines
    for line in lines[:4]:
        # Filter out common decorative characters, long text, or lines with separators
        if any(char in line for char in [":", "|", "/", "\\", "@", "+"]):
            continue
        words = line.split()
        if 1 <= len(words) <= 4 and len(line) >= 3:
            lower_line = line.lower()
            if not any(bw in lower_line for bw in blacklist_words):
                if line.isupper() or all(w[0].isupper() for w in words if w):
                    return line

    # Pass 2: Check lines adjacent to the email line (common in multi-column layouts)
    email_idx = -1
    for idx, line in enumerate(lines):
        if "@" in line:
            email_idx = idx
            break

    if email_idx != -1:
        for offset in [-1, 1, -2, 2]:
            candidate_idx = email_idx + offset
            if 0 <= candidate_idx < len(lines):
                line = lines[candidate_idx]
                if any(char in line for char in [":", "|", "/", "\\", "@", "+"]):
                    continue
                words = line.split()
                if 1 <= len(words) <= 4 and len(line) >= 3:
                    lower_line = line.lower()
                    if not any(bw in lower_line for bw in blacklist_words):
                        if line.isupper() or all(w[0].isupper() for w in words if w):
                            return line

    # Pass 3: Fallback to filename
    file_stem = Path(file_path).stem
    cleaned_stem = file_stem.replace("_", " ").title()
    return cleaned_stem


def extract_email(text: str) -> Optional[str]:
    """Extracts first valid email address from text."""
    pattern = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    matches = pattern.findall(text)
    for m in matches:
        cleaned = m.strip(".,;:()[]{}")
        if len(cleaned) > 5 and "@" in cleaned:
            return cleaned
    return None


def extract_phone(text: str) -> Optional[str]:
    """Extracts candidate contact phone number if available."""
    # Matches international and standard 10-12 digit phone numbers
    pattern = re.compile(
        r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}"
    )
    for match in pattern.finditer(text):
        raw = match.group().strip()
        digits = re.sub(r"\D", "", raw)
        if 10 <= len(digits) <= 13:
            return raw
    return None


def extract_github_url(text: str) -> Optional[str]:
    """Extracts the first GitHub profile URL from text."""
    pattern = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9_\-]+", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        url = match.group(0).strip(".,;:()[]{}")
        user = extract_github_username(url)
        if user:
            return f"https://github.com/{user}"
    return None


def extract_skills_heuristic(text: str) -> List[str]:
    """Matches known skills against the resume text using word boundaries."""
    found_skills: Set[str] = set()
    lower_text = f" {text.lower()} "

    for raw_skill, canonical_name in SKILL_TAXONOMY.items():
        # Match using word boundaries or symbol-safe boundaries
        escaped = re.escape(raw_skill)
        pattern = rf"(?:^|[\s,;|/()\[\]•\-])({escaped})(?:$|[\s,;|/()\[\]•\-])"
        if re.search(pattern, lower_text):
            found_skills.add(canonical_name)

    return sorted(list(found_skills))


def extract_sections(text: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Extracts content lines belonging to Projects, Experience, and Education sections.
    Returns (projects_lines, experience_lines, education_lines).
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    section_headers = {
        "projects": ["projects", "academic projects", "key projects", "personal projects", "project experience"],
        "experience": ["work experience", "experience", "professional experience", "internship", "employment history"],
        "education": ["education", "academics", "qualifications", "academic background"],
        "skills": ["skills", "technical skills", "skills & tools", "core competencies"],
    }

    current_section = "header"
    projects: List[str] = []
    experience: List[str] = []
    education: List[str] = []

    for line in lines:
        lower_line = line.lower()
        # Check if line is a section header
        is_header = False
        for sec, keywords in section_headers.items():
            if any(lower_line == kw or lower_line.startswith(kw + " ") or lower_line.startswith(kw + ":") for kw in keywords):
                current_section = sec
                is_header = True
                break

        if is_header:
            continue

        if current_section == "projects":
            projects.append(line)
        elif current_section == "experience":
            experience.append(line)
        elif current_section == "education":
            education.append(line)

    return projects, experience, education


def build_extracted_projects_heuristic(project_lines: List[str], full_text: str) -> List[ExtractedProject]:
    """Builds a structured list of ExtractedProject objects from section text."""
    ai_keywords = [
        "langchain", "langgraph", "llamaindex", "rag", "retrieval", "vector",
        "faiss", "chroma", "pinecone", "qdrant", "weaviate", "agent", "llm",
        "openai", "gemini", "embedding", "crewai", "autogen", "adk", "eval"
    ]
    shallow_indicators = [
        "wrapper", "simple chatbot", "api call to chatgpt", "streamlit wrapper",
        "chatgpt clone", "basic prompt"
    ]

    extracted_projects: List[ExtractedProject] = []
    combined_project_text = " ".join(project_lines)

    # If explicit project lines exist, group them by heading or bullets
    if project_lines:
        current_title = "Project 1"
        current_desc: List[str] = []

        for line in project_lines:
            if ("|" in line or line.isupper() or len(line.split()) <= 6) and not line.startswith(("-", "•", "*")):
                if current_desc:
                    desc_str = " ".join(current_desc)
                    lower_d = desc_str.lower()
                    has_ai = any(kw in lower_d for kw in ai_keywords)
                    is_shallow = any(sh in lower_d for sh in shallow_indicators)
                    extracted_projects.append(
                        ExtractedProject(
                            title=current_title,
                            description=desc_str[:300],
                            has_ai_elements=has_ai,
                            is_shallow_wrapper=is_shallow,
                        )
                    )
                    current_desc = []
                current_title = line
            else:
                current_desc.append(line)

        if current_desc:
            desc_str = " ".join(current_desc)
            lower_d = desc_str.lower()
            has_ai = any(kw in lower_d for kw in ai_keywords)
            is_shallow = any(sh in lower_d for sh in shallow_indicators)
            extracted_projects.append(
                ExtractedProject(
                    title=current_title,
                    description=desc_str[:300],
                    has_ai_elements=has_ai,
                    is_shallow_wrapper=is_shallow,
                )
            )

    # Fallback: if project sections weren't explicitly segmented, inspect full text
    if not extracted_projects:
        lower_full = full_text.lower()
        has_ai = any(kw in lower_full for kw in ai_keywords)
        is_shallow = any(sh in lower_full for sh in shallow_indicators)
        extracted_projects.append(
            ExtractedProject(
                title="Engineering Projects",
                description="Parsed from resume content",
                has_ai_elements=has_ai,
                is_shallow_wrapper=is_shallow,
            )
        )

    return extracted_projects


def extract_resume_data(
    text: str,
    file_path: str,
    llm_provider: Optional[BaseLLMProvider] = None,
) -> ExtractedResumeData:
    """
    Extracts all candidate fields from resume text.
    Combines LLM structured extraction (when configured) with deterministic heuristics.
    Guarantees non-crashing execution and schema compliance.
    """
    clean_raw = clean_text(text)

    # 1. Attempt LLM extraction if provider is available
    if llm_provider and llm_provider.is_available():
        llm_data, err = llm_provider.extract_semantic_data(clean_raw)
        if llm_data and not err:
            try:
                candidate_name = llm_data.get("candidate_name") or extract_candidate_name(clean_raw, file_path)
                email = llm_data.get("email") or extract_email(clean_raw)
                phone = llm_data.get("phone") or extract_phone(clean_raw)
                github_url = llm_data.get("github_url") or extract_github_url(clean_raw)

                raw_skills = llm_data.get("skills", [])
                heuristic_skills = extract_skills_heuristic(clean_raw)
                combined_skills = sorted(list(set(raw_skills + heuristic_skills)))

                raw_projects = llm_data.get("projects", [])
                structured_projects = [
                    ExtractedProject(
                        title=p.get("title", "Project"),
                        description=p.get("description", ""),
                        technologies=p.get("technologies", []),
                        has_ai_elements=p.get("has_ai_elements", False),
                    )
                    for p in raw_projects
                    if isinstance(p, dict)
                ]
                if not structured_projects:
                    proj_lines, _, _ = extract_sections(clean_raw)
                    structured_projects = build_extracted_projects_heuristic(proj_lines, clean_raw)

                return ExtractedResumeData(
                    candidate_name=candidate_name,
                    email=email,
                    phone=phone,
                    github_url=github_url,
                    skills=combined_skills,
                    projects=structured_projects,
                    experience=llm_data.get("experience", []),
                    education=llm_data.get("education", []),
                    raw_text_length=len(clean_raw),
                )
            except Exception as e:
                logger.warning(f"Error mapping LLM extraction data: {str(e)}; falling back to heuristics.")

    # 2. Rule-based / Heuristic extraction
    candidate_name = extract_candidate_name(clean_raw, file_path)
    email = extract_email(clean_raw)
    phone = extract_phone(clean_raw)
    github_url = extract_github_url(clean_raw)
    skills = extract_skills_heuristic(clean_raw)
    proj_lines, exp_lines, edu_lines = extract_sections(clean_raw)
    projects = build_extracted_projects_heuristic(proj_lines, clean_raw)

    return ExtractedResumeData(
        candidate_name=candidate_name,
        email=email,
        phone=phone,
        github_url=github_url,
        skills=skills,
        projects=projects,
        experience=exp_lines,
        education=edu_lines,
        raw_text_length=len(clean_raw),
    )
