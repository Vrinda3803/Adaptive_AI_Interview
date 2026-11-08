
import os
import re
import pymupdf
from PIL import Image
from docx import Document

SKILL_KEYWORDS = {
    "Python": ["python", "pandas", "numpy"],
    "SQL": ["sql", "mysql", "postgresql", "sqlite"],
    "Machine Learning": [
        "machine learning", "scikit-learn", "tensorflow", "pytorch"
    ],
    "Deep Learning": [
        "deep learning", "neural networks", "cnn", "rnn", "lstm"
    ],
    "Data Structures": [
        "data structures", "algorithms", "linked list",
        "stack", "queue", "tree"
    ],
    "Problem Solving": [
        "problem solving", "competitive programming",
        "coding challenges", "leetcode"
    ],
    "Java": ["java", "spring boot"],
    "C++": ["c++"],
    "Web Development": [
        "html", "css", "javascript", "react"
    ],
    "Artificial Intelligence": [
        "artificial intelligence", "deep learning",
        "llm", "ai"
    ],
    "Operating Systems": [
        "operating system", "operating systems", "os"
    ],
    "Computer Networks": [
        "computer networks", "networking"
    ],
    "Embedded Systems": [
        "embedded systems", "raspberry pi", "esp32", "arduino"
    ],
    "Database Management": [
        "database management", "mongodb", "oracle"
    ]
}

def extract_pdf_text(file_path):
    document = pymupdf.open(file_path)
    extracted_text = []
    for page in document:
        text = page.get_text("text").strip()
        if not text:
            print("Applying OCR...")
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
            image = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )
            try:
                import pytesseract
            except ImportError as exc:
                raise RuntimeError("Scanned PDF pages require the optional 'pytesseract' package and Tesseract OCR executable.") from exc
            pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", pytesseract.pytesseract.tesseract_cmd)
            text = pytesseract.image_to_string(
                image,
                lang="eng"
            )
        extracted_text.append(text)
    document.close()

    return "\n".join(extracted_text)

#etxract using ocr
def extract_docx_text(file_path):
    document = Document(file_path)
    text = ""
    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                text += cell.text + "\n"
    return text


#extract txt from pdf,doc,text
def extract_resume_text(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".pdf":
        return extract_pdf_text(file_path)
    elif extension == ".docx":
        return extract_docx_text(file_path)
    elif extension == ".txt":
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    else:
        raise ValueError(
            "Unsupported file format. Use PDF, DOCX or TXT."
        )


#retrieve tech skills
def identify_skills(resume_text):
    text = resume_text.lower()
    identified_skills = []
    for skill, keywords in SKILL_KEYWORDS.items():
        for keyword in keywords:
            pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
            if re.search(pattern, text):
                identified_skills.append(skill)
                break
    return identified_skills


#etxract sections
def extract_section(resume_text, headings):
    lines = resume_text.splitlines()
    collecting = False
    section_text = []
    all_headings = [
    "education", "skills", "skills & software proficiencies",
    "experience", "work experience", "projects",
    "certifications", "achievements", "summary",
    "professional summary", "internships",
    "academic background", "volunteering",
    "community outreach program", "publications",
    "technical skills"
]

    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
        normalized = cleaned_line.lower().strip(" :")
        if normalized in headings:
            collecting = True
            continue
        if collecting:
            if normalized in all_headings:
                break
            section_text.append(cleaned_line)
    return "\n".join(section_text)


def normalize_heading(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = " ".join(text.split())
    return text


def extract_section(resume_text, headings):
    lines = resume_text.splitlines()
    collecting = False
    section_text = []
    all_headings = [
        "education",
        "skills",
        "skills software proficiencies",
        "technical skills",
        "experience",
        "work experience",
        "internships",
        "projects",
        "academic projects",
        "personal projects",
        "certifications",
        "achievements",
        "summary",
        "professional summary",
        "volunteering",
        "community outreach program",
        "publications"
    ]

    target_headings = [
        normalize_heading(h) for h in headings
    ]
    stop_headings = [
        normalize_heading(h) for h in all_headings
    ]
    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
        normalized = normalize_heading(cleaned_line)
        if normalized in target_headings:
            collecting = True
            continue
        if normalized in stop_headings:
            if collecting:
                break
        if collecting:
            section_text.append(cleaned_line)
    return "\n".join(section_text)

#building profile
def build_candidate_profile(file_path, profile_generator=None):
    resume_text = extract_resume_text(file_path)
    if not resume_text.strip():
        raise ValueError(
            "Could not extract text from resume. "
            "Please upload a readable PDF, DOCX or TXT file."
        )
    skills = identify_skills(resume_text)
    volunteering = extract_section(
        resume_text,
        ["volunteering"]
    )
    community_outreach = extract_section(
        resume_text,
        ["community outreach program"]
    )
    projects = extract_section(
        resume_text,
        ["projects", "academic projects", "personal projects"]
    )
    experience = extract_section(
        resume_text,
        ["experience", "work experience", "internships"]
    )
    education = extract_section(
        resume_text,
        ["education", "academic background"]
    )
    candidate_profile = {
        "resume_text": resume_text,
        "skills": skills,
        "projects": projects,
        "experience": experience,
        "education": education,
        "volunteering": volunteering,
        "community_outreach": community_outreach
    }
    from llm.profile_extractor import extract_resume_profile
    return extract_resume_profile(resume_text, candidate_profile, generator=profile_generator)
