import re
import difflib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#12 technical domains plus behavioral assessment dimension.
DOMAIN_SPEC = {
    "Programming Languages": [
        ("Python", 1, ["python", "py", "python3"]),
        ("Java", 1, ["java"]),
        ("C++", 1, ["c++", "cpp"]),
        ("JavaScript", 1, ["javascript", "js", "node.js", "nodejs", "typescript"]),
    ],
    "Core CS": [
        ("DSA", 1, ["dsa", "data structures", "algorithms", "data structures and algorithms"]),
        ("OOP", 0, ["oop", "object oriented", "object-oriented"]),
        ("Operating Systems", 0, ["os", "operating system", "operating systems"]),
        ("DBMS", 0, ["dbms", "database", "databases", "rdbms"]),
        ("Computer Networks", 0, ["networks", "networking", "computer networks", "tcp/ip"]),
    ],
    "AI/ML": [
        ("Machine Learning", 0, ["ml", "machine learning", "scikit-learn", "sklearn"]),
        ("Deep Learning", 0, ["dl", "deep learning", "pytorch", "tensorflow", "neural networks"]),
        ("NLP", 0, ["nlp", "natural language processing"]),
        ("Generative AI", 0, ["genai", "generative ai", "llm", "llms"]),
        ("MLOps", 0, ["mlops", "model deployment"]),
    ],
    "Web Development": [
        ("Frontend", 0, ["frontend", "front-end", "react", "html", "css", "angular", "vue"]),
        ("Backend", 1, ["backend", "back-end", "django", "flask", "spring", "express"]),
        ("REST APIs", 0, ["rest", "rest api", "rest apis", "api", "apis"]),
    ],
    "Cloud/DevOps": [
        ("Docker", 0, ["docker", "containers", "containerization"]),
        ("Kubernetes", 0, ["kubernetes", "k8s"]),
        ("AWS", 0, ["aws", "ec2", "s3", "cloud"]),
        ("CI/CD", 0, ["ci/cd", "jenkins", "github actions", "cicd"]),
    ],
    "Data Engineering": [
        ("SQL", 1, ["sql", "mysql", "postgres", "postgresql"]),
        ("ETL", 0, ["etl", "data pipelines", "airflow"]),
        ("Spark", 0, ["spark", "pyspark"]),
    ],
    "Cybersecurity": [
        ("Security Fundamentals", 0, ["security", "cybersecurity", "cryptography"]),
        ("Application Security", 0, ["appsec", "application security", "owasp"]),
    ],
    "Mobile Development": [
        ("Android", 0, ["android", "kotlin"]),
        ("iOS", 0, ["ios", "swift", "swiftui"]),
        ("Cross-platform", 0, ["flutter", "react native", "cross-platform", "xamarin"]),
    ],
    "Systems/Architecture": [
        ("System Design", 0, ["system design", "architecture"]),
        ("Microservices", 0, ["microservices", "microservice"]),
        ("Distributed Systems", 0, ["distributed systems", "distributed"]),
    ],
    "Software Engineering": [
        ("Git", 0, ["git", "github", "gitlab", "version control"]),
        ("Testing", 0, ["testing", "pytest", "unit tests", "unit testing", "junit"]),
        ("Debugging", 0, ["debugging", "debugger", "troubleshooting"]),
    ],
    "Analytics": [
        ("Statistics", 0, ["statistics", "statistical", "probability"]),
        ("Data Visualization", 0, ["visualization", "matplotlib", "data visualization", "seaborn"]),
        ("Business Intelligence", 0, ["bi", "power bi", "tableau", "business intelligence"]),
    ],
    "Specialized Technologies": [
        ("IoT", 0, ["iot", "internet of things"]),
        ("Embedded Systems", 0, ["embedded", "embedded systems", "firmware", "microcontrollers"]),
        ("Robotics", 0, ["robotics", "ros"]),
        ("Blockchain", 0, ["blockchain", "solidity", "smart contracts"]),
    ],
    "Behavioral Competencies": [
        ("Communication & Leadership", 0, ["communication and leadership", "teamwork", "conflict resolution", "leadership", "stakeholder communication"]),
    ],
}

#prerequisite -> skill edges of the knowledge graph
PREREQS = [
    ("Python", "Machine Learning"),
    ("Statistics", "Machine Learning"),
    ("Machine Learning", "Deep Learning"),
    ("Deep Learning", "NLP"),
    ("NLP", "Generative AI"),
    ("Machine Learning", "MLOps"),
    ("Docker", "MLOps"),
    ("Docker", "Kubernetes"),
    ("Docker", "CI/CD"),
    ("Git", "CI/CD"),
    ("DBMS", "SQL"),
    ("SQL", "ETL"),
    ("ETL", "Spark"),
    ("Python", "Spark"),
    ("Computer Networks", "REST APIs"),
    ("Backend", "REST APIs"),
    ("REST APIs", "Microservices"),
    ("Microservices", "Distributed Systems"),
    ("Operating Systems", "Distributed Systems"),
    ("Computer Networks", "Security Fundamentals"),
    ("Security Fundamentals", "Application Security"),
    ("OOP", "Java"),
    ("Java", "Android"),
    ("JavaScript", "Frontend"),
    ("JavaScript", "Cross-platform"),
    ("Operating Systems", "Embedded Systems"),
    ("C++", "Embedded Systems"),
    ("Embedded Systems", "IoT"),
    ("Embedded Systems", "Robotics"),
    ("Distributed Systems", "Blockchain"),
    ("Statistics", "Data Visualization"),
    ("Data Visualization", "Business Intelligence"),
    ("SQL", "Business Intelligence"),
    ("DSA", "System Design"),
    ("Testing", "Debugging"),
]


class SkillRegistry:
    def __init__(self):
        self.names, self.domain, self.coding, self.alias_map = [], [], [], {}
        for dom, items in DOMAIN_SPEC.items():
            for n, c, al in items:
                self.names.append(n)
                self.domain.append(dom)
                self.coding.append(c)
        self.N = len(self.names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.domains = list(DOMAIN_SPEC)
        self.dom_idx = np.array([self.domains.index(d) for d in self.domain])
        self.coding_ok = np.array(self.coding, dtype=bool)

        for i, (dom, items) in enumerate([(d, it) for d, its in DOMAIN_SPEC.items() for it in its]):
            for al in items[2] + [items[0].lower()]:
                self.alias_map[al.lower()] = i

        G = nx.DiGraph()
        G.add_nodes_from(range(self.N))
        for u, v in PREREQS:
            G.add_edge(self.idx[u], self.idx[v], kind="prereq")
        assert nx.is_directed_acyclic_graph(G), "Knowledge graph must be a DAG!"
        self.kg = G
        self.topo = list(nx.topological_sort(G))
        self.prereq = [list(G.predecessors(i)) for i in range(self.N)]

        W = np.zeros((self.N, self.N))
        for i in range(self.N):  # sibling edges inside a domain
            for j in range(self.N):
                if i != j and self.domain[i] == self.domain[j]:
                    W[i, j] = 0.25
        for u, v in G.edges:
            W[u, v] = W[v, u] = max(W[u, v], 0.45)  # prerequisite edges are stronger
        self.W = W

        self.Ppre = np.zeros((self.N, self.N))  # row-normalised prerequisite adjacency
        for v in range(self.N):
            for u in self.prereq[v]:
                self.Ppre[v, u] = 1.0 / len(self.prereq[v])

        self._pat = {
            al: re.compile(r"(?<![a-z0-9+#/.])" + re.escape(al) + r"(?![a-z0-9+#])")
            for al in self.alias_map
        }

    def extract(self, text):
        """Map free text (resume/JD) onto registry IDs. Unknown skills are flagged."""
        t = text.lower()
        found = {}
        for al, pat in self._pat.items():
            n = len(pat.findall(t))
            if n:
                found[self.alias_map[al]] = found.get(self.alias_map[al], 0) + n

        unknown = []
        for line in text.splitlines():
            if re.match(r"\s*(technical\s+)?skills?\s*[:\-]", line, re.I):
                for tok in re.split(r"[,;/|]", line.split(":", 1)[-1]):
                    tok = tok.strip().lower()
                    if not tok or tok in self.alias_map or any(p.search(tok) for p in self._pat.values()):
                        continue
                    m = difflib.get_close_matches(tok, list(self.alias_map), n=1, cutoff=0.86)
                    if m:
                        found[self.alias_map[m[0]]] = found.get(self.alias_map[m[0]], 0) + 1
                    else:
                        unknown.append(tok)
        return dict(found=found, unknown=sorted(set(unknown)))


REG = SkillRegistry()


def plot_kg(reg=REG, path=None):
    G = nx.Graph()
    G.add_nodes_from(range(reg.N))
    for i in range(reg.N):
        for j in range(i + 1, reg.N):
            if reg.W[i, j] > 0:
                G.add_edge(i, j, w=reg.W[i, j])
    pos = nx.spring_layout(G, seed=3, k=0.6, iterations=200)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(13, 9))
    nx.draw_networkx_edges(G, pos, alpha=0.25, ax=ax)
    for u, v in reg.kg.edges:
        ax.annotate("", xy=pos[v], xytext=pos[u], arrowprops=dict(arrowstyle="->", color="k", alpha=0.5))
    nx.draw_networkx_nodes(G, pos, node_color=[cmap(reg.dom_idx[i]) for i in range(reg.N)], node_size=420, ax=ax)
    nx.draw_networkx_labels(G, pos, {i: reg.names[i] for i in range(reg.N)}, font_size=7, ax=ax)
    ax.set_title("Skill Knowledge Graph (arrows = prerequisite, colour = domain)")
    ax.axis("off")
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig, ax
