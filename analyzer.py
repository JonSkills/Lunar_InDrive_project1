import re
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =================================================================================
# ЛОКАЛЬНЫЕ ML МОДЕЛИ (ОФФЛАЙН АРХИТЕКТУРА - LEGAL GENOME PROJECT)
# ГИБРИДТІ ЖҮЙЕ: Статистика + Білім базасы + Кері байланыс + Рекомендациялар + RIA
# =================================================================================

GLOBAL_LEGAL_KNOWLEDGE = [
    "Конституция - высший закон государства. Барлық заңдар Конституцияға сәйкес келуі тиіс.",
    "Нормативные правовые акты не имеют обратной силы, если только они не смягчают ответственность.",
    "Заңды білмеу жауапкершіліктен босатпайды. Ignorantia juris non excusat.",
    "Никто не может быть судим дважды за одно и то же правонарушение.",
    "Срок обжалования решения суда составляет 1 месяц со дня вынесения.",
    "Мемлекеттік сатып алу туралы заң ашықтық пен бәсекелестікті талап етеді."
]

def extract_sentences(text: str):
    text = re.sub(r'^\s*\{\s*"[A-Za-z0-9_]+"\s*:\s*"', '', text)
    text = re.sub(r'"\s*\}\s*$', '', text)
    text = text.replace('\\n', ' ').replace('\\"', '"').replace('\\t', ' ')
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw_sentences if len(s.strip()) > 5]

def analyze_text(text: str):
    sentences = extract_sentences(text)
    if len(sentences) < 2:
        return {
            "contradictions": "Текст слишком короткий.",
            "duplicates": "-",
            "issues": "-",
            "recommendations": [],
            "summary": "Мәтін тым қысқа.",
            "ria_score": 0,
            "law_score": 0,
            "explanation": "Слишком короткий текст.",
            "note": "Пайдаланушыға ескерту: Мәтінді толықтырыңыз.",
            "graph_data": {"nodes": [], "edges": []}
        }

    nodes, edges = [], []
    contradictions, duplicates, recommendations = [], [], []
    issues = set()

    for i, seq in enumerate(sentences):
        words = seq.split()
        short_label = " ".join(words[:4]) + ("..." if len(words)>4 else "")
        nodes.append({"id": i, "label": f"Норма {i+1}\n{short_label}", "full_text": seq, "details": [], "group": "normal"})

    try:
        all_corpus = sentences + GLOBAL_LEGAL_KNOWLEDGE
        vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
        tfidf_matrix = vectorizer.fit_transform(all_corpus)
        input_matrix = tfidf_matrix[:len(sentences)]
        cosine_scores = cosine_similarity(input_matrix, input_matrix)
        
        knowledge_matrix = tfidf_matrix[len(sentences):]
        for i in range(len(sentences)):
            global_sim = cosine_similarity(input_matrix[i], knowledge_matrix)
            if global_sim.max() > 0.3:
                nodes[i]["details"].append(f"📘 Глобалды базамен сәйкестік: {global_sim.max()*100:.1f}%")
            
            for j in range(i + 1, len(sentences)):
                score = cosine_scores[i][j]
                if score > 0.45:
                    duplicates.append(f"- Дублирование: Норма {i+1} мен {j+1}")
                    edges.append({"from": i, "to": j, "color": "#f59e0b", "width": score * 5, "dashes": True})
                    nodes[i]["group"] = "duplicate"; nodes[j]["group"] = "duplicate"
                    recommendations.append(f"💡 {i+1} және {j+1} баптарды біріктіру немесе біреуін алып тастау ұсынылады.")
                
                s1_low, s2_low = sentences[i].lower(), sentences[j].lower()
                permit = ["разрешен", "вправе", "рұқсат", "құқылы"]
                forbid = ["запрещен", "тыйым", "болмайды"]
                
                if (any(w in s1_low for w in permit) and any(w in s2_low for w in forbid)) or \
                   (any(w in s1_low for w in forbid) and any(w in s2_low for w in permit)):
                    if score > 0.03:
                        contradictions.append(f"- ҚАЙШЫЛЫҚ: Норма {i+1} мен {j+1}")
                        edges.append({"from": i, "to": j, "color": "#ef4444", "width": 4})
                        nodes[i]["group"] = "conflict"; nodes[j]["group"] = "conflict"
                        recommendations.append(f"🛑 {i+1} және {j+1} баптар арасындағы қайшылықты шешу үшін құзыретті нақтылау қажет.")
    except Exception as e: print(str(e))

    vague = ["по усмотрению", "қажет болған жағдайда", "өзге адамдар"]
    for i, s in enumerate(sentences):
        if any(v in s.lower() for v in vague):
            issues.add(f"- Риск: {s[:50]}...")
            nodes[i]["color"] = "#a855f7"
            recommendations.append(f"🟣 {i+1}-баптағы 'екұшты' сөздерді нақты мерзіммен немесе шартпен алмастыру ұсынылады.")

    score = 100 - (15 * len(duplicates)) - (35 * len(contradictions)) - (10 * len(issues))
    if os.path.exists("feedback.json"):
        with open("feedback.json", "r", encoding="utf-8") as f:
            fb = json.load(f)
            score += (len([x for x in fb if not x['is_correct']]) * 2)

    # NEW: RIA Score (Regulatory Impact Analysis) - Симуляция сараптамалық бағалау
    ria_score = score * 0.95 # Жүйелік реттеу сапасы
    summary = f"Жалпы сараптама: Мәтінде {len(contradictions)} қайшылық және {len(issues)} жемқорлық қаупі табылды. "
    if score > 80: summary += "Заң жобасының сапасы жоғары, қолдануға ұсынылады."
    elif score > 50: summary += "Орташа қауіп деңгейі. Түзетулер енгізу қажет."
    else: summary += "Сын көтермейді. Түбегейлі қайта қарау ұсынылады."

    return {
        "contradictions": "\n\n".join(contradictions) if contradictions else "Табылмады.",
        "duplicates": "\n\n".join(duplicates) if duplicates else "Жоқ.",
        "issues": "\n\n".join(list(issues)) if issues else "Таза.",
        "recommendations": list(set(recommendations)),
        "summary": summary,
        "ria_score": int(ria_score),
        "law_score": int(max(0, min(100, score))),
        "explanation": "Hybrid AI Mode: Статистикалық талдау + Global Legal Base + Конструктивті ұсыныстар + RIA Сараптамасы.",
        "note": "🚀 Жүйе эксперттік ұсыныстарды және RIA сараптамасын сгенерирледі.",
        "graph_data": {"nodes": nodes, "edges": edges}
    }
