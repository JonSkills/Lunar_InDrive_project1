import re
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# =================================================================================
# ЛОКАЛЬНЫЕ ML МОДЕЛИ + GEMINI HYBRID REASONING (LEGAL GENOME)
# =================================================================================

MODEL = None
GLOBAL_LEGAL_KNOWLEDGE = [
    "Конституция - высший закон государства. Барлық заңдар Конституцияға сәйкес келуі тиіс.",
    "Нормативные правовые акты не имеют обратной силы, если только они не смягчают ответственность.",
    "Заңды білмеу жауапкершіліктен босатпайды. Ignorantia juris non excusat.",
    "Никто не может быть судим дважды за одно и то же правонарушение.",
    "Срок обжалования решения суда составляет 1 месяц со дня вынесения.",
    "Мемлекеттік сатып алу туралы заң ашықтық пен бәсекелестікті талап етеді."
]

def get_model():
    global MODEL
    if MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        except:
            MODEL = "TFIDF"
    return MODEL

def get_gemini_summary(text, contradictions, issues):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        # Using 2.0 Flash Lite for speed/efficiency
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        prompt = f"""
        Сен кәсіби заңгер-сарапшысың. Төмендегі заң мәтіні мен табылған қателерге сүйеніп, заңгерлік қорытынды (Executive Summary) жаз.
        
        МӘТІН: {text[:2000]}
        
        ТАБЫЛҒАН ҚАЙШЫЛЫҚТАР: {contradictions}
        ЖЕМҚОРЛЫҚ ҚАУІПТЕРІ: {issues}
        
        ТАЛАПТАР:
        1. Қазақ тілінде жаз.
        2. Қысқа әрі нақты бол.
        3. Заңның сапасына баға бер (Сын көтермейді / Орташа / Жақсы).
        4. Қандай бапты тез арада өзгерту керектігін айт.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return None

def extract_sentences(text: str):
    text = re.sub(r'^\s*\{\s*"[A-Za-z0-9_]+"\s*:\s*"', '', text)
    text = re.sub(r'"\s*\}\s*$', '', text)
    text = text.replace('\\n', ' ').replace('\\"', '"').replace('\\t', ' ')
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw_sentences if len(s.strip()) > 5]

def analyze_text(text: str):
    sentences = extract_sentences(text)
    if len(sentences) < 1:
        return {"error": "Текст слишком короткий"}

    model = get_model()
    nodes, edges = [], []
    contradictions, duplicates, recommendations = [], [], []
    issues = set()

    for i, seq in enumerate(sentences):
        words = seq.split()
        short_label = " ".join(words[:4]) + ("..." if len(words)>4 else "")
        nodes.append({"id": i, "label": f"Норма {i+1}\n{short_label}", "full_text": seq, "details": [], "group": "normal"})

    try:
        if model != "TFIDF":
            embeddings = model.encode(sentences + GLOBAL_LEGAL_KNOWLEDGE)
            input_embeddings = embeddings[:len(sentences)]
            knowledge_embeddings = embeddings[len(sentences):]
            cosine_scores = cosine_similarity(input_embeddings, input_embeddings)
        else:
            vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
            tfidf_matrix = vectorizer.fit_transform(sentences + GLOBAL_LEGAL_KNOWLEDGE)
            input_embeddings = tfidf_matrix[:len(sentences)]
            knowledge_embeddings = tfidf_matrix[len(sentences):]
            cosine_scores = cosine_similarity(input_embeddings, input_embeddings)

        for i in range(len(sentences)):
            global_sim = cosine_similarity(input_embeddings[i].reshape(1, -1), knowledge_embeddings)
            if global_sim.max() > 0.4:
                nodes[i]["details"].append(f"📘 Глобалды базамен сәйкестік: {global_sim.max()*100:.1f}%")
            
            for j in range(i + 1, len(sentences)):
                score = cosine_scores[i][j]
                dup_threshold = 0.85 if model != "TFIDF" else 0.45
                conflict_threshold = 0.4 if model != "TFIDF" else 0.05

                if score > dup_threshold:
                    duplicates.append(f"- Дублирование: Норма {i+1} мен {j+1}")
                    edges.append({"from": i, "to": j, "color": "#f59e0b", "width": score * 5, "dashes": True})
                    nodes[i]["group"] = "duplicate"; nodes[j]["group"] = "duplicate"
                    nodes[i]["details"].append(f"🟠 Семантикалық Дубликат ({j+1}-бап)")
                    nodes[j]["details"].append(f"🟠 Семантикалық Дубликат ({i+1}-бап)")
                    recommendations.append(f"💡 {i+1} және {j+1} баптарды біріктіру ұс.")
                
                s1_low, s2_low = sentences[i].lower(), sentences[j].lower()
                permit = ["разрешен", "вправе", "рұқсат", "құқылы", "болады", "право", "имеет"]
                forbid = ["запрещен", "тыйым", "болмайды", "етілмейді", "не вправе", "не допускается", "лишать"]
                
                if (any(w in s1_low for w in permit) and any(w in s2_low for w in forbid)) or \
                   (any(w in s1_low for w in forbid) and any(w in s2_low for w in permit)):
                    if score > conflict_threshold:
                        contradictions.append(f"- ҚАЙШЫЛЫҚ (Mutation): Норма {i+1} мен {j+1}")
                        edges.append({"from": i, "to": j, "color": "#ef4444", "width": 4})
                        nodes[i]["group"] = "conflict"; nodes[j]["group"] = "conflict"
                        nodes[i]["details"].append(f"🔴 МУТАЦИЯ (Қайшылық): {j+1}-баппен")
                        nodes[j]["details"].append(f"🔴 МУТАЦИЯ (Қайшылық): {i+1}-баппен")
    except Exception as e: print(str(e))

    vague = ["по усмотрению", "қажет болған жағдайда", "өзге адамдар", "ерекше жағдайларда"]
    for i, s in enumerate(sentences):
        if any(v in s.lower() for v in vague):
            issues.add(f"- Риск: {s[:50]}...")
            nodes[i]["color"] = "#a855f7"
            nodes[i]["details"].append("🟣 Коррупциогенный риск (Екұшты сөз)")

    raw_score = 100 - (15 * len(duplicates)) - (35 * len(contradictions)) - (10 * len(issues))
    score = max(0, min(100, raw_score))
    ria_score = max(0, int(score * 0.95))

    # GEMINI HYBRID SUMMARY
    gemini_sum = get_gemini_summary(text, contradictions, list(issues))
    if gemini_sum:
        summary = gemini_sum
    else:
        summary = f"Жалпы сараптама: Мәтінде {len(contradictions)} қайшылық табылды. "
        summary += "Түзетулер енгізу қажет." if score < 80 else "Заң сапалы."

    return {
        "contradictions": "\n\n".join(contradictions) if contradictions else "Табылмады.",
        "duplicates": "\n\n".join(duplicates) if duplicates else "Жоқ.",
        "issues": "\n\n".join(list(issues)) if issues else "Таза.",
        "recommendations": list(set(recommendations)),
        "summary": summary,
        "ria_score": ria_score,
        "law_score": int(score),
        "explanation": f"Mode: Hybrid (Local ML + Gemini Reasoning)",
        "graph_data": {"nodes": nodes, "edges": edges}
    }
