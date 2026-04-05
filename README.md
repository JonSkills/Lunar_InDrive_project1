<div align="center">

# ⚖️ Legal Genome — Hybrid AI Legal Analyzer

> **Заң жобаларын сараптауға, қайшылықтарды анықтауға және реттеушілік әсерді (RIA) бағалауға арналған гибридті AI платформасы**

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Logic-Python%203.11-3776AB?style=flat-square&logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?style=flat-square&logo=scikitlearn)](https://scikit-learn.org/)
[![HybridAI](https://img.shields.io/badge/AI-Hybrid%20Mode-blueviolet?style=flat-square)](https://github.com/JonSkills/Lunar_InDrive_project1)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

</div>

---

## 📌 Проблема және шешім

### Текущее состояние

Заң шығару процесіндегі басты мәселе — нормативтік актілердің тым күрделілігі және олардың ішкі/сыртқы қайшылықтары. Бұл келесі қауіптерге әкеледі:

| Проблема | Заңдық салдары |
|----------|-------------|
| **Логикалық қайшылықтар** | Бір-біріне қарама-қайшы баптар құқық қолдануда коллизия тудырады |
| **Жемқорлық қаупі** | "Екіұшты" терминдер инспекторларға манипуляция жасауға мүмкіндік береді |
| **Нормалардың қайталануы** | Кодекстердің шамадан тыс көлемді болуы және тиімсіз реттеу |
| **Математикалық қателер** | Мерзімдер мен сандар арасындағы техникалық алшақтықтар |

### Решение: Legal Genome (Hybrid AI)

**Legal Genome** — бұл заңгердің "цифрлық көмекшісі". Ол тек мәтінді оқып қана қоймай, оны **Global Legal Knowledge Base** (Конституция және базалық нормалар) деректерімен салыстырады. 

> **Принцип:** AI — заңның "ДНҚ-сын" сканерлейді, ал заңгер эксперт жүйенің кері байланысы (Feedback Loop) арқылы нәтижелерді растайды.

---

## ✨ Ключевые возможности

### 1. 🧬 DNA Neural Graph (Visual Analysis)
- Заң нормалары арасындағы семантикалық байланыстарды интерактивті граф түрінде бейнелеу.
- Қайшылықтар (Red), дубликаттар (Orange) және жемқорлық қауіптері (Purple) түсті маркерлермен көрсетіледі.

### 2. 🤖 Hybrid AI Engine
- **Layer 1: Semantic Vectorization (TF-IDF)** — Мәтін ішіндегі ұқсастықтар мен қайталануларды табу.
- **Layer 2: Global Cross-Check** — Конституция және эталондық нормалар базасымен автоматты салыстыру.
- **Layer 3: Cognitive Logic** — Мәтіндегі "екіұшты" (vague) терминдерді анықтау.

### 3. 🛡️ RIA Analytics (Regulatory Impact Analysis)
- Заңның сапасын (Law Score) және оның реттеушілік әсерін (RIA Score) автоматты есептеу.
- **Explainability:** Әрбір балдың неге қойылғаны туралы толық AI-түсіндіру.

### 4. 🧠 Human-in-the-Loop Feedback
- Пайдаланушы "✅ Дұрыс" немесе "❌ Қате" батырмалары арқылы жүйені оқыта алады.
- Әрбір фидбек жүйеның балдық шкаласына (Law Score) тікелей әсер етіп, алгоритмді жетілдіреді.

### 5. 💡 Expert AI Recommendations
- Жүйе тек қатені тауып қана қоймай, оны түзету бойынша (мысалы: "Нормаларды біріктіру", "Мерзімді нақтылау") эксперттік ұсыныстар береді.

---

## 🏗️ Архитектура системы

```
┌─────────────────────────────────────────────────────────┐
│                 FRONTEND (Tailwind + Vis.js)             │
│  DNA Graph UI │ Executive Summary │ Feedback Console      │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│              BACKEND (Python FastAPI Service)             │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Hybrid Core  │  │ Global Base  │  │ Feedback API  │  │
│  │ TF-IDF / Cos │  │ Const / Laws │  │ ML Correction │  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘  │
│         │                 │                  │          │
│         └─────────────────┴──────────────────┘          │
│                           │                             │
│                    ┌──────▼──────┐                      │
│                    │ Recommendation Engine              │
│                    └─────────────┘                      │
└─────────────────────────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
    Input Legal Text    Knowledge Base     feedback.json
    (DNA Source)        (Logic Reference) (Learning Data)
```

---

## 🚀 Инструкция по запуску

### Требования
- **Python**: 3.10+
- **Space**: Minimal (Optimized requirements)

### 1. Клонировать репозиторий
```bash
git clone https://github.com/JonSkills/Lunar_InDrive_project1.git
cd Lunar_InDrive_project1
```

### 2. Виртуалды ортаны құру
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Іске қосу
```bash
uvicorn main:app --reload
```
*Жүйе `http://localhost:8000` адресінде қолжетімді болады.*

---

## 🛠️ Технологический стек

| Слой | Технология | Назначение |
|------|-----------|------------|
| **Backend** | FastAPI | Жылдам әрі жеңіл REST API |
| **AI/ML Core** | Scikit-learn | Векторлау және Косинустық ұқсастық |
| **Visualization** | Vis.js | Интерактивті Нейрондық Граф |
| **Styling** | Tailwind CSS | Премиум Dark-Glass UI |
| **Logic** | Custom Hybrid Engine | Заңдық білім базасымен интеграция |

---

## 📡 API Reference

| Метод | Эндпоинт | Описание |
|-------|----------|----------|
| `POST` | `/api/v1/analyze` | Мәтінді толық гибридті талдау (RIA, DNA Graph) |
| `POST` | `/api/v1/feedback` | Жүйені оқыту (Feedback loop) |

---

## 👥 Команда: Lunar InDrive Team
- **Bekbolat Bolebay** — Lead Fullstack / ML Engineer

---

<div align="center">

**Legal Genome** — Заңның тазалығы — қоғамның айнасы

*Сделано специально для хакатона inDrive | AI for Government*

</div>
