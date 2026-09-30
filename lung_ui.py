"""
================================================
  Lung Disease Analysis System
  Multilingual: English | Telugu | Hindi
================================================
  Run: streamlit run lung_simple.py
  Place in same folder as best_model_full.pth
================================================
"""

import os
import numpy as np
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# ──────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Lung Disease Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model_full.pth")

# ──────────────────────────────────────────────
#  TRANSLATIONS
# ──────────────────────────────────────────────
T = {
    # ── General ──
    "lang_label": {
        "English": "🌐 Language",
        "Telugu":  "🌐 భాష",
        "Hindi":   "🌐 भाषा",
    },
    "title": {
        "English": "Lung Disease Analysis System",
        "Telugu":  "ఊపిరితిత్తుల వ్యాధి విశ్లేషణ వ్యవస్థ",
        "Hindi":   "फेफड़ों की बीमारी विश्लेषण प्रणाली",
    },
    "hero_sub": {
        "English": "Chest X-ray analysis for early detection of Tuberculosis, Pneumonia, and Pneumothorax. Upload a scan to get instant predictions and clinical recommendations.",
        "Telugu":  "క్షయవ్యాధి, న్యుమోనియా మరియు న్యుమోత్రాక్స్‌ను ముందుగా గుర్తించడానికి ఛాతీ X-రే విశ్లేషణ. తక్షణ అంచనాలు మరియు వైద్య సూచనలు పొందండి.",
        "Hindi":   "तपेदिक, निमोनिया और न्यूमोथोरैक्स की शीघ्र पहचान के लिए छाती एक्स-रे विश्लेषण। त्वरित भविष्यवाणी और नैदानिक सिफारिशें प्राप्त करें।",
    },
    "ai_note": {
        "English": "🔬 Screening Tool: This system provides preliminary analysis only. Results are not a medical diagnosis. Please consult a qualified pulmonologist or chest physician for professional advice.",
        "Telugu":  "🔬 స్క్రీనింగ్ సాధనం: ఈ వ్యవస్థ కేవలం ప్రాథమిక విశ్లేషణను అందిస్తుంది. ఫలితాలు వైద్య నిర్ధారణ కాదు. వృత్తిపరమైన సలహా కోసం అర్హత గల వైద్యుడిని సంప్రదించండి.",
        "Hindi":   "🔬 स्क्रीनिंग टूल: यह प्रणाली केवल प्रारंभिक विश्लेषण प्रदान करती है। परिणाम चिकित्सा निदान नहीं हैं। कृपया पेशेवर सलाह के लिए योग्य चिकित्सक से परामर्श करें।",
    },

    # ── Model Overview ──
    "overview_title": {
        "English": "📊 Model Overview",
        "Telugu":  "📊 మోడల్ అవలోకనం",
        "Hindi":   "📊 मॉडल अवलोकन",
    },
    "accuracy_label": {
        "English": "Model Accuracy",
        "Telugu":  "మోడల్ ఖచ్చితత్వం",
        "Hindi":   "मॉडल सटीकता",
    },
    
    # ── Analysis ──
    "analysis_title": {
        "English": "🔬 Start Analysis",
        "Telugu":  "🔬 విశ్లేషణ ప్రారంభించండి",
        "Hindi":   "🔬 विश्लेषण शुरू करें",
    },
    "upload_label": {
        "English": "Upload a chest X-ray image (JPG, PNG, BMP)",
        "Telugu":  "ఛాతీ X-రే చిత్రాన్ని అప్లోడ్ చేయండి (JPG, PNG, BMP)",
        "Hindi":   "छाती का X-रे चित्र अपलोड करें (JPG, PNG, BMP)",
    },
    "upload_help": {
        "English": "Upload a frontal PA or AP chest X-ray for best results.",
        "Telugu":  "అత్యుత్తమ ఫలితాల కోసం ముందు PA లేదా AP ఛాతీ X-రే అప్లోడ్ చేయండి.",
        "Hindi":   "सर्वोत्तम परिणामों के लिए फ्रंटल PA या AP छाती X-रे अपलोड करें।",
    },
    "file_label":   { "English":"File",       "Telugu":"ఫైల్",         "Hindi":"फ़ाइल"        },
    "size_label":   { "English":"Size",       "Telugu":"పరిమాణం",      "Hindi":"आकार"         },
    "dim_label":    { "English":"Dimensions", "Telugu":"కొలతలు",       "Hindi":"आयाम"         },
    "status_label": { "English":"Status",     "Telugu":"స్థితి",       "Hindi":"स्थिति"       },
    "ready_label":  { "English":"Ready for analysis", "Telugu":"విశ్లేషణకు సిద్ధంగా ఉంది", "Hindi":"विश्लेषण के लिए तैयार" },
    "analyze_btn":  { "English":"🚀 Start Analysis", "Telugu":"🚀 విశ్లేషణ ప్రారంభించండి", "Hindi":"🚀 विश्लेषण शुरू करें" },
    "spinner_msg":  { "English":"Analyzing X-ray...", "Telugu":"X-రే విశ్లేషిస్తోంది...", "Hindi":"X-रे विश्लेषण हो रहा है..." },

    # ── Results ──
    "multi_detected": {
        "English": "Multiple Diseases Detected",
        "Telugu":  "బహుళ వ్యాధులు గుర్తించబడ్డాయి",
        "Hindi":   "एकाधिक बीमारियाँ पाई गईं",
    },
    "multi_sub": {
        "English": "Signs of more than one lung condition found. Immediate medical consultation is strongly advised.",
        "Telugu":  "ఒకటి కంటే ఎక్కువ ఊపిరితిత్తుల పరిస్థితి సంకేతాలు కనుగొనబడ్డాయి. తక్షణ వైద్య సంప్రదింపు ఖచ్చితంగా సూచించబడింది.",
        "Hindi":   "एक से अधिक फेफड़ों की स्थिति के संकेत पाए गए। तत्काल चिकित्सा परामर्श की दृढ़ता से सलाह दी जाती है।",
    },
    "primary_diag": {
        "English": "Primary Diagnosis",
        "Telugu":  "ప్రాథమిక నిర్ధారణ",
        "Hindi":   "प्राथमिक निदान",
    },
    "recs_for": {
        "English": "💊 Recommendations for",
        "Telugu":  "💊 సూచనలు",
        "Hindi":   "💊 सिफारिशें",
    },
    "severity_label": {
        "English": "Severity",
        "Telugu":  "తీవ్రత",
        "Hindi":   "गंभीरता",
    },
    "disclaimer": {
        "English": "⚠️ Disclaimer: This analysis is for preliminary screening only and does not constitute a medical diagnosis. Please consult a qualified pulmonologist or physician for professional medical advice, diagnosis, and treatment. Do not make health decisions based solely on these results.",
        "Telugu":  "⚠️ నిరాకరణ: ఈ విశ్లేషణ కేవలం ప్రాథమిక స్క్రీనింగ్ కోసం మాత్రమే మరియు ఇది వైద్య నిర్ధారణ కాదు. వృత్తిపరమైన వైద్య సలహా, నిర్ధారణ మరియు చికిత్స కోసం అర్హత గల వైద్యుడిని సంప్రదించండి.",
        "Hindi":   "⚠️ अस्वीकरण: यह विश्लेषण केवल प्रारंभिक जांच के लिए है और चिकित्सा निदान नहीं है। पेशेवर चिकित्सा सलाह, निदान और उपचार के लिए योग्य चिकित्सक से परामर्श करें। केवल इन परिणामों के आधार पर स्वास्थ्य निर्णय न लें।",
    },

    # ── Severity names ──
    "sev_high":   { "English":"High Risk",  "Telugu":"అధిక ప్రమాదం",  "Hindi":"उच्च जोखिम"  },
    "sev_medium": { "English":"Medium Risk","Telugu":"మధ్యస్థ ప్రమాదం","Hindi":"मध्यम जोखिम" },
    "sev_low":    { "English":"Low Risk",   "Telugu":"తక్కువ ప్రమాదం", "Hindi":"कम जोखिम"    },
    "sev_none":   { "English":"Healthy",    "Telugu":"ఆరోగ్యంగా ఉంది", "Hindi":"स्वस्थ"      },

    # ── About section ──
    "about_title": {
        "English": "🩺 About the Diseases",
        "Telugu":  "🩺 వ్యాధుల గురించి",
        "Hindi":   "🩺 बीमारियों के बारे में",
    },
    "tb_name":    { "English":"🦠 Tuberculosis (TB)",  "Telugu":"🦠 క్షయవ్యాధి (TB)",  "Hindi":"🦠 तपेदिक (TB)"       },
    "pn_name":    { "English":"🫧 Pneumonia",           "Telugu":"🫧 న్యుమోనియా",        "Hindi":"🫧 निमोनिया"          },
    "pt_name":    { "English":"💨 Pneumothorax",        "Telugu":"💨 న్యుమోత్రాక్స్",    "Hindi":"💨 न्यूमोथोरैक्स"    },
    "tb_desc":    {
        "English":"A serious bacterial infection caused by Mycobacterium tuberculosis. Spreads through airborne droplets when an infected person coughs or sneezes.",
        "Telugu": "మైకోబాక్టీరియం ట్యుబర్‌క్యులోసిస్ వల్ల కలిగే తీవ్రమైన బ్యాక్టీరియా సంక్రమణ. సోకిన వ్యక్తి దగ్గినప్పుడు లేదా తుమ్మినప్పుడు గాలి ద్వారా వ్యాపిస్తుంది.",
        "Hindi":  "माइकोबैक्टीरियम ट्यूबरकुलोसिस के कारण होने वाला गंभीर जीवाणु संक्रमण। संक्रमित व्यक्ति के खांसने या छींकने पर हवाई बूंदों के माध्यम से फैलता है।",
    },
    "pn_desc":    {
        "English":"Infection that inflames the air sacs (alveoli) in one or both lungs. Air sacs fill with fluid or pus causing breathing difficulties.",
        "Telugu": "ఒక లేదా రెండు ఊపిరితిత్తులలోని గాలి సంచులను (అల్వియోలి) తాకే సంక్రమణ. శ్వాస తీసుకోవడంలో ఇబ్బంది కలిగించే ద్రవం లేదా చీముతో గాలి సంచులు నిండిపోతాయి.",
        "Hindi":  "संक्रमण जो एक या दोनों फेफड़ों में वायु की थैलियों (एल्वियोली) को सूज देता है। वायु थैलियाँ तरल पदार्थ या मवाद से भर जाती हैं जिससे सांस लेने में कठिनाई होती है।",
    },
    "pt_desc":    {
        "English":"Collapsed lung caused by air leaking into the pleural space between the lung and chest wall, pushing on the outside of the lung.",
        "Telugu": "ఊపిరితిత్తుల మరియు ఛాతీ గోడ మధ్య ఉన్న ప్లూరల్ స్పేస్‌లో గాలి లీకవడం వల్ల కలిగే కుప్పకూలిన ఊపిరితిత్తు.",
        "Hindi":  "फेफड़े और छाती की दीवार के बीच के फुफ्फुस स्थान में हवा के रिसने से फेफड़ा ढह जाता है।",
    },
    "tb_symptoms": {
        "English":["Persistent cough lasting 3 or more weeks","Coughing up blood or thick mucus","Night sweats and low-grade fever","Unexplained weight loss and fatigue","Chest pain during breathing"],
        "Telugu": ["3 లేదా అంతకంటే ఎక్కువ వారాలు నిరంతర దగ్గు","రక్తం లేదా మందపాటి శ్లేష్మం దగ్గినప్పుడు వెలువడటం","రాత్రి చెమటలు మరియు తక్కువ జ్వరం","అకారణ బరువు తగ్గడం మరియు అలసట","శ్వాస తీసుకోనప్పుడు ఛాతీ నొప్పి"],
        "Hindi":  ["3 या अधिक सप्ताह तक लगातार खांसी","खून या गाढ़ा बलगम खांसना","रात को पसीना और हल्का बुखार","अनजाने में वजन कम होना और थकान","सांस लेते समय सीने में दर्द"],
    },
    "pn_symptoms": {
        "English":["Cough with phlegm or pus","Fever, chills, and sweating","Shortness of breath even at rest","Sharp chest pain while breathing","Fatigue and loss of appetite"],
        "Telugu": ["కఫం లేదా చీముతో దగ్గు","జ్వరం, చలి మరియు చెమట","విశ్రాంతిలో కూడా శ్వాస తీసుకోవడం కష్టం","శ్వాస తీసుకోనప్పుడు పదునైన ఛాతీ నొప్పి","అలసట మరియు ఆకలి తగ్గడం"],
        "Hindi":  ["बलगम या मवाद के साथ खांसी","बुखार, ठंड और पसीना","आराम करते समय भी सांस की तकलीफ","सांस लेते समय तेज सीने में दर्द","थकान और भूख न लगना"],
    },
    "pt_symptoms": {
        "English":["Sudden sharp chest or shoulder pain","Shortness of breath with rapid onset","Rapid heart rate (tachycardia)","Low blood pressure in severe cases","Bluish skin (cyanosis) in critical cases"],
        "Telugu": ["అకస్మాత్తుగా ఛాతీ లేదా భుజంలో పదునైన నొప్పి","తక్షణ శ్వాస ఆడకపోవడం","వేగవంతమైన హృదయ స్పందన","తీవ్రమైన సందర్భాలలో తక్కువ రక్తపోటు","విమర్శనాత్మక సందర్భాలలో నీలిరంగు చర్మం"],
        "Hindi":  ["अचानक तेज सीने या कंधे का दर्द","तेजी से सांस लेने में कठिनाई","तेज हृदय गति (टैकीकार्डिया)","गंभीर मामलों में निम्न रक्तचाप","गंभीर मामलों में नीली त्वचा (सायनोसिस)"],
    },

    # ── Footer ──
    "footer": {
        "English": "🫁 Lung Disease Analysis System | EfficientNet-B2 | For screening purposes only | Always consult a certified physician",
        "Telugu":  "🫁 ఊపిరితిత్తుల వ్యాధి విశ్లేషణ వ్యవస్థ | EfficientNet-B2 | స్క్రీనింగ్ ప్రయోజనాల కోసం మాత్రమే | ఎల్లప్పుడూ నిపుణ వైద్యుడిని సంప్రదించండి",
        "Hindi":   "🫁 फेफड़ों की बीमारी विश्लेषण प्रणाली | EfficientNet-B2 | केवल स्क्रीनिंग उद्देश्यों के लिए | हमेशा प्रमाणित चिकित्सक से परामर्श करें",
    },
}

def t(key, lang):
    """Translate a key to the selected language."""
    return T.get(key, {}).get(lang, T.get(key, {}).get("English", key))

# ──────────────────────────────────────────────
#  CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Telugu:wght@400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Noto Sans Telugu', 'Noto Sans Devanagari', sans-serif;
    background-color: #F4F7FB;
    font-size: 16px;
}
h1,h2,h3,h4 { font-family: 'Poppins', 'Noto Sans Telugu', 'Noto Sans Devanagari', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
/* ── Global base font size ── */
body { font-size: 17px !important; }

/* ── Responsive full-width layout ── */
.block-container {
    padding: 1.5rem 2rem !important;
    max-width: 100% !important;
    width: 100% !important;
}
section.main > div { max-width: 100% !important; }

/* ── Responsive columns ── */
@media (max-width: 768px) {
    .hero { padding: 28px 24px !important; }
    .hero h1 { font-size: 1.5rem !important; }
    .block-container { padding: 1rem !important; }
    .stat-val { font-size: 1.5rem !important; }
}

.hero {
    background: linear-gradient(135deg, #0D1B2A 0%, #1565C0 60%, #0288D1 100%);
    border-radius: 20px; padding: 44px 52px; color: white;
    margin-bottom: 28px; position: relative; overflow: hidden;
}
.hero::after {
    content:''; position:absolute; top:-80px; right:-80px;
    width:280px; height:280px; border-radius:50%;
    background:rgba(255,255,255,0.05);
}
.hero::before {
    content:''; position:absolute; bottom:-60px; left:-40px;
    width:200px; height:200px; border-radius:50%;
    background:rgba(255,255,255,0.03);
}
.hero h1 { font-size:2.8rem !important; font-weight:800 !important; color:white !important; margin:0 0 10px !important; }
.hero p  { color:#B3D4F5; font-size:1.15rem; margin:0 0 20px; line-height:1.65; }

.hero-badge {
    display:inline-block;
    background:rgba(255,255,255,0.15);
    border:1px solid rgba(255,255,255,0.25);
    border-radius:50px;
    padding:6px 18px;
    font-size:1rem;
    font-weight:600;
    color:white;
    margin-bottom:18px;
    backdrop-filter:blur(6px);
}

.hero-chips { display:flex; flex-wrap:wrap; gap:10px; margin-top:4px; }
.chip {
    background:rgba(255,255,255,0.12);
    border:1px solid rgba(255,255,255,0.2);
    border-radius:50px;
    padding:5px 14px;
    font-size:1rem;
    color:white;
    font-weight:500;
}
.chip-green {
    background:rgba(76,175,80,0.25);
    border-color:rgba(76,175,80,0.4);
}

/* ── Language selector (top-right) ── */
#lang-label {
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #1565C0 !important;
}
/* Style only the lang selectbox using its key */
div[data-testid="stSelectbox"]:has(> label) > label {
    font-size: 1rem !important;
    font-weight: 700 !important;
    color: #1565C0 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] {
    cursor: pointer !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child {
    background: white !important;
    border: 2px solid #1565C0 !important;
    border-radius: 50px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #1565C0 !important;
    cursor: pointer !important;
    min-height: 38px !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child:hover {
    background: #E3F2FD !important;
    border-color: #0D47A1 !important;
}


.stat {
    background:white; border-radius:16px; padding:22px 18px;
    text-align:center; border:1px solid #E2ECF6;
    box-shadow:0 2px 10px rgba(0,0,0,0.05);
}
.stat-val { font-family:'Poppins',sans-serif; font-size:2.6rem; font-weight:800; color:#1565C0; line-height:1; margin-bottom:6px; }
.stat-lbl { color:#78909C; font-size:1rem; text-transform:uppercase; letter-spacing:0.6px; font-weight:600; }

.sec-title {
    font-family:'Poppins',sans-serif; font-size:1.6rem; font-weight:700;
    color:#1E3A5F; border-left:5px solid #1565C0;
    padding-left:14px; margin:32px 0 18px;
}

.d-card {
    background:white; border-radius:16px; padding:22px;
    border:1px solid #E2ECF6; border-top:5px solid var(--c,#1565C0);
    box-shadow:0 2px 10px rgba(0,0,0,0.05);
}
.d-card h4 { font-family:'Poppins',sans-serif; font-size:1.2rem; font-weight:700; color:var(--c,#1565C0); margin:0 0 10px; }
.d-card p  { color:#546E7A; font-size:1.05rem; line-height:1.6; margin:0 0 10px; }
.d-card ul { padding-left:18px; margin:0; }
.d-card li { color:#607D8B; font-size:1.05rem; line-height:1.75; }

.res-block {
    border-radius:14px; padding:22px 26px;
    border:2px solid var(--bc,#90CAF9); background:var(--bg,#E3F2FD);
    margin-bottom:16px;
}
.res-block h3 { font-family:'Poppins',sans-serif; color:var(--tc,#1565C0); font-size:2rem; margin:0 0 6px; }
.res-block p  { color:#546E7A; margin:0; font-size:1.1rem; }

.sev { display:inline-block; padding:7px 20px; border-radius:50px; font-weight:700; font-size:1rem; }
.sev-H { background:#FFEBEE; color:#C62828; border:1.5px solid #EF9A9A; }
.sev-M { background:#FFF8E1; color:#F57F17; border:1.5px solid #FFE082; }
.sev-L { background:#E8F5E9; color:#2E7D32; border:1.5px solid #A5D6A7; }
.sev-N { background:#E3F2FD; color:#1565C0; border:1.5px solid #90CAF9; }

.rec {
    background:#F8FBFF; border-left:4px solid #1565C0;
    border-radius:0 10px 10px 0; padding:11px 16px; margin:7px 0;
    font-size:1.05rem; color:#263238; line-height:1.7;
}

.ai-note {
    background:#EEF4FF; border:1.5px solid #90CAF9; border-radius:12px;
    padding:14px 20px; font-size:1.05rem; color:#1A237E; margin-bottom:20px;
}

.lang-bar {
    display:flex; align-items:center; justify-content:flex-end;
    gap:12px; margin-bottom:16px;
}

.footer {
    text-align:center; color:#90A4AE; font-size:1rem;
    margin-top:48px; padding-top:18px; border-top:1px solid #E0E0E0;
}

div[data-testid="stButton"] > button { border-radius:12px !important; font-weight:600 !important; }
.stFileUploader > div {
    border-radius:14px !important;
    border:2px dashed #90CAF9 !important;
    background:#F8FBFF !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  MODEL
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    def build(n):
        m = models.efficientnet_b2(weights=None)
        inf = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.BatchNorm1d(inf), nn.Dropout(0.5),
            nn.Linear(inf,256), nn.SiLU(),
            nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256,n),
        )
        return m
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()
    ck = torch.load(MODEL_PATH, map_location="cpu",weights_only=False)
    thresholds=[0.45,0.40,0.15,0.40]
    m  = build(len(ck["classes"]))
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, ck["classes"], thresholds, ck["image_size"]

def predict(img):
    model, classes, thres, sz = load_model()
    tf = transforms.Compose([
        transforms.Resize((sz,sz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tensor = tf(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).squeeze().numpy()
    prob_dict = {c: float(p) for c,p in zip(classes,probs)}
    detected  = sorted(
        [(c,float(p)) for c,p,th in zip(classes,probs,thres) if p>=th],
        key=lambda x: x[1], reverse=True
    )
    if not detected:
        top = int(np.argmax(probs))
        detected = [(classes[top], float(probs[top]))]

    # If Normal + other diseases detected together, remove Normal
    if len(detected) > 1:
        filtered = [(c, p) for c, p in detected if c != "Normal"]
        if filtered:
            detected = filtered

    return detected

# ──────────────────────────────────────────────
#  SEVERITY & RECOMMENDATIONS
# ──────────────────────────────────────────────
def get_severity(conf, disease):
    if disease == "Normal": return "None"
    if conf >= 0.85: return "High"
    if conf >= 0.60: return "Medium"
    return "Low"

RECS = {
    "Tuberculosis": {
        "English": {
            "High":   ["Begin anti-TB DOTS therapy under physician supervision immediately.","Isolate from close contacts to prevent airborne transmission.","Monthly sputum cultures to monitor treatment response.","High-protein diet: eggs, chicken, legumes to support recovery.","Complete the full 6-month course without interruption."],
            "Medium": ["Consult a pulmonologist for TB culture and sensitivity testing.","Begin standard therapy once lab confirmation is received.","Avoid crowded enclosed spaces to reduce transmission risk.","Monitor weight weekly and report significant loss.","Follow-up chest X-ray after 2 months of treatment."],
            "Low":    ["Confirm with sputum test and Mantoux tuberculin skin test.","Maintain good home ventilation and adequate rest.","Watch for persistent cough, night sweats, and weight loss.","Eat a balanced diet rich in vitamins A, C, and E.","Schedule follow-up chest X-ray in 4 to 6 weeks."],
        },
        "Telugu": {
            "High":   ["వైద్యుని పర్యవేక్షణలో వెంటనే యాంటీ-TB DOTS చికిత్స ప్రారంభించండి.","గాలి ద్వారా వ్యాపించడాన్ని నిరోధించడానికి దగ్గరి వ్యక్తుల నుండి వేరుపడండి.","చికిత్స ప్రతిస్పందనను పర్యవేక్షించడానికి నెలవారీ స్పుటం కల్చర్లు.","కోలుకోవడానికి మద్దతుగా అధిక-ప్రోటీన్ ఆహారం తీసుకోండి.","అంతరాయం లేకుండా పూర్తి 6-నెలల కోర్సు పూర్తి చేయండి."],
            "Medium": ["TB కల్చర్ మరియు సెన్సిటివిటీ పరీక్షల కోసం పల్మోనాలజిస్ట్‌ను సంప్రదించండి.","ల్యాబ్ నిర్ధారణ అందిన తర్వాత ప్రమాణ చికిత్స ప్రారంభించండి.","వ్యాప్తి ప్రమాదాన్ని తగ్గించడానికి రద్దీగా ఉండే మూసిన ప్రదేశాలను నివారించండి.","వారానికొకసారి బరువు పర్యవేక్షించండి.","2 నెలల చికిత్స తర్వాత ఫాలో-అప్ ఛాతీ X-రే తీయించుకోండి."],
            "Low":    ["స్పుటం పరీక్ష మరియు మాంటౌక్స్ చర్మ పరీక్షతో నిర్ధారించుకోండి.","ఇంటిలో మంచి వెంటిలేషన్ మరియు తగినంత విశ్రాంతి పాటించండి.","నిరంతర దగ్గు, రాత్రి చెమటలు మరియు బరువు తగ్గడాన్ని గమనించండి.","విటమిన్ A, C మరియు E తో సమతుల్య ఆహారం తీసుకోండి.","4 నుండి 6 వారాలలో ఫాలో-అప్ ఛాతీ X-రే షెడ్యూల్ చేయండి."],
        },
        "Hindi": {
            "High":   ["चिकित्सक की देखरेख में तुरंत एंटी-TB DOTS थेरेपी शुरू करें।","हवाई संचरण को रोकने के लिए करीबी संपर्कों से अलग रहें।","उपचार प्रतिक्रिया की निगरानी के लिए मासिक थूक संस्कृतियाँ।","रिकवरी के लिए उच्च-प्रोटीन आहार लें।","बिना रुकावट पूरे 6 महीने का कोर्स पूरा करें।"],
            "Medium": ["TB कल्चर परीक्षण के लिए फुफ्फुस विशेषज्ञ से परामर्श करें।","लैब पुष्टि मिलने के बाद मानक चिकित्सा शुरू करें।","भीड़भाड़ वाले बंद स्थानों से बचें।","साप्ताहिक वजन की निगरानी करें।","2 महीने के उपचार के बाद फॉलो-अप छाती X-रे करवाएं।"],
            "Low":    ["थूक परीक्षण और मंटौक्स त्वचा परीक्षण से पुष्टि करें।","घर में अच्छा वेंटिलेशन और पर्याप्त आराम रखें।","लगातार खांसी, रात को पसीना और वजन घटने पर ध्यान दें।","विटामिन A, C और E से भरपूर संतुलित आहार लें।","4 से 6 सप्ताह में फॉलो-अप छाती X-रे शेड्यूल करें।"],
        },
    },
    "Pneumonia": {
        "English": {
            "High":   ["Hospitalization may be required — consult emergency services.","IV antibiotics and oxygen therapy under physician supervision.","Maintain SpO2 above 95 percent with supplemental oxygen.","Chest physiotherapy to assist mucus clearance.","Monitor vitals every 4 hours."],
            "Medium": ["Start oral antibiotics as prescribed by your physician.","Complete bed rest until fever fully subsides.","Drink 8 to 10 glasses of warm water or herbal tea daily.","Use a room humidifier to ease breathing.","Return to hospital immediately if breathlessness worsens."],
            "Low":    ["Rest at home and complete the full antibiotic course.","Stay well-hydrated with warm fluids throughout the day.","Avoid smoking and second-hand smoke completely.","Sleep with head elevated to ease breathing.","Schedule a follow-up appointment in 5 to 7 days."],
        },
        "Telugu": {
            "High":   ["ఆసుపత్రిలో చేరడం అవసరం కావచ్చు — అత్యవసర సేవలను సంప్రదించండి.","వైద్యుని పర్యవేక్షణలో IV యాంటీబయాటిక్స్ మరియు ఆక్సిజన్ చికిత్స.","అదనపు ఆక్సిజన్‌తో SpO2 95 శాతానికి పైగా నిర్వహించండి.","శ్లేష్మాన్ని తొలగించడానికి ఛాతీ ఫిజియోథెరపీ సెషన్లు.","ప్రతి 4 గంటలకు వైటల్స్ పర్యవేక్షించండి."],
            "Medium": ["మీ వైద్యుని ప్రిస్క్రిప్షన్ ప్రకారం నోటి యాంటీబయాటిక్స్ ప్రారంభించండి.","జ్వరం పూర్తిగా తగ్గే వరకు పూర్తి విశ్రాంతి తీసుకోండి.","రోజూ 8 నుండి 10 గ్లాసుల వేడి నీళ్ళు లేదా మూలిక చాయ్ తాగండి.","శ్వాస సులభంగా తీసుకోవడానికి రూమ్ హ్యూమిడిఫయర్ వాడండి.","శ్వాస ఆడకపోవడం మరింత దిగజారితే వెంటనే ఆసుపత్రికి వెళ్ళండి."],
            "Low":    ["ఇంట్లో విశ్రాంతి తీసుకోండి మరియు పూర్తి యాంటీబయాటిక్ కోర్సు పూర్తి చేయండి.","రోజంతా వేడి ద్రవాలు తాగి చక్కగా హైడ్రేట్‌గా ఉండండి.","ధూమపానం మరియు పాసివ్ స్మోకింగ్ పూర్తిగా నివారించండి.","శ్వాస సులభంగా తీసుకోవడానికి తల ఎత్తుగా ఉండేలా పడుకోండి.","5 నుండి 7 రోజులలో ఫాలో-అప్ అపాయింట్‌మెంట్ తీసుకోండి."],
        },
        "Hindi": {
            "High":   ["अस्पताल में भर्ती होना पड़ सकता है — आपातकालीन सेवाओं से परामर्श करें।","चिकित्सक की देखरेख में IV एंटीबायोटिक्स और ऑक्सीजन थेरेपी।","पूरक ऑक्सीजन से SpO2 95 प्रतिशत से ऊपर बनाए रखें।","बलगम निकालने में मदद के लिए छाती की फिजियोथेरेपी।","हर 4 घंटे में महत्वपूर्ण संकेतों की निगरानी करें।"],
            "Medium": ["चिकित्सक द्वारा निर्धारित मौखिक एंटीबायोटिक्स शुरू करें।","बुखार पूरी तरह उतरने तक पूर्ण बिस्तर आराम करें।","रोजाना 8 से 10 गिलास गर्म पानी या हर्बल चाय पिएं।","सांस लेने में राहत के लिए रूम ह्यूमिडिफायर का उपयोग करें।","सांस की तकलीफ बढ़ने पर तुरंत अस्पताल जाएं।"],
            "Low":    ["घर पर आराम करें और पूरा एंटीबायोटिक कोर्स पूरा करें।","पूरे दिन गर्म तरल पदार्थ पीकर अच्छी तरह हाइड्रेटेड रहें।","धूम्रपान और पैसिव स्मोकिंग से पूरी तरह बचें।","सांस लेने में राहत के लिए सिर ऊंचा करके सोएं।","5 से 7 दिनों में फॉलो-अप अपॉइंटमेंट शेड्यूल करें।"],
        },
    },
    "Pneumothorax": {
        "English": {
            "High":   ["Emergency chest tube insertion required — seek immediate medical attention.","Do NOT travel by air until fully resolved.","Strict bed rest — avoid all physical exertion.","Continuous pulse oximetry monitoring in hospital.","Serial chest X-rays every 6 hours to monitor re-expansion."],
            "Medium": ["Urgent X-ray confirmation and specialist review needed.","Needle aspiration may be performed if symptomatic.","Avoid all strenuous activity and heavy lifting.","Report any worsening shortness of breath to a physician.","Avoid high altitudes until cleared by a chest specialist."],
            "Low":    ["Observation and rest — small pneumothorax may resolve on its own.","Supplemental oxygen to speed up nitrogen reabsorption.","Avoid activities that increase thoracic pressure.","Repeat X-ray in 24 to 48 hours to confirm stability.","Stop smoking immediately to reduce recurrence risk."],
        },
        "Telugu": {
            "High":   ["అత్యవసర ఛాతీ ట్యూబ్ చొప్పించడం అవసరం — తక్షణ వైద్య సహాయం కోసం వెళ్ళండి.","పూర్తిగా పరిష్కారమయ్యే వరకు విమానంలో ప్రయాణించవద్దు.","కఠిన విశ్రాంతి — అన్ని శారీరక శ్రమను నివారించండి.","ఆసుపత్రిలో నిరంతర పల్స్ ఆక్సిమెట్రీ పర్యవేక్షణ.","పున:వ్యాకోచాన్ని పర్యవేక్షించడానికి ప్రతి 6 గంటలకు ఛాతీ X-రే."],
            "Medium": ["అత్యవసర X-రే నిర్ధారణ మరియు నిపుణుల సమీక్ష అవసరం.","లక్షణాలు ఉంటే నీడిల్ ఆస్పిరేషన్ చేయవచ్చు.","అన్ని శ్రమతో కూడిన కార్యకలాపాలు మరియు భారమైన వస్తువులు మోయడం నివారించండి.","శ్వాస ఆడకపోవడం మరింత దిగజారితే వైద్యుడికి చెప్పండి.","ఛాతీ నిపుణుడు అనుమతించే వరకు అధిక ఎత్తులను నివారించండి."],
            "Low":    ["పరిశీలన మరియు విశ్రాంతి — చిన్న న్యుమోత్రాక్స్ దానంతట అదే మెరుగవుతుంది.","నైట్రోజన్ పునశ్శోషణాన్ని వేగవంతం చేయడానికి అదనపు ఆక్సిజన్.","ఛాతీ ఒత్తిడిని పెంచే కార్యకలాపాలు నివారించండి.","స్థిరత్వాన్ని నిర్ధారించడానికి 24 నుండి 48 గంటలలో పునరావృత X-రే.","పునరావృతం ప్రమాదాన్ని తగ్గించడానికి వెంటనే ధూమపానం మానుకోండి."],
        },
        "Hindi": {
            "High":   ["आपातकालीन छाती ट्यूब डालना आवश्यक — तत्काल चिकित्सा सहायता लें।","पूरी तरह ठीक होने तक हवाई यात्रा न करें।","सख्त बिस्तर आराम — सभी शारीरिक परिश्रम से बचें।","अस्पताल में निरंतर पल्स ऑक्सीमेट्री निगरानी।","पुनः विस्तार की निगरानी के लिए हर 6 घंटे में छाती X-रे।"],
            "Medium": ["तत्काल X-रे पुष्टि और विशेषज्ञ समीक्षा आवश्यक।","लक्षण होने पर सुई एस्पिरेशन किया जा सकता है।","सभी कठिन गतिविधियों और भारी सामान उठाने से बचें।","सांस की तकलीफ बढ़ने पर चिकित्सक को बताएं।","छाती विशेषज्ञ की अनुमति तक ऊंचाई से बचें।"],
            "Low":    ["निगरानी और आराम — छोटा न्यूमोथोरैक्स अपने आप ठीक हो सकता है।","नाइट्रोजन पुनः अवशोषण को तेज करने के लिए पूरक ऑक्सीजन।","वक्षीय दबाव बढ़ाने वाली गतिविधियों से बचें।","स्थिरता की पुष्टि के लिए 24 से 48 घंटों में पुनः X-रे।","पुनरावृत्ति जोखिम को कम करने के लिए तुरंत धूम्रपान बंद करें।"],
        },
    },
    "Normal": {
        "English": {"None": ["No lung disease detected — maintain your healthy lifestyle.","Regular aerobic exercise: 30 minutes, 5 days per week.","Avoid smoking and prolonged exposure to air pollutants.","Annual chest X-ray if you have occupational risk factors.","Balanced antioxidant-rich diet for long-term lung health."]},
        "Telugu":  {"None": ["ఊపిరితిత్తుల వ్యాధి గుర్తించబడలేదు — ఆరోగ్యకరమైన జీవనశైలిని కొనసాగించండి.","వారంలో 5 రోజులు, 30 నిమిషాల సాధారణ వ్యాయామం చేయండి.","ధూమపానం మరియు వాయు కాలుష్యానికి దీర్ఘకాలిక గురికావడం నివారించండి.","వృత్తిపరమైన ప్రమాద కారకాలు ఉంటే వార్షిక ఛాతీ X-రే తీయించుకోండి.","దీర్ఘకాలిక ఊపిరితిత్తుల ఆరోగ్యం కోసం యాంటీఆక్సిడెంట్ సమృద్ధమైన సమతుల్య ఆహారం."]},
        "Hindi":   {"None": ["कोई फेफड़ों की बीमारी नहीं पाई गई — अपनी स्वस्थ जीवनशैली बनाए रखें।","नियमित एरोबिक व्यायाम: सप्ताह में 5 दिन, 30 मिनट।","धूम्रपान और वायु प्रदूषकों के लंबे समय तक संपर्क से बचें।","व्यावसायिक जोखिम कारक होने पर वार्षिक छाती X-रे।","दीर्घकालिक फेफड़ों के स्वास्थ्य के लिए एंटीऑक्सीडेंट युक्त संतुलित आहार।"]},
    },
}

def get_recs_ml(disease, sev, lang):
    fallback = {
        "English": ["Consult your physician for a detailed clinical evaluation.","Follow up with appropriate diagnostic tests.","Maintain a healthy lifestyle and balanced diet.","Avoid smoking and pollutant exposure.","Schedule regular health checkups."],
        "Telugu":  ["వివరణాత్మక క్లినికల్ అంచనా కోసం మీ వైద్యుడిని సంప్రదించండి.","తగిన రోగనిర్ధారణ పరీక్షలతో ఫాలో-అప్ చేయండి.","ఆరోగ్యకరమైన జీవనశైలిని నిర్వహించండి.","ధూమపానం మరియు కాలుష్య గురికావడం నివారించండి.","క్రమం తప్పకుండా ఆరోగ్య పరీక్షలు నిర్వహించండి."],
        "Hindi":   ["विस्तृत नैदानिक मूल्यांकन के लिए अपने चिकित्सक से परामर्श करें।","उचित नैदानिक परीक्षणों के साथ फॉलो-अप करें।","स्वस्थ जीवनशैली और संतुलित आहार बनाए रखें।","धूम्रपान और प्रदूषण से बचें।","नियमित स्वास्थ्य जांच कराएं।"],
    }
    return RECS.get(disease,{}).get(lang,{}).get(sev, fallback.get(lang, fallback["English"]))

SEV_BADGE_HTML = {
    "High":   lambda l: f"<span class='sev sev-H'>{t('sev_high',l)}</span>",
    "Medium": lambda l: f"<span class='sev sev-M'>{t('sev_medium',l)}</span>",
    "Low":    lambda l: f"<span class='sev sev-L'>{t('sev_low',l)}</span>",
    "None":   lambda l: f"<span class='sev sev-N'>{t('sev_none',l)}</span>",
}

RES_STYLE = {
    "High":   ("#FFEBEE","#EF9A9A","#C62828"),
    "Medium": ("#FFF8E1","#FFE082","#F57F17"),
    "Low":    ("#E8F5E9","#A5D6A7","#2E7D32"),
    "None":   ("#E3F2FD","#90CAF9","#1565C0"),
}

DC = {
    "Tuberculosis":"#E53935",
    "Pneumonia":   "#1E88E5",
    "Pneumothorax":"#00897B",
    "Normal":      "#1565C0",
}

# ──────────────────────────────────────────────
#  MAIN APP
# ──────────────────────────────────────────────
def main():
    load_model()

    # ── LANGUAGE SELECTOR — top-right ──
    _, col_lang = st.columns([6, 2])
    with col_lang:
        lang = st.selectbox(
            "🌐 Language",
            options=["English", "Telugu", "Hindi"],
            key="lang",
        )

    # ── HERO ──
    st.markdown(f"""
    <div class="hero">
        <h1>🫁 {t("title", lang)}</h1>
        <p>{t("hero_sub", lang)}</p>
        <div class="hero-chips">
            <span class="chip">🦠 Tuberculosis</span>
            <span class="chip">🫧 Pneumonia</span>
            <span class="chip">💨 Pneumothorax</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Note ──
    st.markdown(f'<div class="ai-note">{t("ai_note", lang)}</div>', unsafe_allow_html=True)

    # ════════════════════════════
    #  1 — MODEL OVERVIEW
    # ════════════════════════════
    st.markdown(f'<div class="sec-title">{t("overview_title", lang)}</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, val, lbl in [
        (c1, "88.93%",        t("accuracy_label", lang))
    ]:
        with col:
            st.markdown(f"""
            <div class="stat">
                <div class="stat-val">{val}</div>
                <div class="stat-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════
    #  2 — ANALYSIS
    # ════════════════════════════
    st.markdown(f'<div class="sec-title">{t("analysis_title", lang)}</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        t("upload_label", lang),
        type=["jpg","jpeg","png","bmp"],
        help=t("upload_help", lang),
    )

    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")

        ci, cm = st.columns([1,1])
        with ci:
            st.image(img_pil, use_container_width=True)
        with cm:
            st.markdown(f"""
            <div style="background:#F8FBFF; border-radius:12px; padding:18px;
                 border:1px solid #E3F2FD; margin-top:8px">
                <p style="margin:0 0 8px"><b>{t("file_label",lang)}:</b> {uploaded.name}</p>
                <p style="margin:0 0 8px"><b>{t("size_label",lang)}:</b> {uploaded.size/1024:.1f} KB</p>
                <p style="margin:0 0 8px"><b>{t("dim_label",lang)}:</b> {img_pil.width} x {img_pil.height} px</p>
                <p style="margin:0; color:#2E7D32; font-weight:600">
                    {t("status_label",lang)}: {t("ready_label",lang)}
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(t("analyze_btn", lang), type="primary", use_container_width=True):
            with st.spinner(t("spinner_msg", lang)):
                detected = predict(img_pil)

            st.markdown("---")
            det_names = [d for d,_ in detected]

            # ── Result header ──
            if len(detected) > 1:
                diseases_str = "  +  ".join(det_names)
                st.markdown(f"""
                <div style="background:#FFF3E0; border:2px solid #FFCC80;
                     border-radius:16px; padding:22px 28px; margin-bottom:20px">
                    <p style="color:#E65100; font-size:0.8rem; font-weight:700;
                       text-transform:uppercase; letter-spacing:1px; margin:0 0 4px">
                        {t("multi_detected", lang)}
                    </p>
                    <h2 style="font-family:Poppins,sans-serif; color:#BF360C;
                        margin:0 0 8px; font-size:1.7rem">{diseases_str}</h2>
                    <p style="color:#6D4C41; margin:0; font-size:0.88rem">
                        {t("multi_sub", lang)}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                disease, conf = detected[0]
                sev = get_severity(conf, disease)
                bg, bc, tc = RES_STYLE[sev]
                st.markdown(f"""
                <div class="res-block" style="--bg:{bg}; --bc:{bc}; --tc:{tc}">
                    <p style="color:#90A4AE; font-size:0.8rem; margin:0 0 4px;
                       text-transform:uppercase; letter-spacing:0.8px">
                       {t("primary_diag", lang)}
                    </p>
                    <h3>{disease}</h3>
                    <p>
                        {t("severity_label", lang)}: <strong style="color:{tc}">{t(f"sev_{sev[0].lower() if sev!='None' else 'none'}", lang)}</strong>
                        &nbsp;&nbsp;{SEV_BADGE_HTML[sev](lang)}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # ── Per-disease details ──
            for disease, conf in detected:
                sev  = get_severity(conf, disease)
                recs = get_recs_ml(disease, sev, lang)
                dc   = DC.get(disease, "#1565C0")
                bg, bc, tc = RES_STYLE[sev]

                if len(detected) > 1:
                    st.markdown(f"""
                    <div style="border-left:5px solid {dc}; background:{bg};
                         border-radius:0 12px 12px 0; padding:14px 20px; margin:12px 0 8px">
                        <b style="color:{dc}; font-family:Poppins,sans-serif; font-size:1rem">
                            {disease}
                        </b>
                        &nbsp;&nbsp;{SEV_BADGE_HTML[sev](lang)}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"**{t('recs_for', lang)} {disease}:**")
                for i, rec in enumerate(recs, 1):
                    st.markdown(f'<div class="rec"><b>{i}.</b> {rec}</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            # ── Disclaimer ──
            st.markdown(f"""
            <div style="background:#FFF8E1; border:1.5px solid #FFD54F;
                 border-radius:12px; padding:14px 18px; font-size:0.84rem;
                 color:#5D4037; line-height:1.65; margin-top:8px">
                {t("disclaimer", lang)}
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════
    #  3 — ABOUT DISEASES
    # ════════════════════════════
    st.markdown(f'<div class="sec-title">{t("about_title", lang)}</div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)

    for col, name_key, desc_key, sym_key, color in [
        (d1, "tb_name", "tb_desc", "tb_symptoms", "#E53935"),
        (d2, "pn_name", "pn_desc", "pn_symptoms", "#1E88E5"),
        (d3, "pt_name", "pt_desc", "pt_symptoms", "#00897B"),
    ]:
        with col:
            symptoms = t(sym_key, lang)
            sym_html = "".join(f"<li>{s}</li>" for s in symptoms)
            st.markdown(f"""
            <div class="d-card" style="--c:{color}">
                <h4>{t(name_key, lang)}</h4>
                <p>{t(desc_key, lang)}</p>
                <ul>{sym_html}</ul>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── Footer ──
    st.markdown(f'<div class="footer">{t("footer", lang)}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
