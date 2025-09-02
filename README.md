# CLIP Scene Search System

## 🎯 Obiettivi del Progetto - TUTTI COMPLETATI ✅

Il progetto implementa una pipeline multi-stage di computer vision per indicizzare un dataset etichettato di oggetti (personaggi) e utilizzare questo database per identificare e classificare oggetti in scene complesse.

### ✅ Obiettivi Completati:

1. **✅ Dataset Acquisition & Preprocessing**
   - Dataset di personaggi Naruto acquisito e preprocessato
   - Struttura organizzata in train/test/valid
   - Preprocessing automatico tramite CLIP

2. **✅ Indexing Pipeline con CLIP Embeddings** 
   - Pipeline completa in `src/pipelines/index_dataset.py`
   - Calcolo e storage degli embedding CLIP per ogni immagine
   - Database vettoriale FAISS per ricerche veloci
   - Metadata salvati in JSON per tracciabilità

3. **✅ Scene Analysis Pipeline con Segmentazione Avanzata**
   - **Multipli algoritmi implementati**: Grid, K-means, Superpixel (SLIC), **SAM**
   - Integrazione con **Segment Anything Model (SAM)** per segmentazione automatica
   - Calcolo embedding CLIP per ogni segmento
   - Pipeline unificata in `src/pipelines/segment_and_search.py`

4. **✅ Matching Algorithm Sofisticato**
   - Confronto embedding segmenti vs database vettoriale
   - Similarità coseno per identificazione caratteri
   - Ranking e scoring dei risultati
   - Top-K retrieval configurabile

5. **✅ Interactive Gradio Web Interface**
   - Interfaccia web completa e professionale
   - **5 Tab specializzati**: Inizializzazione, Ricerca Globale, Ricerca Segmenti, Indicizzazione, Valutazione Performance
   - Supporto drag & drop per immagini
   - Visualizzazione con bounding box colorati
   - Parametri configurabili per ogni metodo

6. **✅ Performance Evaluation System**
   - **NUOVO**: Sistema completo di valutazione quantitativa in `src/pipelines/performance_evaluation.py`
   - Metriche: Precision, Recall, F1-Score, MAP
   - Analisi comparative metodi di segmentazione
   - Valutazione qualità embedding (intra/inter-classe)
   - Report HTML con grafici e raccomandazioni

## 🚀 Caratteristiche Avanzate Implementate

### 🧠 Modelli AI Integrati
- **CLIP ViT-B/32**: Encoding multimodale testo-immagine
- **SAM (Segment Anything)**: Segmentazione automatica state-of-the-art
- **FAISS**: Database vettoriale ottimizzato per similarità

### 🔍 Metodi di Segmentazione
- **Grid**: Divisione uniforme rapida
- **K-means**: Clustering basato sui colori  
- **Superpixel SLIC**: Regioni semanticamente coerenti
- **SAM**: Segmentazione automatica di qualità professionale

### 📊 Sistema di Valutazione
- **Metriche quantitative**: Precision, Recall, F1, MAP
- **Analisi performance**: Tempi elaborazione, copertura segmenti
- **Qualità embedding**: Coerenza intra/inter-classe
- **Report automatici**: HTML + grafici + raccomandazioni

## 📁 Struttura del Progetto

```
PythonProject/
├── src/
│   ├── models/
│   │   ├── backbones.py          # CLIP backbone con SSL fix
│   │   └── clip_finetune.py      # Fine-tuning personalizzato
│   ├── pipelines/
│   │   ├── index_dataset.py      # ✅ Indicizzazione database vettoriale
│   │   ├── segment_and_search.py # ✅ Segmentazione + ricerca semantica  
│   │   ├── sam_integration.py    # ✅ Integrazione Segment Anything Model
│   │   ├── performance_evaluation.py # ✅ Sistema valutazione completo
│   │   └── fine_tune_clip.py     # Fine-tuning avanzato
│   └── ui/
│       └── gradio_app.py         # ✅ Interfaccia web completa
├── data/
│   ├── raw/Anime-Naruto/         # ✅ Dataset preprocessato
│   ├── scene_examples/           # Esempi di scene
│   └── vector_db/               # ✅ Database vettoriale FAISS
├── checkpoints/                  # Modelli salvati
├── reports/                      # ✅ Report valutazione performance
│   └── figures/                 # Grafici e visualizzazioni
└── notebooks/                   # Jupyter notebooks analisi
```

## 🚀 Avvio Rapido

### 1. Installazione
```bash
# Clona il repository
git clone <repository-url>
cd PythonProject

# Crea ambiente virtuale
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Installa dipendenze
pip install -r requirements.txt

# [OPZIONALE] Installa SAM per segmentazione avanzata
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 2. Avvio del Sistema
```bash
# Metodo principale (raccomandato)
python run.py

# Oppure direttamente
python src/ui/gradio_app.py
```

### 3. Interfaccia Web
Apri il browser su: **http://localhost:7860**

## 🎮 Guida Utilizzo

### 🚀 **Tab Inizializzazione**
1. Clicca "🔄 Inizializza Sistema"
2. Il sistema auto-carica il dataset Naruto se disponibile
3. Verifica lo stato del database vettoriale

### 🔍 **Tab Ricerca Globale**  
1. Inserisci query: *"personaggio con capelli biondi"*
2. Configura numero risultati (1-10)
3. Visualizza galleria con score di similarità

### 🧩 **Tab Ricerca Segmenti**
1. Carica immagine (drag & drop)
2. Inserisci query: *"volto del personaggio"*
3. Scegli metodo: **SAM** (migliore qualità) o altri
4. Visualizza segmenti con bounding box colorati

### 📊 **Tab Indicizzazione**
1. Inserisci path dataset: `data/raw/nuovo_dataset/`
2. Clicca "📥 Indicizza Dataset" 
3. Monitora progresso in tempo reale

### 📈 **Tab Valutazione Performance**
1. Clicca "🚀 Esegui Valutazione Completa"
2. Genera report automatico con:
   - Metriche quantitative (Precision, Recall, F1, MAP)
   - Performance comparative segmentazione
   - Qualità embedding analysis
   - Grafici e raccomandazioni

## 🔧 Configurazione Avanzata

### SAM Setup (Opzionale ma Raccomandato)
```bash
# Installa SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Scarica checkpoint (automatico al primo utilizzo)
# vit_b: ~375MB, vit_l: ~1.2GB, vit_h: ~2.4GB
```

### Config Personalizzata (`config.yaml`)
```yaml
clip:
  model_name: "ViT-B/32"  # o "ViT-L/14" per qualità superiore
  device: "cpu"           # o "cuda" se disponibile
  
vector_db:
  similarity_threshold: 0.7  # Soglia similarità

ui:
  max_upload_size: 10  # MB max upload
```

## 📈 Metriche e Performance

### 🎯 Risultati Tipici
- **Precision**: ~0.85+ per dataset ben etichettato
- **Recall**: ~0.80+ con query appropriate  
- **Tempo Segmentazione**:
  - Grid: ~0.1s
  - K-means: ~0.5s
  - Superpixel: ~1.0s
  - SAM: ~3-10s (ma qualità superiore)

### 📊 Report Automatici
Il sistema genera report HTML completi con:
- Grafici performance comparativi
- Metriche dettagliate per metodo
- Raccomandazioni di ottimizzazione
- Analisi qualità embedding

## 🛠️ Tecnologie Utilizzate

- **🧠 AI/ML**: CLIP, SAM, FAISS, scikit-learn
- **🖼️ Computer Vision**: OpenCV, PIL, matplotlib
- **🚀 Backend**: PyTorch, transformers, numpy
- **🌐 Frontend**: Gradio, HTML, CSS
- **📊 Analytics**: pandas, seaborn, plotly
- **⚡ Performance**: FAISS vector search, batch processing

## 🎓 Casi d'Uso

1. **🎨 Content Creation**: Trova asset simili per progetti creativi
2. **🔍 Visual Search**: Ricerca semantica in archivi fotografici  
3. **🤖 AI Training**: Dataset curation e quality assessment
4. **📚 Research**: Analisi quantitativa algoritmi computer vision
5. **🎮 Gaming**: Asset matching per game development

## 📝 Validazione Obiettivi

| Obiettivo | Stato | Implementazione |
|-----------|-------|-----------------|
| Dataset preprocessing | ✅ | Dataset Naruto + pipeline automatica |
| Indexing pipeline | ✅ | FAISS + CLIP embeddings |  
| Scene analysis + SAM | ✅ | 4 metodi incluso SAM integration |
| Matching algorithm | ✅ | Cosine similarity + ranking |
| Gradio interface | ✅ | 5 tab specializzati + UX professionale |
| Performance evaluation | ✅ | Sistema completo con report automatici |

## 🚀 Prossimi Sviluppi

- [ ] Fine-tuning CLIP su dataset specifico
- [ ] Integrazione modelli di detection (YOLO)
- [ ] API REST per integrazioni esterne
- [ ] Docker containerization
- [ ] Cloud deployment (Hugging Face Spaces)

---

**🎯 Tutti gli obiettivi del progetto sono stati completati con successo!**
