# CLIP Scene Search System

## Descrizione
Sistema avanzato per il fine-tuning di CLIP e la ricerca semantica di scene nelle immagini utilizzando un'interfaccia web interattiva Gradio.

## Struttura del Progetto
```
data/
  raw/              # Dati grezzi
  scene_examples/   # Esempi di scene
  vector_db/        # Database vettoriale
src/
  models/           # Modelli di machine learning
  pipelines/        # Pipeline di elaborazione
  ui/               # Interfaccia utente (Gradio)
checkpoints/        # Checkpoint del modello
notebooks/          # Jupyter notebooks
reports/figures/    # Report e figure
```

## Installazione

### 1. Prerequisiti
- Python 3.8 o superiore
- pip (gestore pacchetti Python)
- Git (opzionale, per clonare il repository)

### 2. Setup dell'ambiente virtuale (raccomandato)
```bash
# Crea un ambiente virtuale
python -m venv .venv

# Attiva l'ambiente virtuale
# Su macOS/Linux:
source .venv/bin/activate
# Su Windows:
.venv\Scripts\activate
```

### 3. Installazione delle dipendenze
```bash
pip install -r requirements.txt
```

### 4. Configurazione iniziale
Il file `config.yaml` contiene tutte le impostazioni del progetto. Le configurazioni di default dovrebbero funzionare per la maggior parte dei casi.

## 🚀 Avvio Rapido

### Metodo 1: Script Principale
```bash
python run.py
```

### Metodo 2: Script Esteso
```bash
python run_gradio.py
```

### Metodo 3: Direttamente
```bash
python src/ui/gradio_app.py
```

L'interfaccia sarà disponibile su:
- **Locale**: `http://localhost:7860`
- **Rete locale**: `http://[tuo-ip]:7860`

## 🎯 Interfaccia Web Gradio

Il sistema utilizza **Gradio** per fornire un'interfaccia demo interattiva e professionale con le seguenti funzionalità:

### 🚀 **Tab Inizializzazione**
- Inizializzazione del sistema CLIP
- Verifica stato del database vettoriale
- Controllo configurazione

### 🔍 **Tab Ricerca Globale**
- Ricerca semantica nel dataset completo
- Input query in linguaggio naturale
- Galleria risultati con score di similarità
- Configurazione numero risultati (1-10)

### 🧩 **Tab Ricerca Segmenti**
- Upload immagini con drag & drop
- Tre metodi di segmentazione:
  - **Grid**: Divisione uniforme in griglia
  - **K-means**: Clustering basato sui colori
  - **Superpixel**: Regioni semanticamente omogenee
- Visualizzazione con bounding box colorati
- Parametri configurabili per ogni metodo

### 📊 **Tab Indicizzazione**
- Indicizzazione automatica del dataset
- Input percorso directory immagini
- Feedback in tempo reale del processo
- Supporto per formati: JPG, JPEG, PNG, BMP, TIFF

### ℹ️ **Tab Informazioni**
- Documentazione completa del sistema
- Esempi di query di ricerca
- Guida all'utilizzo
- Dettagli tecnici

## 📋 Guida Passo-Passo

### Passo 1: Preparare i Dati
```bash
# Copia le tue immagini nella cartella raw
cp /percorso/delle/tue/immagini/* data/raw/

# Oppure crea una sottocartella organizzata
mkdir data/raw/paesaggi
cp /percorso/immagini/paesaggi/* data/raw/paesaggi/
```

### Passo 2: Avviare l'Interfaccia
```bash
python run.py
```

### Passo 3: Utilizzare il Sistema
1. **Inizializza** il sistema nel primo tab
2. **Indicizza** il tuo dataset nel tab "Indicizzazione"
3. **Cerca** immagini nel tab "Ricerca Globale"
4. **Analizza** singole immagini nel tab "Ricerca Segmenti"

## 🛠️ Utilizzo da Riga di Comando

### Indicizzazione Dataset
```bash
python -c "
from src.pipelines.index_dataset import DatasetIndexer
indexer = DatasetIndexer()
indexer.index_dataset('data/raw/')
"
```

### Ricerca Rapida
```bash
python -c "
from src.pipelines.index_dataset import DatasetIndexer
indexer = DatasetIndexer()
indexer.load_index()
results = indexer.search_similar_images('un tramonto sul mare', 5)
for r in results:
    print(f'{r[\"rank\"]}: {r[\"metadata\"][\"filename\"]} (score: {r[\"similarity\"]:.3f})')
"
```

## 📚 Esempi di Utilizzo

### Ricerca Semantica
```python
from src.pipelines.segment_and_search import SemanticSearchPipeline

pipeline = SemanticSearchPipeline()
results = pipeline.batch_search_scenes("montagne innevate", top_k=10)
```

### Segmentazione Immagine
```python
results, mask = pipeline.search_in_segments(
    "data/scene_examples/landscape.jpg", 
    "cielo blu", 
    top_k=3, 
    method="grid"
)
```

### Fine-tuning Personalizzato
```python
from src.pipelines.fine_tune_clip import FineTunePipeline

pipeline = FineTunePipeline()
pipeline.run_fine_tuning("data/training_data.json", "json")
```

## 🎯 Funzionalità Principali

### 🔍 **Ricerca Semantica Avanzata**
- Query in linguaggio naturale italiano/inglese
- Ranking per similarità coseno
- Supporto per dataset di grandi dimensioni
- Database vettoriale FAISS ottimizzato

### 🧩 **Segmentazione Intelligente**
- Algoritmi multipli di segmentazione
- Ricerca granulare in parti dell'immagine
- Visualizzazione interattiva dei risultati
- Parametri configurabili in tempo reale

### 📊 **Sistema di Indicizzazione**
- Elaborazione batch efficiente
- Metadati automatici per ogni immagine
- Persistenza su disco del database
- Supporto per aggiornamenti incrementali

### 🎯 **Fine-tuning CLIP**
- Personalizzazione su domini specifici
- Training con dati etichettati
- Salvataggio checkpoint intermedi
- Validazione automatica

## 🎨 Esempi di Query

### Ricerca Globale
- "un tramonto arancione sul mare"
- "persone che camminano in una città moderna"
- "montagne innevate con cielo sereno"
- "gatti che dormono su un divano"
- "architettura futuristica di notte"
- "bambini che giocano in un parco"

### Ricerca Segmenti
- "cielo blu" (in un paesaggio)
- "finestre illuminate" (in un edificio)
- "foglie verdi" (in una foto di natura)
- "oceano" (in una vista costiera)
- "volto sorridente" (in una foto di gruppo)

## 🔧 Risoluzione Problemi

### Errore CUDA non disponibile
```bash
# Modifica config.yaml per usare CPU
sed -i 's/device: "cuda"/device: "cpu"/g' config.yaml
```

### Errore moduli non trovati
```bash
# Assicurati di essere nella directory del progetto
cd /percorso/del/progetto

# Aggiungi il percorso al PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Errore dipendenze mancanti
```bash
# Reinstalla tutte le dipendenze
pip install --upgrade -r requirements.txt

# Per problemi con OpenCV:
pip install opencv-python-headless
```

### Problemi con Gradio
```bash
# Aggiorna Gradio all'ultima versione
pip install --upgrade gradio

# Controlla le porte disponibili
netstat -an | grep 7860
```

## 🚀 Tecnologie Utilizzate

- **🤖 CLIP**: Modello multimodale di OpenAI
- **⚡ PyTorch**: Framework di deep learning
- **🔍 FAISS**: Ricerca vettoriale efficiente (Facebook AI)
- **🎨 Gradio**: Interfaccia web interattiva
- **🖼️ OpenCV**: Elaborazione immagini
- **📊 Matplotlib**: Visualizzazione risultati
- **🐍 Python 3.8+**: Linguaggio di programmazione

## 📈 Prestazioni

- **Indicizzazione**: ~100-500 immagini/minuto (CPU)
- **Ricerca**: <1 secondo per query (database indicizzato)
- **Segmentazione**: 2-5 secondi per immagine
- **Memoria**: ~2-4GB RAM per dataset medi (10K immagini)

## 🆘 Supporto

Per problemi o domande:
1. Controlla la sezione "Risoluzione Problemi"
2. Verifica la configurazione in `config.yaml`
3. Consulta i log dell'interfaccia Gradio
4. Assicurati che tutte le dipendenze siano aggiornate

## 📝 Note Importanti

- **Primo avvio**: Il download dei modelli CLIP può richiedere tempo
- **Indicizzazione**: Necessaria solo al primo utilizzo o per nuove immagini
- **Prestazioni**: Ottimizzato per CPU, supporta GPU se disponibile
- **Formati**: Supporta JPG, JPEG, PNG, BMP, TIFF
- **Interfaccia**: Accessibile da browser web moderni

## 🔄 Aggiornamenti

Il sistema viene regolarmente aggiornato con:
- Nuovi algoritmi di segmentazione
- Miglioramenti delle prestazioni
- Funzionalità aggiuntive dell'interfaccia
- Supporto per nuovi formati di immagine
