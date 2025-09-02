# Python Project

## Descrizione
Progetto per il fine-tuning di CLIP e la segmentazione/ricerca di scene.

## Struttura del Progetto
```
data/
  raw/              # Dati grezzi
  scene_examples/   # Esempi di scene
  vector_db/        # Database vettoriale
src/
  models/           # Modelli di machine learning
  pipelines/        # Pipeline di elaborazione
  ui/              # Interfaccia utente
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

## Come Far Partire il Progetto

### 🚀 Avvio Rapido

1. **Avvia l'interfaccia web:**
```bash
streamlit run src/ui/app.py
```

2. **Apri il browser** all'indirizzo che apparirà nel terminale (solitamente `http://localhost:8501`)

3. **Inizia subito:**
   - Vai nella sezione "Configurazione" per verificare le impostazioni
   - Carica alcune immagini nella cartella `data/raw/`
   - Usa la sezione "Indicizzazione" per creare il database
   - Inizia a cercare nella sezione "Ricerca Globale"

### 📋 Guida Passo-Passo

#### Passo 1: Preparare i Dati
```bash
# Copia le tue immagini nella cartella raw
cp /percorso/delle/tue/immagini/* data/raw/

# Oppure crea una sottocartella organizzata
mkdir data/raw/paesaggi
cp /percorso/immagini/paesaggi/* data/raw/paesaggi/
```

#### Passo 2: Indicizzare il Dataset
```bash
# Avvia l'interfaccia web
streamlit run src/ui/app.py

# Oppure usa lo script Python direttamente
python src/pipelines/index_dataset.py
```

#### Passo 3: Iniziare la Ricerca
Una volta indicizzato il dataset, puoi:
- Cercare immagini usando descrizioni testuali
- Analizzare singole immagini per segmenti
- Personalizzare il modello con fine-tuning

### 🛠️ Utilizzo da Riga di Comando

#### Indicizzazione Dataset
```bash
python -c "
from src.pipelines.index_dataset import DatasetIndexer
indexer = DatasetIndexer()
indexer.index_dataset('data/raw/')
"
```

#### Ricerca Rapida
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

### 🔧 Risoluzione Problemi

#### Errore CUDA non disponibile
Se vedi errori relativi a CUDA:
```bash
# Modifica config.yaml e cambia device da "cuda" a "cpu"
sed -i 's/device: "cuda"/device: "cpu"/g' config.yaml
```

#### Errore moduli non trovati
```bash
# Assicurati di essere nella directory del progetto
cd /Users/marco/PycharmProjects/PythonProject

# Aggiungi il percorso corrente al PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Errore dipendenze mancanti
```bash
# Reinstalla tutte le dipendenze
pip install --upgrade -r requirements.txt

# Per problemi con OpenCV:
pip install opencv-python-headless
```

### 📚 Esempi di Utilizzo

#### Ricerca Semantica
```python
from src.pipelines.segment_and_search import SemanticSearchPipeline

pipeline = SemanticSearchPipeline()
results = pipeline.batch_search_scenes("montagne innevate", top_k=10)
```

#### Segmentazione Immagine
```python
results, mask = pipeline.search_in_segments(
    "data/scene_examples/landscape.jpg", 
    "cielo blu", 
    top_k=3, 
    segmentation_method="grid"
)
```

#### Fine-tuning Personalizzato
```python
from src.pipelines.fine_tune_clip import FineTunePipeline

pipeline = FineTunePipeline()
pipeline.run_fine_tuning("data/training_data.json", "json")
```

### 🎯 Funzionalità Principali

1. **🔍 Ricerca Globale**: Trova immagini simili in tutto il dataset usando descrizioni testuali
2. **🧩 Ricerca Segmenti**: Analizza parti specifiche di singole immagini
3. **📊 Indicizzazione**: Crea automaticamente un database vettoriale per ricerche veloci
4. **🎯 Fine-tuning**: Personalizza il modello CLIP sui tuoi dati specifici
5. **⚙️ Configurazione**: Interfaccia web per gestire tutte le impostazioni

### 🆘 Supporto

Per problemi o domande:
1. Controlla la sezione "Risoluzione Problemi" sopra
2. Verifica che tutte le dipendenze siano installate correttamente
3. Assicurati che il file `config.yaml` sia configurato correttamente

### 📝 Note

- Il primo avvio potrebbe richiedere del tempo per scaricare i modelli CLIP
- L'indicizzazione è necessaria solo la prima volta o quando aggiungi nuove immagini
- Il sistema è ottimizzato per CPU, ma supporta anche GPU se disponibile
