"""
Interfaccia utente web per il sistema di ricerca semantica delle immagini.
"""

import streamlit as st
import os
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64

# Importa i moduli del progetto
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.pipelines.segment_and_search import SemanticSearchPipeline
from src.pipelines.index_dataset import DatasetIndexer
from src.pipelines.fine_tune_clip import FineTunePipeline


class StreamlitApp:
    """
    Applicazione Streamlit per la ricerca semantica delle immagini.
    """

    def __init__(self):
        self.config_path = "config.yaml"
        self.pipeline = None
        self.indexer = None

        # Inizializza la sessione
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.pipeline = None
            st.session_state.indexer = None

    def load_config(self):
        """Carica la configurazione."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            st.error(f"File di configurazione {self.config_path} non trovato!")
            return None

    def initialize_components(self):
        """Inizializza i componenti principali."""
        if not st.session_state.initialized:
            with st.spinner("Inizializzazione componenti..."):
                try:
                    st.session_state.pipeline = SemanticSearchPipeline(self.config_path)
                    st.session_state.indexer = DatasetIndexer(self.config_path)
                    st.session_state.indexer.load_index()
                    st.session_state.initialized = True
                    st.success("Componenti inizializzati con successo!")
                except Exception as e:
                    st.error(f"Errore nell'inizializzazione: {e}")
                    return False
        return True

    def sidebar_navigation(self):
        """Crea la barra laterale di navigazione."""
        st.sidebar.title("🔍 CLIP Scene Search")

        pages = {
            "🏠 Home": "home",
            "🔍 Ricerca Globale": "global_search",
            "🧩 Ricerca Segmenti": "segment_search",
            "📊 Indicizzazione": "indexing",
            "🎯 Fine-tuning": "fine_tuning",
            "⚙️ Configurazione": "config"
        }

        selected_page = st.sidebar.selectbox("Navigazione", list(pages.keys()))
        return pages[selected_page]

    def home_page(self):
        """Pagina principale."""
        st.title("🎯 CLIP Scene Search System")
        st.markdown("---")

        st.markdown("""
        ## Benvenuto nel sistema di ricerca semantica delle immagini!
        
        Questo sistema utilizza il modello CLIP per:
        - 🔍 **Ricerca globale**: Cerca immagini simili in tutto il dataset
        - 🧩 **Ricerca per segmenti**: Trova parti specifiche all'interno delle immagini
        - 📊 **Indicizzazione**: Crea un database vettoriale delle tue immagini
        - 🎯 **Fine-tuning**: Personalizza il modello sui tuoi dati
        
        ### Come iniziare:
        1. **Configura** il sistema nella sezione Configurazione
        2. **Indicizza** il tuo dataset nella sezione Indicizzazione
        3. **Cerca** immagini usando testo naturale
        4. **Personalizza** il modello con il fine-tuning (opzionale)
        """)

        # Statistiche del sistema
        if st.session_state.initialized and st.session_state.indexer:
            st.markdown("---")
            st.subheader("📈 Statistiche del Sistema")

            col1, col2, col3 = st.columns(3)

            with col1:
                total_images = st.session_state.indexer.index.ntotal if st.session_state.indexer.index else 0
                st.metric("Immagini Indicizzate", total_images)

            with col2:
                config = self.load_config()
                model_name = config['clip']['model_name'] if config else "N/A"
                st.metric("Modello CLIP", model_name)

            with col3:
                device = config['clip']['device'] if config else "N/A"
                st.metric("Device", device)

    def global_search_page(self):
        """Pagina per la ricerca globale."""
        st.title("🔍 Ricerca Globale nel Dataset")
        st.markdown("Cerca immagini simili in tutto il dataset usando descrizioni testuali.")

        if not self.initialize_components():
            return

        # Input della query
        query = st.text_input("Descrivi quello che stai cercando:",
                             placeholder="es. 'un tramonto sul mare', 'una foresta verde', 'un gatto nero'")

        # Parametri di ricerca
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Numero di risultati", 1, 20, 5)
        with col2:
            min_similarity = st.slider("Similarità minima", 0.0, 1.0, 0.3, 0.1)

        if st.button("🔍 Cerca", type="primary") and query:
            with st.spinner("Ricerca in corso..."):
                try:
                    results = st.session_state.indexer.search_similar_images(query, top_k)

                    if results:
                        st.success(f"Trovati {len(results)} risultati!")

                        # Mostra i risultati
                        for result in results:
                            if result['similarity'] >= min_similarity:
                                col1, col2 = st.columns([1, 2])

                                with col1:
                                    try:
                                        img_path = result['metadata']['path']
                                        if os.path.exists(img_path):
                                            image = Image.open(img_path)
                                            st.image(image, caption=f"Similarità: {result['similarity']:.3f}")
                                        else:
                                            st.error("Immagine non trovata")
                                    except Exception as e:
                                        st.error(f"Errore nel caricamento: {e}")

                                with col2:
                                    st.write(f"**Rank:** {result['rank']}")
                                    st.write(f"**Similarità:** {result['similarity']:.3f}")
                                    st.write(f"**File:** {result['metadata']['filename']}")
                                    st.write(f"**Percorso:** {result['metadata']['path']}")

                                st.markdown("---")
                    else:
                        st.warning("Nessun risultato trovato. Assicurati che il dataset sia stato indicizzato.")

                except Exception as e:
                    st.error(f"Errore durante la ricerca: {e}")

    def segment_search_page(self):
        """Pagina per la ricerca nei segmenti."""
        st.title("🧩 Ricerca per Segmenti")
        st.markdown("Carica un'immagine e cerca parti specifiche al suo interno.")

        if not self.initialize_components():
            return

        # Upload dell'immagine
        uploaded_file = st.file_uploader(
            "Carica un'immagine",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Formati supportati: JPG, JPEG, PNG, BMP"
        )

        if uploaded_file:
            # Salva temporaneamente l'immagine
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Mostra l'immagine caricata
            image = Image.open(temp_path)
            st.image(image, caption="Immagine caricata", use_container_width=True)

            # Parametri di segmentazione
            st.subheader("⚙️ Parametri di Segmentazione")

            col1, col2 = st.columns(2)
            with col1:
                seg_method = st.selectbox(
                    "Metodo di segmentazione",
                    ["grid", "kmeans", "superpixel"],
                    help="Grid: divisione uniforme, K-means: per colori, Superpixel: regioni omogenee"
                )

            with col2:
                if seg_method == "grid":
                    grid_rows = st.slider("Righe griglia", 2, 8, 4)
                    grid_cols = st.slider("Colonne griglia", 2, 8, 4)
                    seg_kwargs = {"grid_size": (grid_rows, grid_cols)}
                elif seg_method == "kmeans":
                    n_clusters = st.slider("Numero cluster", 3, 15, 8)
                    seg_kwargs = {"n_clusters": n_clusters}
                else:  # superpixel
                    n_segments = st.slider("Numero segmenti", 50, 300, 100)
                    seg_kwargs = {"n_segments": n_segments}

            # Query di ricerca
            query = st.text_input("Cosa stai cercando nell'immagine?",
                                 placeholder="es. 'cielo blu', 'albero', 'edificio'")

            top_k_segments = st.slider("Numero di segmenti da mostrare", 1, 10, 3)

            if st.button("🔍 Cerca nei Segmenti", type="primary") and query:
                with st.spinner("Segmentazione e ricerca in corso..."):
                    try:
                        results, mask = st.session_state.pipeline.search_in_segments(
                            temp_path, query, top_k_segments, seg_method, **seg_kwargs
                        )

                        if results:
                            st.success(f"Trovati {len(results)} segmenti rilevanti!")

                            # Visualizza i risultati
                            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                            # Immagine originale
                            axes[0].imshow(image)
                            axes[0].set_title("Immagine Originale")
                            axes[0].axis('off')

                            # Immagine con segmenti evidenziati
                            axes[1].imshow(image)
                            axes[1].set_title(f"Segmenti per: '{query}'")
                            axes[1].axis('off')

                            # Disegna i bounding box
                            colors = plt.cm.Set3(np.linspace(0, 1, len(results)))

                            for i, (result, color) in enumerate(zip(results, colors)):
                                bbox = result['segment_info']['bbox']
                                x, y, x2, y2 = bbox

                                rect = patches.Rectangle((x, y), x2-x, y2-y,
                                                       linewidth=3, edgecolor=color,
                                                       facecolor='none', alpha=0.8)
                                axes[1].add_patch(rect)

                                axes[1].text(x, y-5, f"#{i+1}\n{result['similarity']:.3f}",
                                           fontsize=10, color=color, fontweight='bold',
                                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

                            plt.tight_layout()
                            st.pyplot(fig)

                            # Dettagli dei risultati
                            st.subheader("📊 Dettagli dei Segmenti")
                            for i, result in enumerate(results):
                                with st.expander(f"Segmento #{i+1} - Similarità: {result['similarity']:.3f}"):
                                    st.json(result['segment_info'])

                        else:
                            st.warning("Nessun segmento rilevante trovato.")

                    except Exception as e:
                        st.error(f"Errore durante la segmentazione: {e}")

                    finally:
                        # Pulisci il file temporaneo
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

    def indexing_page(self):
        """Pagina per l'indicizzazione del dataset."""
        st.title("📊 Indicizzazione del Dataset")
        st.markdown("Crea un database vettoriale delle tue immagini per la ricerca veloce.")

        config = self.load_config()
        if not config:
            return

        # Stato attuale dell'indice
        if st.session_state.initialized and st.session_state.indexer:
            total_images = st.session_state.indexer.index.ntotal if st.session_state.indexer.index else 0
            st.info(f"📈 Immagini attualmente indicizzate: **{total_images}**")

        # Configurazione della directory
        st.subheader("📁 Configurazione Directory")
        data_dir = st.text_input(
            "Directory delle immagini",
            value=config['dataset']['raw_data_path'],
            help="Percorso della directory contenente le immagini da indicizzare"
        )

        # Mostra anteprima delle immagini trovate
        if data_dir and os.path.exists(data_dir):
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []

            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))

            st.success(f"🎯 Trovate **{len(image_files)}** immagini nella directory")

            # Mostra alcune immagini di esempio
            if image_files:
                st.subheader("🖼️ Anteprima Immagini")
                sample_images = image_files[:6]  # Mostra massimo 6 immagini

                cols = st.columns(3)
                for i, img_path in enumerate(sample_images):
                    try:
                        with cols[i % 3]:
                            image = Image.open(img_path)
                            st.image(image, caption=os.path.basename(img_path), use_container_width=True)
                    except Exception as e:
                        st.error(f"Errore nel caricamento di {img_path}: {e}")

            # Pulsante per avviare l'indicizzazione
            st.subheader("🚀 Avvia Indicizzazione")

            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.slider("Batch size", 8, 64, 32, help="Numero di immagini elaborate insieme")
            with col2:
                overwrite = st.checkbox("Sovrascrivi indice esistente", help="Cancella l'indice esistente e ricrea")

            if st.button("📊 Avvia Indicizzazione", type="primary"):
                if not self.initialize_components():
                    return

                with st.spinner("Indicizzazione in corso... Questo potrebbe richiedere del tempo."):
                    try:
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Simula il progresso (in una implementazione reale,
                        # dovremmo modificare la classe DatasetIndexer per supportare callback)
                        status_text.text("Inizializzazione...")
                        progress_bar.progress(10)

                        # Esegui l'indicizzazione
                        st.session_state.indexer.index_dataset(data_dir)

                        progress_bar.progress(100)
                        status_text.text("Indicizzazione completata!")

                        st.success("✅ Indicizzazione completata con successo!")
                        st.balloons()

                        # Aggiorna le statistiche
                        total_images = st.session_state.indexer.index.ntotal
                        st.success(f"🎯 **{total_images}** immagini sono ora disponibili per la ricerca!")

                    except Exception as e:
                        st.error(f"❌ Errore durante l'indicizzazione: {e}")
        else:
            st.warning("⚠️ Directory non trovata o non valida!")

    def fine_tuning_page(self):
        """Pagina per il fine-tuning."""
        st.title("🎯 Fine-tuning del Modello CLIP")
        st.markdown("Personalizza il modello CLIP sui tuoi dati specifici.")

        st.info("⚠️ **Nota**: Il fine-tuning richiede dati etichettati e risorse computazionali significative.")

        # Upload del file di training
        st.subheader("📁 Dati di Training")

        training_file = st.file_uploader(
            "Carica file di training (JSON)",
            type=['json'],
            help="File JSON con formato: [{'image_path': '...', 'text': '...'}]"
        )

        if training_file:
            st.success("File di training caricato!")

            # Parametri di fine-tuning
            st.subheader("⚙️ Parametri di Training")

            col1, col2 = st.columns(2)
            with col1:
                epochs = st.slider("Numero di epoche", 1, 20, 5)
                batch_size = st.slider("Batch size", 4, 32, 16)

            with col2:
                learning_rate = st.select_slider(
                    "Learning rate",
                    options=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
                    value=1e-5,
                    format_func=lambda x: f"{x:.0e}"
                )
                train_split = st.slider("Split training", 0.6, 0.9, 0.8)

            # Avvia fine-tuning
            if st.button("🚀 Avvia Fine-tuning", type="primary"):
                st.warning("⚠️ Funzionalità in sviluppo. Il fine-tuning richiede un setup più avanzato.")
                st.info("Per implementare il fine-tuning completo, consulta la documentazione del progetto.")

        else:
            # Crea file di esempio
            if st.button("📄 Crea File di Esempio"):
                try:
                    pipeline = FineTunePipeline()
                    pipeline.create_sample_data("sample_training_data.json")
                    st.success("File di esempio creato: `sample_training_data.json`")

                    # Mostra il contenuto del file di esempio
                    with open("sample_training_data.json", 'r') as f:
                        sample_content = f.read()

                    st.subheader("📋 Formato del File di Training")
                    st.code(sample_content, language='json')

                except Exception as e:
                    st.error(f"Errore nella creazione del file di esempio: {e}")

    def config_page(self):
        """Pagina di configurazione."""
        st.title("⚙️ Configurazione Sistema")
        st.markdown("Modifica le impostazioni del sistema.")

        config = self.load_config()
        if not config:
            return

        # Configurazione CLIP
        st.subheader("🤖 Configurazione CLIP")

        col1, col2 = st.columns(2)
        with col1:
            model_name = st.selectbox(
                "Modello CLIP",
                ["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"],
                index=0 if config['clip']['model_name'] == "ViT-B/32" else 0
            )

        with col2:
            device = st.selectbox(
                "Device",
                ["cpu", "cuda"],
                index=1 if config['clip']['device'] == "cuda" else 0
            )

        # Configurazione Dataset
        st.subheader("📊 Configurazione Dataset")

        raw_data_path = st.text_input(
            "Percorso dati grezzi",
            value=config['dataset']['raw_data_path']
        )

        vector_db_path = st.text_input(
            "Percorso database vettoriale",
            value=config['vector_db']['path']
        )

        # Salva configurazione
        if st.button("💾 Salva Configurazione", type="primary"):
            try:
                # Aggiorna la configurazione
                config['clip']['model_name'] = model_name
                config['clip']['device'] = device
                config['dataset']['raw_data_path'] = raw_data_path
                config['vector_db']['path'] = vector_db_path

                # Salva su file
                with open(self.config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)

                st.success("✅ Configurazione salvata!")
                st.info("🔄 Riavvia l'applicazione per applicare le modifiche.")

            except Exception as e:
                st.error(f"❌ Errore nel salvare la configurazione: {e}")

        # Mostra configurazione corrente
        st.subheader("📋 Configurazione Corrente")
        st.json(config)

    def run(self):
        """Esegue l'applicazione Streamlit."""
        st.set_page_config(
            page_title="CLIP Scene Search",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # CSS personalizzato
        st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .stSelectbox > div > div {
            background-color: #f0f2f6;
        }
        </style>
        """, unsafe_allow_html=True)

        # Navigazione
        current_page = self.sidebar_navigation()

        # Router delle pagine
        if current_page == "home":
            self.home_page()
        elif current_page == "global_search":
            self.global_search_page()
        elif current_page == "segment_search":
            self.segment_search_page()
        elif current_page == "indexing":
            self.indexing_page()
        elif current_page == "fine_tuning":
            self.fine_tuning_page()
        elif current_page == "config":
            self.config_page()


def main():
    """Funzione principale per avviare l'app."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
