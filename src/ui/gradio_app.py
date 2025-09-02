"""
Interfaccia Gradio per il sistema di ricerca semantica delle immagini.
Questa interfaccia fornisce una demo interattiva delle capacità del sistema CLIP Scene Search.
"""

import gradio as gr
import os
import sys
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from typing import List, Tuple, Optional

# Importa i moduli del progetto
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.pipelines.segment_and_search import SemanticSearchPipeline
from src.pipelines.index_dataset import DatasetIndexer


class GradioDemo:
    """
    Demo Gradio per il sistema di ricerca semantica CLIP.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.pipeline = None
        self.indexer = None
        self.is_initialized = False

    def load_config(self) -> dict:
        """Carica la configurazione."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return {}

    def initialize_system(self) -> str:
        """Inizializza il sistema."""
        if self.is_initialized:
            return "✅ Sistema già inizializzato!"

        try:
            self.pipeline = SemanticSearchPipeline(self.config_path)
            self.indexer = DatasetIndexer(self.config_path)

            # Prova a caricare l'indice esistente
            try:
                self.indexer.load_index()
                total_images = self.indexer.index.ntotal if self.indexer.index else 0
                if total_images > 0:
                    self.is_initialized = True
                    return f"✅ Sistema inizializzato! Database con {total_images} immagini caricato."
                else:
                    raise Exception("Database vuoto")
            except:
                # Se non esiste un database, prova ad auto-indicizzare il dataset di Naruto
                config = self.load_config()
                raw_data_path = config.get('dataset', {}).get('raw_data_path', 'data/raw/')
                naruto_path = os.path.join(raw_data_path, 'Anime-Naruto')

                if os.path.exists(naruto_path):
                    try:
                        # Auto-indicizza il dataset di Naruto
                        self.indexer.index_dataset(naruto_path)

                        # Ricarica l'indice dopo l'indicizzazione
                        self.indexer.load_index()
                        total_images = self.indexer.index.ntotal if self.indexer.index else 0
                        self.is_initialized = True
                        return f"✅ Sistema inizializzato! Dataset Naruto auto-indicizzato con {total_images} immagini."
                    except Exception as e:
                        self.is_initialized = True
                        return f"✅ Sistema inizializzato! ⚠️ Errore nell'auto-indicizzazione: {str(e)}"
                else:
                    self.is_initialized = True
                    return "✅ Sistema inizializzato! ⚠️ Nessun dataset trovato - carica manualmente nella tab Indicizzazione."

        except Exception as e:
            return f"❌ Errore nell'inizializzazione: {str(e)}"

    def search_images(self, query: str, top_k: int = 5) -> Tuple[str, List]:
        """Ricerca immagini nel dataset."""
        if not self.is_initialized:
            return "❌ Sistema non inizializzato!", []

        if not query.strip():
            return "⚠️ Inserisci una query di ricerca!", []

        try:
            results = self.indexer.search_similar_images(query, top_k)

            if not results:
                return "❌ Nessun risultato trovato. Assicurati che il dataset sia indicizzato.", []

            # Prepara le immagini per Gradio
            images = []
            for result in results:
                img_path = result['metadata']['path']
                if os.path.exists(img_path):
                    images.append((img_path, f"Similarità: {result['similarity']:.3f}"))
                else:
                    # Crea un placeholder se l'immagine non esiste
                    placeholder = Image.new('RGB', (224, 224), color='gray')
                    images.append((placeholder, f"Immagine non trovata - Sim: {result['similarity']:.3f}"))

            return f"✅ Trovati {len(results)} risultati per: '{query}'", images

        except Exception as e:
            return f"❌ Errore durante la ricerca: {str(e)}", []

    def search_segments(self, image, query: str, method: str = "grid",
                       grid_size: int = 4, n_clusters: int = 8, n_segments: int = 100) -> Tuple[str, Optional[Image.Image]]:
        """Ricerca segmenti nell'immagine."""
        if not self.is_initialized:
            return "❌ Sistema non inizializzato!", None

        if image is None:
            return "⚠️ Carica un'immagine!", None

        if not query.strip():
            return "⚠️ Inserisci una query di ricerca!", None

        try:
            # Salva temporaneamente l'immagine
            temp_path = "temp_gradio_image.jpg"
            image.save(temp_path)

            # Prepara i parametri di segmentazione
            if method == "grid":
                seg_kwargs = {"grid_size": (grid_size, grid_size)}
            elif method == "kmeans":
                seg_kwargs = {"n_clusters": n_clusters}
            elif method == "sam":
                seg_kwargs = {}  # SAM non ha parametri configurabili nell'interfaccia
            else:  # superpixel
                seg_kwargs = {"n_segments": n_segments}

            # Esegui la ricerca nei segmenti
            results, mask = self.pipeline.search_in_segments(
                temp_path, query, top_k=3, segmentation_method=method, **seg_kwargs
            )

            if not results:
                os.remove(temp_path)
                return "❌ Nessun segmento rilevante trovato.", None

            # Crea la visualizzazione
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            ax.imshow(image)
            ax.set_title(f"Segmenti per: '{query}' (Metodo: {method})")
            ax.axis('off')

            # Disegna i bounding box dei segmenti migliori
            colors = plt.cm.get_cmap('tab10', len(results))

            for i, result in enumerate(results):
                bbox = result['segment_info']['bbox']
                x, y, x2, y2 = bbox

                rect = patches.Rectangle((x, y), x2-x, y2-y,
                                       linewidth=3, edgecolor=colors(i),
                                       facecolor='none', alpha=0.8)
                ax.add_patch(rect)

                ax.text(x, y-5, f"#{i+1}\n{result['similarity']:.3f}",
                       fontsize=12, color=colors(i), fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

            # Salva la figura
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            result_image = Image.open(buf)
            plt.close()

            # Pulisci
            os.remove(temp_path)

            message = f"✅ Trovati {len(results)} segmenti rilevanti!"
            return message, result_image

        except Exception as e:
            if os.path.exists("temp_gradio_image.jpg"):
                os.remove("temp_gradio_image.jpg")
            return f"❌ Errore durante la segmentazione: {str(e)}", None

    def index_sample_data(self, data_path: str) -> str:
        """Indicizza dati di esempio."""
        if not self.is_initialized:
            return "❌ Sistema non inizializzato!"

        if not data_path.strip():
            return "⚠️ Inserisci il percorso della directory!"

        if not os.path.exists(data_path):
            return f"❌ Directory non trovata: {data_path}"

        try:
            # Conta le immagini
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []

            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))

            if not image_files:
                return f"❌ Nessuna immagine trovata in: {data_path}"

            # Esegui l'indicizzazione
            self.indexer.index_dataset(data_path)

            return f"✅ Indicizzazione completata! {len(image_files)} immagini elaborate."

        except Exception as e:
            return f"❌ Errore durante l'indicizzazione: {str(e)}"

    def evaluate_performance(self) -> str:
        """Esegui valutazione delle performance del sistema."""
        if not self.is_initialized:
            return "❌ Sistema non inizializzato!"

        try:
            from src.pipelines.performance_evaluation import PerformanceEvaluator

            evaluator = PerformanceEvaluator(self.config_path)

            # Genera report completo
            report_path = evaluator.generate_comprehensive_report()

            return f"✅ Valutazione completata!\n📋 Report salvato in: {report_path}\n\n" \
                   f"Il report include:\n" \
                   f"• Metriche di ricerca semantica\n" \
                   f"• Performance dei metodi di segmentazione\n" \
                   f"• Qualità degli embedding CLIP\n" \
                   f"• Grafici e raccomandazioni"

        except Exception as e:
            return f"❌ Errore durante la valutazione: {str(e)}"

    def create_interface(self) -> gr.Blocks:
        """Crea l'interfaccia Gradio."""

        with gr.Blocks(
            title="🔍 CLIP Scene Search Demo",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .tab-nav button {
                font-size: 16px !important;
            }
            """
        ) as demo:

            gr.Markdown("""
            # 🎯 CLIP Scene Search System Demo
            
            Questo sistema utilizza il modello CLIP per la ricerca semantica avanzata nelle immagini.
            
            **Funzionalità disponibili:**
            - 🔍 **Ricerca Globale**: Trova immagini simili nel dataset usando descrizioni testuali
            - 🧩 **Ricerca Segmenti**: Analizza parti specifiche di un'immagine
            - 📊 **Indicizzazione**: Prepara il dataset per la ricerca veloce
            """)

            # Tab per l'inizializzazione
            with gr.Tab("🚀 Inizializzazione"):
                gr.Markdown("### Inizializza il sistema prima di utilizzare le altre funzionalità")

                init_button = gr.Button("🔄 Inizializza Sistema", variant="primary", size="lg")
                init_output = gr.Textbox(label="Stato Sistema", lines=3)

                init_button.click(
                    fn=self.initialize_system,
                    outputs=init_output
                )

            # Tab per la ricerca globale
            with gr.Tab("🔍 Ricerca Globale"):
                gr.Markdown("### Cerca immagini simili nel dataset usando descrizioni testuali")

                with gr.Row():
                    with gr.Column(scale=2):
                        search_query = gr.Textbox(
                            label="Query di Ricerca",
                            placeholder="es. 'un tramonto sul mare', 'una foresta verde', 'un gatto nero'",
                            lines=2
                        )
                        search_top_k = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Numero di risultati"
                        )
                        search_button = gr.Button("🔍 Cerca", variant="primary")

                    with gr.Column(scale=1):
                        search_status = gr.Textbox(label="Stato", lines=3)

                search_gallery = gr.Gallery(
                    label="Risultati della Ricerca",
                    show_label=True,
                    elem_id="search_gallery",
                    columns=3,
                    rows=2,
                    object_fit="contain",
                    height="auto"
                )

                search_button.click(
                    fn=self.search_images,
                    inputs=[search_query, search_top_k],
                    outputs=[search_status, search_gallery]
                )

            # Tab per la ricerca nei segmenti
            with gr.Tab("🧩 Ricerca Segmenti"):
                gr.Markdown("### Carica un'immagine e cerca parti specifiche al suo interno")

                with gr.Row():
                    with gr.Column(scale=1):
                        segment_image = gr.Image(
                            label="Carica Immagine",
                            type="pil",
                            height=300
                        )

                        segment_query = gr.Textbox(
                            label="Cosa stai cercando?",
                            placeholder="es. 'volto di una persona', 'cielo blu', 'edificio'",
                            lines=2
                        )

                        segment_method = gr.Dropdown(
                            choices=["grid", "kmeans", "superpixel", "sam"],
                            value="grid",
                            label="Metodo di Segmentazione",
                            info="SAM fornisce la segmentazione più accurata ma richiede più tempo"
                        )

                        with gr.Accordion("⚙️ Parametri Avanzati", open=False):
                            grid_size = gr.Slider(2, 8, 4, step=1, label="Dimensione Griglia")
                            n_clusters = gr.Slider(4, 16, 8, step=1, label="Numero Cluster K-means")
                            n_segments = gr.Slider(50, 200, 100, step=10, label="Numero Superpixel")

                        segment_button = gr.Button("🔍 Analizza Segmenti", variant="primary")

                    with gr.Column(scale=2):
                        segment_status = gr.Textbox(label="Stato", lines=2)
                        segment_result = gr.Image(label="Risultati Segmentazione", height=400)

                segment_button.click(
                    fn=self.search_segments,
                    inputs=[segment_image, segment_query, segment_method,
                           grid_size, n_clusters, n_segments],
                    outputs=[segment_status, segment_result]
                )

            # Tab per l'indicizzazione
            with gr.Tab("📊 Indicizzazione"):
                gr.Markdown("### Aggiungi nuove immagini al database per la ricerca")

                index_path = gr.Textbox(
                    label="Percorso Directory Immagini",
                    placeholder="es. data/raw/nuovo_dataset/",
                    value="data/raw/Anime-Naruto/"
                )

                index_button = gr.Button("📥 Indicizza Dataset", variant="primary")
                index_status = gr.Textbox(label="Stato Indicizzazione", lines=4)

                index_button.click(
                    fn=self.index_sample_data,
                    inputs=index_path,
                    outputs=index_status
                )

            # Nuovo Tab per la valutazione delle performance
            with gr.Tab("📈 Valutazione Performance"):
                gr.Markdown("""
                ### Valutazione quantitativa del sistema
                
                Questo strumento esegue una valutazione completa delle performance del sistema:
                - **Accuratezza ricerca semantica**: Precision, Recall, F1-Score, MAP
                - **Performance segmentazione**: Tempi di elaborazione, qualità dei segmenti
                - **Qualità embedding**: Coerenza intra/inter-classe
                """)

                eval_button = gr.Button("🚀 Esegui Valutazione Completa", variant="primary", size="lg")
                eval_status = gr.Textbox(
                    label="Risultati Valutazione",
                    lines=10,
                    placeholder="I risultati della valutazione appariranno qui..."
                )

                eval_button.click(
                    fn=self.evaluate_performance,
                    outputs=eval_status
                )

            # Tab informazioni aggiornato
            with gr.Tab("ℹ️ Informazioni"):
                gr.Markdown("""
                ## 🎯 CLIP Scene Search System
                
                ### Caratteristiche Principali
                
                **🔍 Ricerca Semantica Avanzata**
                - Utilizza il modello CLIP per comprendere il contenuto delle immagini
                - Ricerca tramite descrizioni in linguaggio naturale
                - Database vettoriale FAISS per ricerche veloci
                
                **🧩 Segmentazione Multi-Metodo**
                - **Grid**: Divisione uniforme dell'immagine in regioni
                - **K-means**: Clustering basato sui colori
                - **Superpixel**: Regioni semanticamente coerenti (SLIC)
                - **SAM**: Segmentazione automatica avanzata (Segment Anything Model)
                
                **📊 Valutazione Performance**
                - Metriche quantitative di accuratezza
                - Analisi comparative dei metodi di segmentazione
                - Report dettagliati con visualizzazioni
                
                ### Obiettivi del Progetto ✅
                
                ✅ **Dataset preprocessing**: Acquisizione e preprocessamento dataset caratteri  
                ✅ **Indexing pipeline**: Creazione database vettoriale con embedding CLIP  
                ✅ **Scene analysis**: Segmentazione immagini con multipli algoritmi (+SAM)  
                ✅ **Matching algorithm**: Confronto embedding per identificazione caratteri  
                ✅ **Interactive interface**: Interfaccia web Gradio completa  
                ✅ **Performance evaluation**: Sistema di valutazione quantitativa e qualitativa  
                
                ### Esempi di Query
                
                **Ricerca Globale:**
                - "personaggio con capelli biondi"
                - "ninja con fascia sulla fronte"
                - "personaggio con occhi rossi"
                - "scena di combattimento"
                
                **Ricerca Segmenti:**
                - "volto del personaggio"
                - "simbolo sulla fronte"
                - "vestiti colorati"
                - "sfondo naturale"
                
                ### Supporto Tecnico
                
                - **Modello**: CLIP ViT-B/32
                - **Database**: FAISS IndexFlatIP
                - **Segmentazione**: Grid, K-means, SLIC, SAM
                - **Interfaccia**: Gradio con tema personalizzato
                """)

        return demo


def create_demo(config_path: str = "config.yaml") -> gr.Blocks:
    """Crea e restituisce la demo Gradio."""
    demo_app = GradioDemo(config_path)
    return demo_app.create_interface()


def main():
    """Avvia la demo Gradio."""
    print("🚀 Avvio CLIP Scene Search Demo (Gradio)...")

    demo = create_demo()

    # Configurazione del server
    demo.launch(
        server_name="0.0.0.0",  # Accessibile da rete locale
        server_port=7860,       # Porta standard Gradio
        share=False,            # Cambia a True per condivisione pubblica
        debug=True,
        show_error=True,
        inbrowser=True          # Apre automaticamente nel browser
    )


if __name__ == "__main__":
    main()
