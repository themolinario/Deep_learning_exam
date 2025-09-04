"""
Interfaccia utente Gradio per il sistema di Scene Analysis.
"""

import gradio as gr
import numpy as np
from PIL import Image
import yaml
import os
from typing import List, Dict, Any, Tuple, Optional
import json

# Import delle pipeline
from ..pipelines.segment_and_search import SceneAnalyzer
from ..pipelines.index_dataset import DatasetIndexer
from ..pipelines.fine_tune_clip import CLIPFineTuner, split_dataset


def convert_numpy_types(obj):
    """
    Converte ricorsivamente i tipi numpy in tipi Python nativi per la serializzazione JSON.

    Args:
        obj: Oggetto da convertire

    Returns:
        Oggetto con tipi Python nativi
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


class GradioApp:
    """
    Applicazione Gradio per il sistema di Scene Analysis.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Inizializza l'applicazione Gradio.

        Args:
            config_path: Percorso del file di configurazione
        """
        # Carica configurazione
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Inizializza l'analizzatore di scene
        self.scene_analyzer = SceneAnalyzer(self.config)

        # Carica il vector database se disponibile
        vector_db_config = self.config.get('vector_db', {})
        index_path = vector_db_config.get('index_path', 'data/vector_db/image_index.faiss')
        metadata_path = vector_db_config.get('metadata_path', 'data/vector_db/image_metadata.json')

        if os.path.exists(metadata_path):
            self.scene_analyzer.load_vector_database(index_path, metadata_path)

        # Configurazione UI
        ui_config = self.config.get('ui', {})
        self.title = ui_config.get('title', "Sistema di Scene Analysis - Personaggi Naruto")
        self.port = ui_config.get('port', 7860)
        self.share = ui_config.get('share', False)
        self.debug = ui_config.get('debug', True)
        self.max_image_size = ui_config.get('max_image_size', 1024)

    def analyze_scene_interface(self, image: np.ndarray,
                               use_prompts: bool = False) -> Tuple[np.ndarray, str, str]:
        """
        Interfaccia per l'analisi delle scene.

        Args:
            image: Immagine caricata dall'utente
            use_prompts: Se usare segmentazione guidata

        Returns:
            Tupla con (immagine_annotata, risultati_json, riassunto)
        """
        if image is None:
            return None, "❌ Nessuna immagine caricata", "Carica un'immagine per iniziare l'analisi."

        try:
            # Ridimensiona immagine se troppo grande
            if max(image.shape[:2]) > self.max_image_size:
                scale = self.max_image_size / max(image.shape[:2])
                new_height = int(image.shape[0] * scale)
                new_width = int(image.shape[1] * scale)
                image = np.array(Image.fromarray(image).resize((new_width, new_height)))

            # Analizza la scena
            results = self.scene_analyzer.analyze_scene(image, use_prompts=use_prompts)

            # Prepara output
            annotated_image = results.get('annotated_image', image)

            # Crea JSON dei risultati
            results_for_json = {
                'detected_characters': results['detected_characters'],
                'analysis_summary': results['analysis_summary']
            }
            results_json = json.dumps(convert_numpy_types(results_for_json), indent=2, ensure_ascii=False)

            # Crea riassunto testuale
            summary = self._create_analysis_summary(results)

            return annotated_image, results_json, summary

        except Exception as e:
            error_msg = f"❌ Errore durante l'analisi: {str(e)}"
            return image, error_msg, error_msg

    def query_database_interface(self, query_image: np.ndarray,
                                top_k: int = 5) -> Tuple[List[np.ndarray], str]:
        """
        Interfaccia per la ricerca nel database.

        Args:
            query_image: Immagine di query
            top_k: Numero di risultati da mostrare

        Returns:
            Tupla con (lista_immagini_simili, risultati_json)
        """
        if query_image is None:
            return [], "❌ Nessuna immagine caricata"

        try:
            # Cerca immagini simili
            results = self.scene_analyzer.query_database(query_image, top_k=top_k)

            if not results:
                return [], "❌ Nessun risultato trovato o database non caricato"

            # Estrai immagini e metadati
            similar_images = []
            results_data = []

            for result in results:
                if 'image' in result:
                    similar_images.append(result['image'])

                # Prepara metadati per JSON
                result_data = {
                    'character': result.get('character', 'Sconosciuto'),
                    'similarity_score': result.get('similarity_score', 0.0),
                    'path': result.get('path', ''),
                    'split': result.get('split', '')
                }
                results_data.append(result_data)

            # Crea JSON dei risultati
            results_json = json.dumps({
                'query_results': results_data,
                'total_results': len(results_data)
            }, indent=2, ensure_ascii=False)

            return similar_images, results_json

        except Exception as e:
            error_msg = f"❌ Errore durante la ricerca: {str(e)}"
            return [], error_msg

    def build_index_interface(self, dataset_path: str) -> str:
        """
        Interfaccia per costruire l'indice del dataset.

        Args:
            dataset_path: Percorso del dataset

        Returns:
            Messaggio di stato
        """
        if not dataset_path:
            return "❌ Inserisci il percorso del dataset"

        if not os.path.exists(dataset_path):
            return f"❌ Percorso non trovato: {dataset_path}"

        try:
            # Crea indicizzatore
            indexer = DatasetIndexer(self.config)

            # Costruisci indice
            indexer.build_full_index(dataset_path, use_faiss=True)

            # Ricarica il vector database nell'analizzatore
            vector_db_config = self.config.get('vector_db', {})
            index_path = vector_db_config.get('index_path', 'data/vector_db/image_index.faiss')
            metadata_path = vector_db_config.get('metadata_path', 'data/vector_db/image_metadata.json')

            self.scene_analyzer.load_vector_database(index_path, metadata_path)

            return "✅ Indice costruito con successo! Il database è ora disponibile per le ricerche."

        except Exception as e:
            return f"❌ Errore durante la costruzione dell'indice: {str(e)}"

    def finetune_clip_interface(self, dataset_path: str, num_epochs: int = 5) -> str:
        """
        Interfaccia per il fine-tuning di CLIP.

        Args:
            dataset_path: Percorso del dataset
            num_epochs: Numero di epoche di training

        Returns:
            Messaggio di stato
        """
        if not dataset_path:
            return "❌ Inserisci il percorso del dataset"

        if not os.path.exists(dataset_path):
            return f"❌ Percorso non trovato: {dataset_path}"

        try:
            # Carica dataset
            indexer = DatasetIndexer(self.config)
            dataset = indexer.load_dataset(dataset_path)

            if len(dataset) == 0:
                return "❌ Dataset vuoto o non valido"

            # Dividi dataset
            train_set, val_set, test_set = split_dataset(dataset)

            # Aggiorna configurazione con le epoche richieste
            self.config['training']['num_epochs'] = num_epochs

            # Crea fine-tuner
            fine_tuner = CLIPFineTuner(self.config)

            # Avvia training
            fine_tuner.train(train_set, val_set)

            return f"✅ Fine-tuning completato! Modello salvato in: {fine_tuner.checkpoint_dir}"

        except Exception as e:
            return f"❌ Errore durante il fine-tuning: {str(e)}"

    def _create_analysis_summary(self, results: Dict[str, Any]) -> str:
        """
        Crea un riassunto testuale dei risultati dell'analisi.

        Args:
            results: Risultati dell'analisi

        Returns:
            Riassunto testuale
        """
        summary = results.get('analysis_summary', {})
        characters = results.get('detected_characters', [])

        # Intestazione
        text = "📊 **RIASSUNTO ANALISI SCENA**\n\n"

        # Statistiche generali
        text += f"🔍 **Oggetti rilevati:** {summary.get('total_objects_detected', 0)}\n"
        text += f"👥 **Personaggi identificati:** {summary.get('characters_identified', 0)}\n"
        text += f"❓ **Oggetti sconosciuti:** {summary.get('unknown_objects', 0)}\n"
        text += f"📈 **Confidenza media:** {summary.get('average_confidence', 0.0):.2f}\n\n"

        # Personaggi unici trovati
        unique_chars = summary.get('unique_characters', [])
        if unique_chars:
            text += "🎭 **Personaggi riconosciuti:**\n"
            for char in unique_chars:
                text += f"  • {char}\n"
            text += "\n"

        # Dettagli per ogni personaggio identificato
        if characters:
            text += "📋 **Dettagli identificazioni:**\n\n"
            for i, char in enumerate(characters, 1):
                name = char.get('character_name', 'Sconosciuto')
                confidence = char.get('confidence', 0.0)
                area = char.get('area', 0)

                text += f"**{i}. {name}**\n"
                text += f"   Confidenza: {confidence:.3f}\n"
                text += f"   Area: {area} pixel\n"

                if confidence > 0.8:
                    text += "   ✅ Identificazione molto sicura\n"
                elif confidence > 0.6:
                    text += "   ⚠️ Identificazione probabile\n"
                else:
                    text += "   ❓ Identificazione incerta\n"

                text += "\n"

        return text

    def create_interface(self):
        """
        Crea l'interfaccia Gradio completa.

        Returns:
            Interfaccia Gradio
        """
        with gr.Blocks(
            title=self.title,
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {font-family: 'Arial', sans-serif;}
            .tab-nav button {font-weight: bold;}
            .output-image {border-radius: 10px;}
            """
        ) as interface:

            # Titolo principale
            gr.Markdown(f"""
            # 🎭 {self.title}
            
            Sistema di analisi automatica delle scene per l'identificazione dei personaggi di Naruto.
            Utilizza **SAM** per la segmentazione e **CLIP** per il riconoscimento.
            """)

            with gr.Tabs():
                # TAB 1: Analisi delle Scene
                with gr.Tab("🔍 Analisi Scene", id="scene_analysis"):
                    gr.Markdown("""
                    ### Carica un'immagine con più personaggi per identificarli automaticamente
                    Il sistema segmenterà l'immagine e identificherà ogni personaggio presente.
                    """)

                    with gr.Row():
                        with gr.Column(scale=1):
                            scene_input = gr.Image(
                                label="Immagine della Scena",
                                type="numpy",
                                height=400
                            )

                            use_prompts = gr.Checkbox(
                                label="Usa segmentazione guidata (sperimentale)",
                                value=False
                            )

                            analyze_btn = gr.Button(
                                "🚀 Analizza Scena",
                                variant="primary",
                                size="lg"
                            )

                        with gr.Column(scale=1):
                            scene_output = gr.Image(
                                label="Risultato Annotato",
                                type="numpy",
                                height=400
                            )

                    with gr.Row():
                        with gr.Column():
                            analysis_summary = gr.Textbox(
                                label="📊 Riassunto Analisi",
                                lines=10,
                                max_lines=15
                            )

                        with gr.Column():
                            results_json = gr.Textbox(
                                label="📋 Risultati Dettagliati (JSON)",
                                lines=10,
                                max_lines=15
                            )

                    # Event handler per analisi scene
                    analyze_btn.click(
                        fn=self.analyze_scene_interface,
                        inputs=[scene_input, use_prompts],
                        outputs=[scene_output, results_json, analysis_summary]
                    )

                # TAB 2: Ricerca nel Database
                with gr.Tab("📚 Ricerca Database", id="database_query"):
                    gr.Markdown("""
                    ### Cerca personaggi simili nel database
                    Carica un'immagine di un personaggio per trovare immagini simili nel database.
                    """)

                    with gr.Row():
                        with gr.Column(scale=1):
                            query_input = gr.Image(
                                label="Immagine Query",
                                type="numpy",
                                height=300
                            )

                            top_k_slider = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1,
                                label="Numero di risultati da mostrare"
                            )

                            search_btn = gr.Button(
                                "🔎 Cerca Simili",
                                variant="primary"
                            )

                        with gr.Column(scale=2):
                            search_results = gr.Gallery(
                                label="Risultati Simili",
                                columns=3,
                                rows=2,
                                height=400
                            )

                    search_results_json = gr.Textbox(
                        label="📋 Dettagli Risultati",
                        lines=8,
                        max_lines=10
                    )

                    # Event handler per ricerca database
                    search_btn.click(
                        fn=self.query_database_interface,
                        inputs=[query_input, top_k_slider],
                        outputs=[search_results, search_results_json]
                    )

                # TAB 3: Gestione Sistema
                with gr.Tab("⚙️ Gestione Sistema", id="system_management"):
                    gr.Markdown("""
                    ### Configurazione e manutenzione del sistema
                    Funzioni per costruire l'indice del dataset e fare fine-tuning del modello.
                    """)

                    with gr.Accordion("🏗️ Costruzione Indice Dataset", open=False):
                        gr.Markdown("""
                        Costruisci l'indice vettoriale dal dataset di immagini.
                        **Nota:** Questa operazione può richiedere diversi minuti.
                        """)

                        dataset_path_input = gr.Textbox(
                            label="Percorso Dataset",
                            placeholder="data/raw/Anime-Naruto",
                            value="data/raw/Anime-Naruto"
                        )

                        build_index_btn = gr.Button(
                            "🏗️ Costruisci Indice",
                            variant="secondary"
                        )

                        index_status = gr.Textbox(
                            label="Stato Costruzione",
                            lines=3
                        )

                        build_index_btn.click(
                            fn=self.build_index_interface,
                            inputs=[dataset_path_input],
                            outputs=[index_status]
                        )

                    with gr.Accordion("🎯 Fine-tuning CLIP", open=False):
                        gr.Markdown("""
                        Esegui il fine-tuning del modello CLIP sui personaggi di Naruto.
                        **Attenzione:** Questa operazione è molto intensiva e può richiedere ore.
                        """)

                        finetune_dataset_path = gr.Textbox(
                            label="Percorso Dataset per Fine-tuning",
                            placeholder="data/raw/Anime-Naruto",
                            value="data/raw/Anime-Naruto"
                        )

                        epochs_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Numero di Epoche"
                        )

                        finetune_btn = gr.Button(
                            "🎯 Avvia Fine-tuning",
                            variant="secondary"
                        )

                        finetune_status = gr.Textbox(
                            label="Stato Fine-tuning",
                            lines=3
                        )

                        finetune_btn.click(
                            fn=self.finetune_clip_interface,
                            inputs=[finetune_dataset_path, epochs_slider],
                            outputs=[finetune_status]
                        )

                    with gr.Accordion("ℹ️ Informazioni Sistema", open=True):
                        system_info = self._get_system_info()
                        gr.Markdown(system_info)

            # Footer
            gr.Markdown("""
            ---
            *Sistema di Scene Analysis - Progetto Deep Learning*  
            Tecnologie: SAM, CLIP, FAISS, Gradio
            """)

        return interface

    def _get_system_info(self) -> str:
        """
        Ottiene informazioni di stato del sistema.

        Returns:
            Informazioni formattate
        """
        info = "### 📊 Stato del Sistema\n\n"

        # Controlla vector database
        if self.scene_analyzer.vector_index is not None:
            db_size = len(self.scene_analyzer.metadata)
            info += f"✅ **Vector Database:** Caricato ({db_size} immagini)\n"
        else:
            info += "❌ **Vector Database:** Non caricato\n"

        # Controlla modello CLIP
        info += f"✅ **Modello CLIP:** Caricato\n"
        info += f"🖥️ **Device:** {self.scene_analyzer.device}\n"

        # Configurazione
        info += f"🔧 **Soglia similarità:** {self.scene_analyzer.similarity_threshold}\n"
        info += f"📏 **Top-K ricerca:** {self.scene_analyzer.top_k}\n"

        return info

    def launch(self):
        """
        Avvia l'applicazione Gradio.
        """
        interface = self.create_interface()

        print(f"🚀 Avvio applicazione Gradio...")
        print(f"📱 Porta: {self.port}")
        print(f"🌐 Condivisione: {self.share}")

        interface.launch(
            server_port=self.port,
            share=self.share,
            debug=self.debug,
            show_error=True
        )
