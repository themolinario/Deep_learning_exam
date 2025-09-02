"""
Sistema di valutazione delle performance per CLIP Scene Search.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import time
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

from src.pipelines.index_dataset import DatasetIndexer
from src.pipelines.segment_and_search import SemanticSearchPipeline


class PerformanceEvaluator:
    """
    Classe per la valutazione quantitativa e qualitativa del sistema.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.indexer = DatasetIndexer(config_path)
        self.pipeline = SemanticSearchPipeline(config_path)
        self.results = {}

    def evaluate_search_accuracy(self, test_queries: List[Dict]) -> Dict[str, float]:
        """
        Valuta l'accuratezza della ricerca semantica.

        Args:
            test_queries: Lista di query di test con ground truth
                Format: [{"query": "naruto", "expected_characters": ["naruto", "uzumaki"]}]

        Returns:
            Dizionario con metriche di accuratezza
        """
        print("📊 Valutazione accuratezza ricerca...")

        precision_scores = []
        recall_scores = []
        map_scores = []

        for query_data in test_queries:
            query = query_data["query"]
            expected = set(query_data["expected_characters"])

            # Esegui ricerca
            results = self.indexer.search_similar_images(query, top_k=10)

            if not results:
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                map_scores.append(0.0)
                continue

            # Estrai caratteri predetti (da path o metadata)
            predicted = set()
            relevance_scores = []

            for result in results:
                # Estrai nome carattere dal path dell'immagine
                img_path = result['metadata']['path']
                character_name = self._extract_character_from_path(img_path)
                predicted.add(character_name)

                # Score binario di rilevanza
                is_relevant = 1 if character_name in expected else 0
                relevance_scores.append(is_relevant)

            # Calcola metriche
            true_positives = len(predicted.intersection(expected))
            precision = true_positives / len(predicted) if predicted else 0
            recall = true_positives / len(expected) if expected else 0

            precision_scores.append(precision)
            recall_scores.append(recall)

            # Mean Average Precision
            if sum(relevance_scores) > 0:
                ap = average_precision_score([1 if char in expected else 0 for char in predicted],
                                           [r['similarity'] for r in results[:len(predicted)]])
                map_scores.append(ap)
            else:
                map_scores.append(0.0)

        metrics = {
            "precision": np.mean(precision_scores),
            "recall": np.mean(recall_scores),
            "f1_score": 2 * np.mean(precision_scores) * np.mean(recall_scores) /
                      (np.mean(precision_scores) + np.mean(recall_scores)) if (np.mean(precision_scores) + np.mean(recall_scores)) > 0 else 0,
            "map": np.mean(map_scores)
        }

        return metrics

    def evaluate_segmentation_performance(self, test_images: List[str],
                                        methods: List[str] = ["grid", "kmeans", "superpixel"]) -> Dict[str, Dict]:
        """
        Valuta le performance dei diversi metodi di segmentazione.

        Args:
            test_images: Lista di path di immagini di test
            methods: Metodi di segmentazione da testare

        Returns:
            Risultati per ogni metodo
        """
        print("🔍 Valutazione performance segmentazione...")

        results = {}

        for method in methods:
            print(f"  Testando metodo: {method}")

            processing_times = []
            segment_counts = []
            coverage_scores = []

            for img_path in test_images:
                if not os.path.exists(img_path):
                    continue

                start_time = time.time()

                try:
                    # Esegui segmentazione
                    if method == "grid":
                        results_data, mask = self.pipeline.search_in_segments(
                            img_path, "test query", method=method, grid_size=(4, 4)
                        )
                    elif method == "kmeans":
                        results_data, mask = self.pipeline.search_in_segments(
                            img_path, "test query", method=method, n_clusters=8
                        )
                    else:  # superpixel
                        results_data, mask = self.pipeline.search_in_segments(
                            img_path, "test query", method=method, n_segments=100
                        )

                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)

                    # Conta segmenti
                    if results_data:
                        segment_counts.append(len(results_data))

                        # Calcola copertura dell'immagine
                        total_area = 0
                        for result in results_data:
                            bbox = result['segment_info']['bbox']
                            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            total_area += area

                        # Assumendo immagine 224x224 (da config)
                        image_area = 224 * 224
                        coverage = min(total_area / image_area, 1.0)
                        coverage_scores.append(coverage)
                    else:
                        segment_counts.append(0)
                        coverage_scores.append(0.0)

                except Exception as e:
                    print(f"    Errore con {img_path}: {e}")
                    processing_times.append(float('inf'))
                    segment_counts.append(0)
                    coverage_scores.append(0.0)

            results[method] = {
                "avg_processing_time": np.mean(processing_times),
                "avg_segments": np.mean(segment_counts),
                "avg_coverage": np.mean(coverage_scores),
                "success_rate": sum(1 for t in processing_times if t != float('inf')) / len(processing_times)
            }

        return results

    def evaluate_embedding_quality(self, sample_size: int = 100) -> Dict[str, float]:
        """
        Valuta la qualità degli embedding CLIP.

        Args:
            sample_size: Numero di campioni da testare

        Returns:
            Metriche di qualità degli embedding
        """
        print("🧠 Valutazione qualità embedding...")

        if not hasattr(self.indexer, 'image_metadata') or not self.indexer.image_metadata:
            try:
                self.indexer.load_index()
            except:
                return {"error": "Impossibile caricare database vettoriale"}

        if not self.indexer.index or self.indexer.index.ntotal == 0:
            return {"error": "Database vettoriale vuoto"}

        # Prendi un campione di indici casuali
        total_vectors = self.indexer.index.ntotal
        sample_indices = np.random.choice(
            total_vectors,
            min(sample_size, total_vectors),
            replace=False
        )

        similarities = []
        intra_class_similarities = []
        inter_class_similarities = []

        # Estrai gli embedding dal database FAISS
        embeddings = []
        characters = []

        for idx in sample_indices:
            # Estrai embedding dall'indice FAISS
            vector = self.indexer.index.reconstruct(int(idx))
            embeddings.append(vector)

            # Estrai il carattere dal path del metadata corrispondente
            if idx < len(self.indexer.image_metadata):
                char = self._extract_character_from_path(self.indexer.image_metadata[idx]['path'])
                characters.append(char)
            else:
                characters.append("unknown")

        # Calcola similarità tra coppie di embedding
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                char1, char2 = characters[i], characters[j]

                # Calcola similarità coseno
                emb1 = embeddings[i].reshape(1, -1)
                emb2 = embeddings[j].reshape(1, -1)
                sim = cosine_similarity(emb1, emb2)[0][0]

                similarities.append(sim)

                if char1 == char2 and char1 != "unknown":
                    intra_class_similarities.append(sim)
                elif char1 != "unknown" and char2 != "unknown":
                    inter_class_similarities.append(sim)

        # Calcola metriche
        metrics = {
            "avg_similarity": np.mean(similarities) if similarities else 0,
            "intra_class_similarity": np.mean(intra_class_similarities) if intra_class_similarities else 0,
            "inter_class_similarity": np.mean(inter_class_similarities) if inter_class_similarities else 0,
            "embedding_quality_score": (np.mean(intra_class_similarities) - np.mean(inter_class_similarities))
                                     if intra_class_similarities and inter_class_similarities else 0,
            "total_comparisons": len(similarities),
            "intra_class_comparisons": len(intra_class_similarities),
            "inter_class_comparisons": len(inter_class_similarities)
        }

        return metrics

    def generate_comprehensive_report(self, save_path: str = "reports/performance_report.html"):
        """
        Genera un report completo delle performance.
        """
        print("📋 Generazione report completo...")

        # Esegui tutte le valutazioni
        test_queries = self._create_test_queries()
        test_images = self._get_test_images()

        search_metrics = self.evaluate_search_accuracy(test_queries)
        segmentation_metrics = self.evaluate_segmentation_performance(test_images)
        embedding_metrics = self.evaluate_embedding_quality()

        # Crea visualizzazioni
        self._create_performance_plots(search_metrics, segmentation_metrics, embedding_metrics)

        # Genera report HTML
        html_report = self._generate_html_report(search_metrics, segmentation_metrics, embedding_metrics)

        # Salva report
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_report)

        print(f"✅ Report salvato in: {save_path}")
        return save_path

    def _extract_character_from_path(self, path: str) -> str:
        """Estrae il nome del carattere dal path dell'immagine."""
        # Logica semplificata - da adattare al formato del dataset
        filename = os.path.basename(path).lower()

        # Caratteri comuni nel dataset Naruto
        characters = ["naruto", "sasuke", "sakura", "kakashi", "hinata", "shikamaru", "gaara"]

        for char in characters:
            if char in filename:
                return char

        return "unknown"

    def _create_test_queries(self) -> List[Dict]:
        """Crea query di test standard."""
        return [
            {"query": "naruto uzumaki", "expected_characters": ["naruto"]},
            {"query": "sasuke uchiha", "expected_characters": ["sasuke"]},
            {"query": "pink hair ninja", "expected_characters": ["sakura"]},
            {"query": "copy ninja", "expected_characters": ["kakashi"]},
            {"query": "byakugan user", "expected_characters": ["hinata"]},
            {"query": "shadow manipulation", "expected_characters": ["shikamaru"]},
            {"query": "sand ninja", "expected_characters": ["gaara"]}
        ]

    def _get_test_images(self, max_images: int = 20) -> List[str]:
        """Ottiene lista di immagini di test."""
        test_dir = "data/raw/Anime-Naruto/test/"
        if not os.path.exists(test_dir):
            return []

        images = []
        for file in os.listdir(test_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(test_dir, file))
                if len(images) >= max_images:
                    break

        return images

    def _create_performance_plots(self, search_metrics, segmentation_metrics, embedding_metrics):
        """Crea grafici delle performance."""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Metriche di ricerca
        metrics_names = list(search_metrics.keys())
        metrics_values = list(search_metrics.values())

        axes[0, 0].bar(metrics_names, metrics_values, color='skyblue')
        axes[0, 0].set_title('Metriche di Ricerca Semantica')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)

        # 2. Performance segmentazione
        methods = list(segmentation_metrics.keys())
        times = [segmentation_metrics[m]['avg_processing_time'] for m in methods]

        axes[0, 1].bar(methods, times, color='lightcoral')
        axes[0, 1].set_title('Tempi di Segmentazione')
        axes[0, 1].set_ylabel('Tempo (secondi)')

        # 3. Qualità embedding
        emb_metrics = ['intra_class_similarity', 'inter_class_similarity', 'embedding_quality_score']
        emb_values = [embedding_metrics.get(m, 0) for m in emb_metrics]

        axes[1, 0].bar(emb_metrics, emb_values, color='lightgreen')
        axes[1, 0].set_title('Qualità Embedding')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Coverage segmentazione
        coverage_scores = [segmentation_metrics[m]['avg_coverage'] for m in methods]

        axes[1, 1].bar(methods, coverage_scores, color='orange')
        axes[1, 1].set_title('Copertura Media Segmentazione')
        axes[1, 1].set_ylabel('Coverage Score')
        axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig('reports/figures/performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_html_report(self, search_metrics, segmentation_metrics, embedding_metrics) -> str:
        """Genera report HTML completo."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CLIP Scene Search - Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                .metric {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .good {{ border-left: 5px solid #4CAF50; }}
                .warning {{ border-left: 5px solid #FF9800; }}
                .poor {{ border-left: 5px solid #F44336; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>🎯 CLIP Scene Search - Report delle Performance</h1>
            <p>Generato il: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>📊 Metriche di Ricerca Semantica</h2>
            <div class="metric {'good' if search_metrics.get('precision', 0) > 0.7 else 'warning' if search_metrics.get('precision', 0) > 0.5 else 'poor'}">
                <strong>Precision:</strong> {search_metrics.get('precision', 0):.3f}
            </div>
            <div class="metric {'good' if search_metrics.get('recall', 0) > 0.7 else 'warning' if search_metrics.get('recall', 0) > 0.5 else 'poor'}">
                <strong>Recall:</strong> {search_metrics.get('recall', 0):.3f}
            </div>
            <div class="metric {'good' if search_metrics.get('f1_score', 0) > 0.7 else 'warning' if search_metrics.get('f1_score', 0) > 0.5 else 'poor'}">
                <strong>F1-Score:</strong> {search_metrics.get('f1_score', 0):.3f}
            </div>
            <div class="metric {'good' if search_metrics.get('map', 0) > 0.7 else 'warning' if search_metrics.get('map', 0) > 0.5 else 'poor'}">
                <strong>Mean Average Precision:</strong> {search_metrics.get('map', 0):.3f}
            </div>
            
            <h2>🔍 Performance Segmentazione</h2>
            <table>
                <tr>
                    <th>Metodo</th>
                    <th>Tempo Medio (s)</th>
                    <th>Segmenti Medi</th>
                    <th>Copertura Media</th>
                    <th>Tasso di Successo</th>
                </tr>
        """

        for method, metrics in segmentation_metrics.items():
            html += f"""
                <tr>
                    <td>{method}</td>
                    <td>{metrics['avg_processing_time']:.3f}</td>
                    <td>{metrics['avg_segments']:.1f}</td>
                    <td>{metrics['avg_coverage']:.3f}</td>
                    <td>{metrics['success_rate']:.3f}</td>
                </tr>
            """

        html += f"""
            </table>
            
            <h2>🧠 Qualità Embedding</h2>
            <div class="metric">
                <strong>Similarità Intra-Classe:</strong> {embedding_metrics.get('intra_class_similarity', 0):.3f}
            </div>
            <div class="metric">
                <strong>Similarità Inter-Classe:</strong> {embedding_metrics.get('inter_class_similarity', 0):.3f}
            </div>
            <div class="metric {'good' if embedding_metrics.get('embedding_quality_score', 0) > 0.3 else 'warning' if embedding_metrics.get('embedding_quality_score', 0) > 0.1 else 'poor'}">
                <strong>Score Qualità Embedding:</strong> {embedding_metrics.get('embedding_quality_score', 0):.3f}
            </div>
            
            <h2>📈 Grafici delle Performance</h2>
            <img src="figures/performance_metrics.png" alt="Performance Metrics" style="max-width: 100%; height: auto;">
            
            <h2>📝 Raccomandazioni</h2>
            <ul>
        """

        # Aggiungi raccomandazioni basate sui risultati
        if search_metrics.get('precision', 0) < 0.6:
            html += "<li>❗ Precision bassa: considera il fine-tuning del modello CLIP</li>"

        if embedding_metrics.get('embedding_quality_score', 0) < 0.2:
            html += "<li>❗ Qualità embedding scarsa: rivedi il preprocessing delle immagini</li>"

        best_method = max(segmentation_metrics.keys(),
                         key=lambda x: segmentation_metrics[x]['success_rate'])
        html += f"<li>✅ Metodo di segmentazione migliore: {best_method}</li>"

        html += """
            </ul>
        </body>
        </html>
        """

        return html


def main():
    """Esegui valutazione completa del sistema."""
    evaluator = PerformanceEvaluator()
    report_path = evaluator.generate_comprehensive_report()
    print(f"📋 Report completo generato: {report_path}")


if __name__ == "__main__":
    main()
