"""
Pipeline per la valutazione delle performance del sistema di Scene Analysis.
"""

import os
import numpy as np
from PIL import Image
import json
import csv
from typing import List, Dict, Any, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import defaultdict
import time

from .segment_and_search import SceneAnalyzer


class PerformanceEvaluator:
    """
    Classe per valutare le performance del sistema di Scene Analysis.
    """

    def __init__(self, config: Dict[str, Any], scene_analyzer: SceneAnalyzer):
        """
        Inizializza il valutatore di performance.

        Args:
            config: Configurazione del progetto
            scene_analyzer: Analizzatore di scene configurato
        """
        self.config = config
        self.scene_analyzer = scene_analyzer

        # Configurazione valutazione
        eval_config = config.get('evaluation', {})
        self.test_scenes_path = eval_config.get('test_scenes_path', 'data/test_scenes')
        self.metrics = eval_config.get('metrics', ['accuracy', 'precision', 'recall', 'f1'])
        self.output_dir = eval_config.get('output_dir', 'reports')

        # Crea directory output se non esiste
        os.makedirs(self.output_dir, exist_ok=True)

        # Risultati valutazione
        self.evaluation_results = {}
        self.detailed_results = []

    def load_test_dataset(self, test_path: str) -> List[Dict[str, Any]]:
        """
        Carica il dataset di test con ground truth.

        Args:
            test_path: Percorso del dataset di test

        Returns:
            Lista di scene di test con annotazioni
        """
        test_dataset = []
        test_path = Path(test_path)

        if not test_path.exists():
            print(f"⚠️ Percorso test non trovato: {test_path}")
            return []

        # Cerca file di annotazioni
        annotations_file = test_path / 'annotations.json'
        if annotations_file.exists():
            # Carica annotazioni da JSON
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)

            for item in annotations:
                image_path = test_path / item['image_path']
                if image_path.exists():
                    test_dataset.append({
                        'image_path': str(image_path),
                        'ground_truth': item['characters'],
                        'scene_id': item.get('scene_id', image_path.stem)
                    })
        else:
            # Fallback: cerca immagini e file CSV di annotazioni
            for image_file in test_path.glob('*.jpg'):
                annotation_file = image_file.with_suffix('.csv')
                if annotation_file.exists():
                    # Leggi annotazioni da CSV
                    characters = []
                    with open(annotation_file, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            characters.append({
                                'character': row['character'],
                                'bbox': [int(row['x1']), int(row['y1'],
                                        int(row['x2']), int(row['y2'])]
                            })

                    test_dataset.append({
                        'image_path': str(image_file),
                        'ground_truth': characters,
                        'scene_id': image_file.stem
                    })

        print(f"Caricato dataset di test: {len(test_dataset)} scene")
        return test_dataset

    def evaluate_scene_analysis(self, test_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Valuta le performance dell'analisi delle scene.

        Args:
            test_dataset: Dataset di test con ground truth

        Returns:
            Dizionario con metriche di performance
        """
        print("🔍 Inizio valutazione scene analysis...")

        total_scenes = len(test_dataset)
        correct_identifications = 0
        total_characters = 0
        total_predictions = 0

        y_true = []  # Ground truth
        y_pred = []  # Predizioni

        scene_results = []
        processing_times = []

        for i, scene_data in enumerate(test_dataset):
            print(f"Valutazione scena {i+1}/{total_scenes}: {scene_data['scene_id']}")

            try:
                # Carica immagine
                image = np.array(Image.open(scene_data['image_path']).convert('RGB'))

                # Misura tempo di processing
                start_time = time.time()

                # Analizza scena
                results = self.scene_analyzer.analyze_scene(image)

                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                # Estrai ground truth
                gt_characters = scene_data['ground_truth']
                gt_names = [char['character'] for char in gt_characters]

                # Estrai predizioni
                predicted_chars = results['detected_characters']
                pred_names = [char['character_name'] for char in predicted_chars
                             if char['character_name'] != 'Sconosciuto']

                # Calcola metriche per questa scena
                scene_metrics = self._calculate_scene_metrics(gt_names, pred_names)

                scene_result = {
                    'scene_id': scene_data['scene_id'],
                    'ground_truth': gt_names,
                    'predictions': pred_names,
                    'metrics': scene_metrics,
                    'processing_time': processing_time,
                    'num_objects_detected': len(predicted_chars),
                    'analysis_summary': results['analysis_summary']
                }

                scene_results.append(scene_result)

                # Aggiorna contatori globali
                total_characters += len(gt_names)
                total_predictions += len(pred_names)

                # Per metriche globali, usiamo character-level matching
                for gt_char in gt_names:
                    y_true.append(gt_char)
                    if gt_char in pred_names:
                        y_pred.append(gt_char)
                        correct_identifications += 1
                    else:
                        y_pred.append('NOT_FOUND')

                # Aggiungi false positives
                for pred_char in pred_names:
                    if pred_char not in gt_names:
                        y_true.append('FALSE_POSITIVE')
                        y_pred.append(pred_char)

            except Exception as e:
                print(f"❌ Errore valutando scena {scene_data['scene_id']}: {e}")
                continue

        # Calcola metriche globali
        global_metrics = self._calculate_global_metrics(y_true, y_pred)

        # Statistiche aggiuntive
        avg_processing_time = np.mean(processing_times) if processing_times else 0

        evaluation_results = {
            'total_scenes_evaluated': len(scene_results),
            'total_characters_gt': total_characters,
            'total_predictions': total_predictions,
            'correct_identifications': correct_identifications,
            'global_metrics': global_metrics,
            'average_processing_time': avg_processing_time,
            'scene_results': scene_results
        }

        self.evaluation_results = evaluation_results
        self.detailed_results = scene_results

        print(f"✅ Valutazione completata: {len(scene_results)} scene processate")
        return evaluation_results

    def _calculate_scene_metrics(self, gt_names: List[str], pred_names: List[str]) -> Dict[str, float]:
        """
        Calcola metriche per una singola scena.

        Args:
            gt_names: Nomi dei personaggi ground truth
            pred_names: Nomi dei personaggi predetti

        Returns:
            Dizionario con metriche della scena
        """
        gt_set = set(gt_names)
        pred_set = set(pred_names)

        # True positives: personaggi correttamente identificati
        tp = len(gt_set.intersection(pred_set))

        # False positives: personaggi identificati ma non presenti
        fp = len(pred_set - gt_set)

        # False negatives: personaggi presenti ma non identificati
        fn = len(gt_set - pred_set)

        # Calcola metriche
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / len(gt_set.union(pred_set)) if len(gt_set.union(pred_set)) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }

    def _calculate_global_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        """
        Calcola metriche globali su tutto il dataset.

        Args:
            y_true: Labels ground truth
            y_pred: Labels predette

        Returns:
            Dizionario con metriche globali
        """
        # Ottieni nomi unici dei personaggi
        unique_characters = list(set(y_true + y_pred))
        unique_characters = [char for char in unique_characters
                           if char not in ['NOT_FOUND', 'FALSE_POSITIVE']]

        # Calcola metriche macro-averaged
        try:
            precision = precision_score(y_true, y_pred, labels=unique_characters,
                                      average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, labels=unique_characters,
                                average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, labels=unique_characters,
                         average='macro', zero_division=0)

            # Accuracy personalizzata (escludendo false positives)
            filtered_pairs = [(true, pred) for true, pred in zip(y_true, y_pred)
                            if true != 'FALSE_POSITIVE']
            if filtered_pairs:
                true_filtered, pred_filtered = zip(*filtered_pairs)
                accuracy = accuracy_score(true_filtered, pred_filtered)
            else:
                accuracy = 0.0

        except Exception as e:
            print(f"⚠️ Errore calcolando metriche globali: {e}")
            precision = recall = f1 = accuracy = 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy
        }

    def create_confusion_matrix(self, save_path: str = None) -> np.ndarray:
        """
        Crea la matrice di confusione per i risultati.

        Args:
            save_path: Percorso dove salvare la figura

        Returns:
            Matrice di confusione
        """
        if not self.detailed_results:
            print("❌ Nessun risultato disponibile per la matrice di confusione")
            return np.array([])

        # Raccogli tutte le predizioni e ground truth
        all_gt = []
        all_pred = []

        for scene in self.detailed_results:
            gt_chars = scene['ground_truth']
            pred_chars = scene['predictions']

            # Aggiungi character-level matches
            for gt_char in gt_chars:
                all_gt.append(gt_char)
                if gt_char in pred_chars:
                    all_pred.append(gt_char)
                else:
                    all_pred.append('NOT_FOUND')

        # Ottieni labels unici
        all_characters = sorted(list(set(all_gt)))
        all_characters.append('NOT_FOUND')

        # Calcola matrice di confusione
        cm = confusion_matrix(all_gt, all_pred, labels=all_characters)

        # Crea visualizzazione
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=all_characters, yticklabels=all_characters[:-1])
        plt.title('Matrice di Confusione - Identificazione Personaggi')
        plt.xlabel('Predizioni')
        plt.ylabel('Ground Truth')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Matrice di confusione salvata: {save_path}")

        plt.show()
        return cm

    def generate_performance_report(self) -> str:
        """
        Genera un report completo delle performance.

        Returns:
            Percorso del file di report HTML generato
        """
        if not self.evaluation_results:
            print("❌ Nessun risultato di valutazione disponibile")
            return ""

        # Crea report HTML
        html_content = self._create_html_report()

        # Salva report
        report_path = os.path.join(self.output_dir, 'performance_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Salva anche dati JSON
        json_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)

        # Crea grafici
        self._create_performance_plots()

        print(f"✅ Report di performance generato: {report_path}")
        return report_path

    def _create_html_report(self) -> str:
        """
        Crea il contenuto HTML del report di performance.

        Returns:
            Contenuto HTML del report
        """
        results = self.evaluation_results
        global_metrics = results['global_metrics']

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Report Performance - Sistema Scene Analysis</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .metric-box {{ 
                    background-color: #e8f4fd; 
                    padding: 15px; 
                    border-radius: 8px; 
                    text-align: center; 
                    min-width: 150px;
                }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2563eb; }}
                .scene-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .scene-table th, .scene-table td {{ 
                    border: 1px solid #ddd; 
                    padding: 8px; 
                    text-align: left; 
                }}
                .scene-table th {{ background-color: #f2f2f2; }}
                .good {{ color: #16a34a; }}
                .warning {{ color: #ea580c; }}
                .error {{ color: #dc2626; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎭 Report Performance - Sistema Scene Analysis</h1>
                <p>Report generato automaticamente per valutare le performance del sistema 
                   di identificazione dei personaggi di Naruto.</p>
            </div>
            
            <h2>📊 Metriche Globali</h2>
            <div class="metrics">
                <div class="metric-box">
                    <div class="metric-value">{global_metrics['accuracy']:.3f}</div>
                    <div>Accuracy</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{global_metrics['precision']:.3f}</div>
                    <div>Precision</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{global_metrics['recall']:.3f}</div>
                    <div>Recall</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">{global_metrics['f1_score']:.3f}</div>
                    <div>F1-Score</div>
                </div>
            </div>
            
            <h2>📈 Statistiche Generali</h2>
            <ul>
                <li><strong>Scene valutate:</strong> {results['total_scenes_evaluated']}</li>
                <li><strong>Personaggi totali (GT):</strong> {results['total_characters_gt']}</li>
                <li><strong>Predizioni totali:</strong> {results['total_predictions']}</li>
                <li><strong>Identificazioni corrette:</strong> {results['correct_identifications']}</li>
                <li><strong>Tempo medio di elaborazione:</strong> {results['average_processing_time']:.2f}s</li>
            </ul>
            
            <h2>🔍 Risultati per Scena</h2>
            <table class="scene-table">
                <thead>
                    <tr>
                        <th>Scena</th>
                        <th>Ground Truth</th>
                        <th>Predizioni</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Tempo (s)</th>
                    </tr>
                </thead>
                <tbody>
        """

        # Aggiungi righe per ogni scena
        for scene in self.detailed_results:
            metrics = scene['metrics']
            gt_str = ', '.join(scene['ground_truth'])
            pred_str = ', '.join(scene['predictions'])

            # Colori basati su performance
            f1_class = 'good' if metrics['f1_score'] > 0.8 else 'warning' if metrics['f1_score'] > 0.5 else 'error'

            html += f"""
                    <tr>
                        <td>{scene['scene_id']}</td>
                        <td>{gt_str}</td>
                        <td>{pred_str}</td>
                        <td>{metrics['precision']:.3f}</td>
                        <td>{metrics['recall']:.3f}</td>
                        <td class="{f1_class}">{metrics['f1_score']:.3f}</td>
                        <td>{scene['processing_time']:.2f}</td>
                    </tr>
            """

        html += """
                </tbody>
            </table>
            
            <h2>📊 Grafici delle Performance</h2>
            <p>I grafici dettagliati sono disponibili nella cartella reports/figures/</p>
            
            <hr>
            <footer>
                <p><em>Report generato dal sistema di valutazione automatica</em></p>
            </footer>
        </body>
        </html>
        """

        return html

    def _create_performance_plots(self):
        """
        Crea grafici delle performance.
        """
        figures_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)

        # Grafico 1: Distribuzione metriche per scena
        scene_metrics = [scene['metrics'] for scene in self.detailed_results]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        metrics_names = ['precision', 'recall', 'f1_score', 'accuracy']

        for i, metric in enumerate(metrics_names):
            ax = axes[i//2, i%2]
            values = [m[metric] for m in scene_metrics]

            ax.hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'Distribuzione {metric.replace("_", " ").title()}')
            ax.set_xlabel('Valore')
            ax.set_ylabel('Frequenza')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'performance_metrics.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Grafico 2: Performance nel tempo
        processing_times = [scene['processing_time'] for scene in self.detailed_results]

        plt.figure(figsize=(12, 6))
        plt.plot(processing_times, 'o-', alpha=0.7)
        plt.title('Tempo di Elaborazione per Scena')
        plt.xlabel('Scena')
        plt.ylabel('Tempo (secondi)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(figures_dir, 'processing_times.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Matrice di confusione
        cm_path = os.path.join(figures_dir, 'confusion_matrix.png')
        self.create_confusion_matrix(cm_path)

        print(f"📊 Grafici salvati in: {figures_dir}")

    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> str:
        """
        Confronta le performance di diversi modelli.

        Args:
            model_results: Dizionario con risultati di diversi modelli

        Returns:
            Percorso del report di confronto
        """
        comparison_data = []

        for model_name, results in model_results.items():
            global_metrics = results.get('global_metrics', {})
            comparison_data.append({
                'model': model_name,
                'accuracy': global_metrics.get('accuracy', 0),
                'precision': global_metrics.get('precision', 0),
                'recall': global_metrics.get('recall', 0),
                'f1_score': global_metrics.get('f1_score', 0),
                'avg_time': results.get('average_processing_time', 0)
            })

        # Crea grafico di confronto
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        models = [data['model'] for data in comparison_data]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']

        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = [data[metric] for data in comparison_data]

            bars = ax.bar(models, values, alpha=0.7)
            ax.set_title(f'Confronto {metric.replace("_", " ").title()}')
            ax.set_ylabel('Valore')
            ax.set_ylim(0, 1)

            # Aggiungi valori sopra le barre
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        comparison_path = os.path.join(self.output_dir, 'figures', 'model_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Confronto modelli salvato: {comparison_path}")
        return comparison_path
