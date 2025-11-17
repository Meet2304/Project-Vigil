#!/usr/bin/env python3
"""
K-Gram Malicious Query Analysis
Complete End-to-End Workflow Script

This script implements leave-one-out ablation analysis to identify
which k-gram phrases are most responsible for malicious query detection.

Usage:
    python kgram_analysis.py --dataset MPDD.csv --model classifier.pkl

Author: Based on notebooks K_Gram_Analysis_v0_6_2 and v0_6_3
"""

import os
import sys
import pickle
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Progress bar
from tqdm.auto import tqdm

# Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 8)


class KGramAnalyzer:
    """
    K-Gram Malicious Query Analyzer
    
    Extracts k-gram phrases, ranks them by maliciousness,
    and performs leave-one-out ablation testing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize analyzer with configuration."""
        self.config = config
        self.vectorizer = None
        self.classifier = None
        self.feature_names = None
        self.X = None
        self.y_true = None
        self.y_pred = None
        self.baseline_accuracy = None
        
    def load_data(self, dataset_path: str, sample_size: int = None) -> Tuple[List[str], np.ndarray]:
        """
        Load and prepare dataset.
        
        Args:
            dataset_path: Path to CSV file with 'text' and 'isMalicious' columns
            sample_size: Optional sample size for faster analysis
            
        Returns:
            texts: List of query strings
            labels: Binary labels (1=malicious, 0=benign)
        """
        print("📂 Loading dataset...")
        df = pd.read_csv(dataset_path)
        print(f"   Loaded {len(df):,} samples")
        
        # Validate required columns
        if 'text' not in df.columns or 'isMalicious' not in df.columns:
            raise ValueError("Dataset must have 'text' and 'isMalicious' columns")
        
        # Stratified sampling if needed
        if sample_size and sample_size < len(df):
            print(f"   Sampling {sample_size:,} samples (stratified)...")
            
            # Calculate class proportions
            mal_count = df['isMalicious'].sum()
            ben_count = len(df) - mal_count
            mal_prop = mal_count / len(df)
            
            # Sample proportionally
            n_mal = int(sample_size * mal_prop)
            n_ben = sample_size - n_mal
            
            mal_df = df[df['isMalicious'] == 1].sample(
                n=n_mal, random_state=self.config['random_state'])
            ben_df = df[df['isMalicious'] == 0].sample(
                n=n_ben, random_state=self.config['random_state'])
            
            df = pd.concat([mal_df, ben_df], ignore_index=True)
            df = df.sample(frac=1, random_state=self.config['random_state']).reset_index(drop=True)
            
            print(f"   Sampled: {n_mal:,} malicious, {n_ben:,} benign")
        
        # Extract texts and labels
        texts = df['text'].astype(str).tolist()
        labels = df['isMalicious'].values
        
        print(f"   Final dataset: {len(texts):,} samples")
        print(f"   Malicious: {labels.sum():,} ({labels.sum()/len(labels)*100:.1f}%)")
        print(f"   Benign: {(1-labels).sum():,} ({(1-labels).sum()/len(labels)*100:.1f}%)")
        
        return texts, labels
    
    def create_features(self, texts: List[str]) -> np.ndarray:
        """
        Create TF-IDF k-gram features.
        
        Args:
            texts: List of query strings
            
        Returns:
            X: Sparse feature matrix
        """
        print("\n🔤 Creating k-gram features...")
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.config['ngram_range'],
            max_features=self.config['max_features'],
            min_df=self.config['min_df'],
            analyzer=self.config['analyzer']
        )
        
        # Fit and transform
        self.X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"   Feature matrix shape: {self.X.shape}")
        print(f"   K-grams extracted: {len(self.feature_names):,}")
        print(f"   N-gram range: {self.config['ngram_range'][0]}-{self.config['ngram_range'][1]} words")
        
        return self.X
    
    def load_classifier(self, model_path: str):
        """Load pre-trained classifier."""
        print("\n🤖 Loading classifier...")
        
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)
        
        print(f"   Classifier: {type(self.classifier).__name__}")
        print(f"   ✓ Model loaded successfully")
    
    def get_baseline_predictions(self, labels: np.ndarray):
        """Get baseline predictions and metrics."""
        print("\n📊 Getting baseline predictions...")
        
        self.y_true = labels
        self.y_pred = self.classifier.predict(self.X)
        
        # Calculate metrics
        self.baseline_accuracy = accuracy_score(self.y_true, self.y_pred)
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        print(f"   Baseline Accuracy: {self.baseline_accuracy:.4f} ({self.baseline_accuracy*100:.2f}%)")
        print(f"\n   Confusion Matrix:")
        print(f"   [[TN={cm[0,0]:5d}  FP={cm[0,1]:5d}]")
        print(f"    [FN={cm[1,0]:5d}  TP={cm[1,1]:5d}]]")
        
        # Classification report
        print(f"\n   Classification Report:")
        print(classification_report(self.y_true, self.y_pred, 
                                   target_names=['Benign', 'Malicious'],
                                   digits=4))
    
    def analyze_malicious_phrases(self, top_k: int = 50) -> List[Dict]:
        """
        Analyze which phrases are most associated with malicious queries.
        
        Args:
            top_k: Number of top phrases to return
            
        Returns:
            List of dictionaries with phrase statistics
        """
        print("\n🔍 Analyzing malicious phrase associations...")
        
        # Create masks
        malicious_mask = self.y_true == 1
        benign_mask = self.y_true == 0
        
        # Calculate mean TF-IDF per class
        mal_mean = np.asarray(self.X[malicious_mask].mean(axis=0)).ravel()
        ben_mean = np.asarray(self.X[benign_mask].mean(axis=0)).ravel()
        
        # Calculate discriminative scores
        malicious_scores = []
        
        for idx, feature in enumerate(tqdm(self.feature_names, desc="   Scoring phrases")):
            # Get feature column
            feature_col = self.X[:, idx].toarray().ravel()
            
            # Calculate metrics
            mal_score = mal_mean[idx]
            ben_score = ben_mean[idx]
            
            # Count occurrences
            mal_count = np.sum((feature_col > 0) & malicious_mask)
            ben_count = np.sum((feature_col > 0) & benign_mask)
            
            # Calculate ratio (handling division by zero)
            if ben_score > 0:
                ratio = mal_score / ben_score
            else:
                ratio = mal_score * 1000  # High ratio if only in malicious
            
            # Combined discriminative score
            discriminative_score = mal_score * ratio
            
            malicious_scores.append({
                'phrase': feature,
                'idx': idx,
                'mal_mean_tfidf': mal_score,
                'ben_mean_tfidf': ben_score,
                'mal_count': int(mal_count),
                'ben_count': int(ben_count),
                'mal_ben_ratio': ratio,
                'discriminative_score': discriminative_score
            })
        
        # Sort by discriminative score
        malicious_scores.sort(key=lambda x: x['discriminative_score'], reverse=True)
        
        # Display top phrases
        self._display_top_phrases(malicious_scores[:top_k])
        
        return malicious_scores[:top_k]
    
    def _display_top_phrases(self, top_phrases: List[Dict], n: int = 20):
        """Display top malicious phrases."""
        print(f"\n{'='*90}")
        print(f"TOP {n} MALICIOUS PHRASES")
        print(f"{'='*90}")
        print(f"\n{'Rank':<6} {'Phrase':<50} {'Ratio':>8} {'Mal':>6} {'Ben':>6}")
        print(f"{'-'*90}")
        
        for rank, item in enumerate(top_phrases[:n], 1):
            phrase = item['phrase']
            if len(phrase) > 48:
                phrase = phrase[:45] + "..."
            print(f"{rank:<6} {phrase:<50} {item['mal_ben_ratio']:>8.2f} "
                  f"{item['mal_count']:>6} {item['ben_count']:>6}")
        
        print(f"{'='*90}\n")
    
    def ablation_analysis(self, top_phrases: List[Dict], top_k: int = 30) -> List[Dict]:
        """
        Perform leave-one-out ablation testing.
        
        For each phrase, remove it from the feature matrix and measure
        how much the model's predictions change.
        
        Args:
            top_phrases: List of top malicious phrases to test
            top_k: Number of phrases to test
            
        Returns:
            List of ablation results
        """
        print(f"\n🔬 Running ablation analysis (testing {top_k} phrases)...")
        print(f"   This tests what happens when each phrase is removed...")
        
        ablation_results = []
        
        for item in tqdm(top_phrases[:top_k], desc="   Testing phrases"):
            # Create ablated feature matrix (set this feature to 0)
            X_ablated = self.X.copy()
            X_ablated[:, item['idx']] = 0
            
            # Get predictions with ablated features
            y_pred_ablated = self.classifier.predict(X_ablated)
            
            # Calculate changes
            acc_ablated = accuracy_score(self.y_true, y_pred_ablated)
            acc_drop = self.baseline_accuracy - acc_ablated
            
            # Count prediction flips
            flips = np.sum(self.y_pred != y_pred_ablated)
            mal_flips = np.sum((self.y_pred != y_pred_ablated) & (self.y_true == 1))
            ben_flips = np.sum((self.y_pred != y_pred_ablated) & (self.y_true == 0))
            
            # Detailed flip analysis
            mal_to_ben = np.sum((self.y_pred == 1) & (y_pred_ablated == 0))
            ben_to_mal = np.sum((self.y_pred == 0) & (y_pred_ablated == 1))
            
            ablation_results.append({
                'phrase': item['phrase'],
                'accuracy_drop': acc_drop,
                'accuracy_ablated': acc_ablated,
                'total_flips': int(flips),
                'malicious_flips': int(mal_flips),
                'benign_flips': int(ben_flips),
                'mal_to_ben_flips': int(mal_to_ben),
                'ben_to_mal_flips': int(ben_to_mal),
                'flip_rate': flips / len(self.y_true)
            })
        
        # Sort by accuracy drop
        ablation_results.sort(key=lambda x: x['accuracy_drop'], reverse=True)
        
        # Display results
        self._display_ablation_results(ablation_results)
        
        return ablation_results
    
    def _display_ablation_results(self, results: List[Dict], n: int = 20):
        """Display ablation results."""
        print(f"\n{'='*100}")
        print(f"ABLATION RESULTS: Impact When Phrases Are Removed")
        print(f"{'='*100}")
        print(f"\n{'Rank':<6} {'Phrase':<45} {'Acc Drop':>10} {'Flips':>7} {'Rate':>8}")
        print(f"{'-'*100}")
        
        for rank, result in enumerate(results[:n], 1):
            phrase = result['phrase']
            if len(phrase) > 43:
                phrase = phrase[:40] + "..."
            print(f"{rank:<6} {phrase:<45} {result['accuracy_drop']:>+10.4f} "
                  f"{result['total_flips']:>7} {result['flip_rate']*100:>7.2f}%")
        
        print(f"{'='*100}\n")
    
    def visualize_results(self, top_phrases: List[Dict], ablation_results: List[Dict],
                         output_dir: str = '.'):
        """Create visualizations of results."""
        print("\n📊 Creating visualizations...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualization 1: Top Malicious Phrases
        self._plot_malicious_phrases(top_phrases[:20], output_dir)
        
        # Visualization 2: Ablation Impact
        self._plot_ablation_impact(ablation_results[:20], output_dir)
        
        print(f"   ✓ Visualizations saved to {output_dir}")
    
    def _plot_malicious_phrases(self, top_phrases: List[Dict], output_dir: Path):
        """Plot top malicious phrases analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        phrases = [item['phrase'][:40] for item in top_phrases]
        ratios = [item['mal_ben_ratio'] for item in top_phrases]
        mal_counts = [item['mal_count'] for item in top_phrases]
        ben_counts = [item['ben_count'] for item in top_phrases]
        
        # Plot 1: Malicious/Benign Ratio
        ax1.barh(range(len(phrases)), ratios, color='crimson', alpha=0.7)
        ax1.set_yticks(range(len(phrases)))
        ax1.set_yticklabels(phrases, fontsize=9)
        ax1.set_xlabel('Malicious/Benign Ratio', fontsize=12, fontweight='bold')
        ax1.set_title('Top 20 Malicious Phrases by Class Ratio', 
                     fontsize=13, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()
        
        # Plot 2: Occurrence counts
        y_pos = np.arange(len(phrases))
        ax2.barh(y_pos + 0.2, mal_counts, 0.4, label='Malicious', 
                color='crimson', alpha=0.7)
        ax2.barh(y_pos - 0.2, ben_counts, 0.4, label='Benign', 
                color='steelblue', alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(phrases, fontsize=9)
        ax2.set_xlabel('Occurrence Count', fontsize=12, fontweight='bold')
        ax2.set_title('Phrase Occurrence by Class', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'malicious_phrases_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ablation_impact(self, ablation_results: List[Dict], output_dir: Path):
        """Plot ablation impact analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        phrases = [r['phrase'][:40] for r in ablation_results]
        acc_drops = [r['accuracy_drop'] * 100 for r in ablation_results]
        mal_flips = [r['mal_to_ben_flips'] for r in ablation_results]
        ben_flips = [r['ben_to_mal_flips'] for r in ablation_results]
        
        # Plot 1: Accuracy drop
        colors = ['red' if x > 0 else 'green' for x in acc_drops]
        ax1.barh(range(len(phrases)), acc_drops, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(phrases)))
        ax1.set_yticklabels(phrases, fontsize=9)
        ax1.set_xlabel('Accuracy Change When Removed (%)', 
                      fontsize=12, fontweight='bold')
        ax1.set_title('Impact of Removing Each Phrase', 
                     fontsize=13, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()
        
        # Plot 2: Prediction flips
        y_pos = np.arange(len(phrases))
        ax2.barh(y_pos + 0.2, mal_flips, 0.4, label='Mal→Ben', 
                color='crimson', alpha=0.7)
        ax2.barh(y_pos - 0.2, ben_flips, 0.4, label='Ben→Mal', 
                color='orange', alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(phrases, fontsize=9)
        ax2.set_xlabel('Number of Prediction Flips', 
                      fontsize=12, fontweight='bold')
        ax2.set_title('Prediction Changes by Class', 
                     fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ablation_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_results(self, top_phrases: List[Dict], ablation_results: List[Dict],
                      output_dir: str = '.'):
        """Export results to files."""
        print("\n💾 Exporting results...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export malicious phrases
        phrases_df = pd.DataFrame(top_phrases)
        phrases_path = output_dir / 'malicious_phrases.csv'
        phrases_df.to_csv(phrases_path, index=False)
        print(f"   ✓ Malicious phrases → {phrases_path}")
        
        # Export ablation results
        ablation_df = pd.DataFrame(ablation_results)
        ablation_path = output_dir / 'ablation_results.csv'
        ablation_df.to_csv(ablation_path, index=False)
        print(f"   ✓ Ablation results → {ablation_path}")
        
        # Export summary
        summary = {
            'configuration': {
                'ngram_range': self.config['ngram_range'],
                'max_features': self.config['max_features'],
                'min_df': self.config['min_df'],
                'sample_size': len(self.y_true)
            },
            'baseline_performance': {
                'accuracy': float(self.baseline_accuracy),
                'total_samples': int(len(self.y_true)),
                'malicious_samples': int(self.y_true.sum()),
                'benign_samples': int((1 - self.y_true).sum())
            },
            'top_malicious_phrase': {
                'phrase': top_phrases[0]['phrase'],
                'mal_ben_ratio': float(top_phrases[0]['mal_ben_ratio']),
                'mal_count': int(top_phrases[0]['mal_count']),
                'ben_count': int(top_phrases[0]['ben_count'])
            },
            'highest_impact_phrase': {
                'phrase': ablation_results[0]['phrase'],
                'accuracy_drop': float(ablation_results[0]['accuracy_drop']),
                'flip_rate': float(ablation_results[0]['flip_rate'])
            },
            'ablation_statistics': {
                'phrases_tested': len(ablation_results),
                'phrases_with_negative_impact': sum(1 for r in ablation_results if r['accuracy_drop'] > 0),
                'phrases_with_positive_impact': sum(1 for r in ablation_results if r['accuracy_drop'] < 0),
                'max_accuracy_drop': float(max(r['accuracy_drop'] for r in ablation_results)),
                'mean_accuracy_drop': float(np.mean([r['accuracy_drop'] for r in ablation_results]))
            }
        }
        
        summary_path = output_dir / 'analysis_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   ✓ Summary → {summary_path}")
    
    def print_summary(self, top_phrases: List[Dict], ablation_results: List[Dict]):
        """Print final analysis summary."""
        print(f"\n{'='*90}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*90}")
        
        print(f"\n📊 Configuration:")
        print(f"   K-Gram Size: {self.config['ngram_range'][0]}-{self.config['ngram_range'][1]} words")
        print(f"   Total Samples: {len(self.y_true):,}")
        print(f"   Total K-Grams: {len(self.feature_names):,}")
        
        print(f"\n🎯 Baseline Performance:")
        print(f"   Accuracy: {self.baseline_accuracy:.4f} ({self.baseline_accuracy*100:.2f}%)")
        
        print(f"\n🔝 Most Discriminative Phrase:")
        top1 = top_phrases[0]
        print(f"   '{top1['phrase']}'")
        print(f"   Mal/Ben Ratio: {top1['mal_ben_ratio']:.2f}x")
        print(f"   Appears in: {top1['mal_count']} malicious, {top1['ben_count']} benign")
        
        print(f"\n💥 Highest Impact Phrase (Ablation):")
        impact1 = ablation_results[0]
        print(f"   '{impact1['phrase']}'")
        print(f"   Accuracy Drop: {impact1['accuracy_drop']:+.4f} ({impact1['accuracy_drop']*100:+.2f}%)")
        print(f"   Prediction Flips: {impact1['total_flips']} ({impact1['flip_rate']*100:.2f}%)")
        
        print(f"\n📈 Ablation Statistics:")
        acc_drops = [r['accuracy_drop'] for r in ablation_results]
        print(f"   Phrases with negative impact: {sum(1 for x in acc_drops if x > 0)}")
        print(f"   Phrases with positive impact: {sum(1 for x in acc_drops if x < 0)}")
        print(f"   Max accuracy drop: {max(acc_drops):+.4f} ({max(acc_drops)*100:+.2f}%)")
        print(f"   Average impact: {np.mean(acc_drops):+.4f}")
        
        print(f"{'='*90}\n")


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='K-Gram Malicious Query Analysis with Ablation Testing'
    )
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset CSV file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained classifier pickle file')
    parser.add_argument('--ngram-min', type=int, default=3,
                       help='Minimum n-gram size (default: 3)')
    parser.add_argument('--ngram-max', type=int, default=5,
                       help='Maximum n-gram size (default: 5)')
    parser.add_argument('--max-features', type=int, default=384,
                       help='Maximum number of features (default: 384)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size (default: use all data)')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Number of top phrases to analyze (default: 50)')
    parser.add_argument('--ablation-k', type=int, default=30,
                       help='Number of phrases to ablate (default: 30)')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory (default: ./output)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'ngram_range': (args.ngram_min, args.ngram_max),
        'max_features': args.max_features,
        'min_df': 2,
        'analyzer': 'word',
        'random_state': 42
    }
    
    print("🚀 K-Gram Malicious Query Analysis")
    print(f"{'='*90}\n")
    
    # Initialize analyzer
    analyzer = KGramAnalyzer(config)
    
    # Run analysis pipeline
    try:
        # 1. Load data
        texts, labels = analyzer.load_data(args.dataset, args.sample_size)
        
        # 2. Create features
        analyzer.create_features(texts)
        
        # 3. Load classifier
        analyzer.load_classifier(args.model)
        
        # 4. Get baseline predictions
        analyzer.get_baseline_predictions(labels)
        
        # 5. Analyze malicious phrases
        top_phrases = analyzer.analyze_malicious_phrases(top_k=args.top_k)
        
        # 6. Ablation analysis
        ablation_results = analyzer.ablation_analysis(top_phrases, top_k=args.ablation_k)
        
        # 7. Visualize results
        if not args.no_plots:
            analyzer.visualize_results(top_phrases, ablation_results, args.output_dir)
        
        # 8. Export results
        analyzer.export_results(top_phrases, ablation_results, args.output_dir)
        
        # 9. Print summary
        analyzer.print_summary(top_phrases, ablation_results)
        
        print("✅ Analysis complete!")
        print(f"   Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
