# Layer-wise Probing for physical Constraints Analysis
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, EsmModel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


#=========1部分====================加载数据==========================
@dataclass
class ProteinEntry:
    uniprot_id: str
    sequence: str
    bucket: str
    split: str
@dataclass
class ResidueEntry:
    uniprot_id: str
    position: int
    label_hec: str #H/E/C
    split: str

def load_split_csvs(protein_csv: str, residue_csv: str):
    """
    Args: protein_csv:蛋白csv路径; residue_csv:残基csv路径
    Returns: proteins: {uniprot_id: ProteinEntry} 字典; residues_by_split: {'train': [ResidueEntry, ...], 'val': [...], 'test': [...]}
    """
    p_df =  pd.read_csv(protein_csv)
    r_df = pd.read_csv(residue_csv)
    #validated split
    p_df = p_df[p_df['split'].isin(['train','val','test'])]
    r_df = r_df[r_df['split'].isin(['train','val','test'])]
    #dict of Proteins
    proteins = {}
    for row in p_df.itertuples(index=False):
        proteins[row.uniprot_id] = ProteinEntry(
            uniprot_id=row.uniprot_id,
            sequence=row.sequence,
            bucket=getattr(row, 'bucket', 'unknown'),
            split=row.split
        )

    residues_by_split = defaultdict(list)
    for row in r_df.itertuples(index=False):
        residues_by_split[row.split].append(
            ResidueEntry(
                uniprot_id=row.uniprot_id,
                position=int(row.position),
                label_hec=row.label_hec,
                split=row.split
            )
        )
    return proteins, residues_by_split

#=========2部分====================ESM2embedding==========================
class ESM2Embedder:
    def __init__(
            self,
            model_name: str="facebook/esm2_t33_650M_UR50D",
            emb_dir: str="emb_cache",
            device: Optional[str]= None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{self.device}")

        print(f"Model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(
            model_name,
            output_hidden_states=True # Uncommented this argument
        ).to(self.device).eval()

        self.emb_dir = Path(emb_dir)
        self.emb_dir.mkdir(parents=True, exist_ok=True)

        # Note: Accessing hidden_states might require a different approach
        # depending on the transformers version and model.
        # The current code assumes hidden_states is accessible after forward pass.
        self.num_layers = self.model.config.num_hidden_layers + 1  # +1 for embedding layer
        self.hidden_size = self.model.config.hidden_size
        print(f"Layer of model: {self.num_layers}, dimension of hidden layer: {self.hidden_size}")


    @torch.no_grad()
    def encode_and_cache(self, uid: str, sequence: str):
        path = self.emb_dir / f"{uid}.pt"
        if path.exists():
            return torch.load(path, map_location="cpu")
        toks = self.tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
        toks = {k:v.to(self.device) for k,v in toks.items()}
        out = self.model(**toks)
        # Access hidden_states from the model output
        hs = out.hidden_states  # tuple(L+1)[1,T,H]
        layers = [x[0,1:-1,:].detach().cpu() for x in hs]  # strip special tokens
        emb = torch.stack(layers, dim=0)  # (L+1, L, H)
        obj = {'emb': emb}
        torch.save(obj, path)
        return obj

    def layer_array(self, cache_obj, layer_idx:int):
        return cache_obj['emb'][layer_idx].numpy()  # (L,H)
    
LABEL2ID = {'H':0,'E':1,'C':2}
ID2LABEL = {0:'H',1:'E',2:'C'}

def build_xy_for_split_layer(split_residues: List[ResidueEntry],
                             proteins: Dict[str, ProteinEntry],
                             embedder: ESM2Embedder,
                             layer: int):
    X_list, y_list, bucket_list, uid_list = [], [], [], []
    by_uid = defaultdict(list)
    for r in split_residues:
        by_uid[r.uniprot_id].append(r)
    for uid, items in by_uid.items():
        cache = embedder.encode_and_cache(uid, proteins[uid].sequence)
        arr = embedder.layer_array(cache, layer)  # (L,H)
        L = arr.shape[0]
        protein_bucket = proteins[uid].bucket # Get bucket from ProteinEntry
        for r in items:
            i = r.position - 1
            if 0 <= i < L:
                X_list.append(arr[i]); y_list.append(LABEL2ID[r.label_hec])
                bucket_list.append(protein_bucket); uid_list.append(uid) # Use the protein bucket
    if not X_list:
        return np.zeros((0,1)), np.zeros((0,),dtype=int), [], []
    return np.stack(X_list,0), np.array(y_list,dtype=int), bucket_list, uid_list

def run_real_dataset_probing(protein_csv, residue_csv, layers_to_probe=None, class_weight=True):
    proteins, residues_by_split = load_split_csvs(protein_csv, residue_csv)
    if layers_to_probe is None:
        layers_to_probe = [0,3,6,9,12,15,18,21,24,27,30,33]
    embedder = ESM2Embedder()

    # 训练集按残基统计类别权重（照顾 E）
    cw = None
    if class_weight:
        from collections import Counter
        cnt = Counter([r.label_hec for r in residues_by_split['train']])
        tot = sum(cnt.values()) + 1e-6
        inv = {k: tot/(v+1e-6) for k,v in cnt.items()}
        mean_inv = np.mean(list(inv.values()))
        cw = {0: float(inv.get('H',1.0)/mean_inv), 
              1: float(inv.get('E',1.0)/mean_inv), 
              2: float(inv.get('C',1.0)/mean_inv)}
        print(f"Class weights: {cw}")

    results = {}
    for layer in layers_to_probe:
        Xtr, ytr, _, _ = build_xy_for_split_layer(residues_by_split['train'], proteins, embedder, layer)
        Xva, yva, _, _ = build_xy_for_split_layer(residues_by_split['val'],   proteins, embedder, layer)
        Xte, yte, bte, _ = build_xy_for_split_layer(residues_by_split['test'], proteins, embedder, layer)
        if len(ytr)==0 or len(yte)==0:
            results[layer] = {'val_acc':0.0, 'test_acc':0.0, 'val_f1':0.0, 'test_f1':0.0}
            continue
        clf = LogisticRegression(
            max_iter=3000,  # 增加迭代次数
            C=0.5,          # 正则化强度（可以试试0.5, 1.0, 2.0）
            solver='lbfgs', # 对于大数据更稳定
            random_state=42,
            class_weight=cw
        )
        #Normalizer pas StandardScaler
        pipe = Pipeline([
            ('normalizer', Normalizer(norm='l2')),  # L2归一化
            ('clf', clf)
        ])
        
        pipe.fit(Xtr, ytr)

        #validation
        val_acc, val_f1 = 0.0, 0.0
        if len(yva) > 0:
           yvap = pipe.predict(Xva)
           val_acc = accuracy_score(yva, yvap)
           val_f1 = f1_score(yva, yvap, average='macro')
           print(f"[L{layer}] val acc={val_acc:.3f} f1={val_f1:.3f}")

        #测试集test
        ytep = pipe.predict(Xte)
        test_acc = accuracy_score(yte, ytep)
        test_f1 = f1_score(yte, ytep, average='macro')

        per_bucket = {}
        for bucket in ['alpha', 'beta', 'alpha_beta']:
            idx = [i for i, b in enumerate(bte) if b == bucket]
            if idx:
                per_bucket[bucket] = {
                    'acc': accuracy_score(yte[idx], ytep[idx]),
                    'f1': f1_score(yte[idx], ytep[idx], average='macro')
                }

        results[layer] = {
            'val_acc': val_acc, 'val_f1': val_f1,
            'test_acc': test_acc, 'test_f1': test_f1,
            'per_bucket': per_bucket,
            'n_train': int(len(ytr)), 'n_test': int(len(yte))
        }
        print(f"[L{layer}] test acc={test_acc:.3f} f1_macro={test_f1:.3f} (n={len(ytr)}/{len(yte)})")
    
    return results

def plot_comprehensive_results(results, save_path='probing_results.png'):
    """
    创建一个综合的、publication-ready的图表
    """
    # 设置样式
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11
    plt.rcParams['figure.dpi'] = 150
    
    # 提取数据
    layers = sorted(results.keys())
    val_accs = [results[l]['val_acc'] for l in layers]
    test_accs = [results[l]['test_acc'] for l in layers]
    val_f1s = [results[l]['val_f1'] for l in layers]
    test_f1s = [results[l]['test_f1'] for l in layers]
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # === 图1: Accuracy曲线 ===
    ax1 = axes[0, 0]
    ax1.plot(layers, val_accs, 'o-', linewidth=2.5, markersize=8, 
             label='Validation', color='#2E86AB', alpha=0.8)
    ax1.plot(layers, test_accs, 's-', linewidth=2.5, markersize=8, 
             label='Test', color='#A23B72', alpha=0.8)
    ax1.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Secondary Structure Prediction Accuracy', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.35, 0.85])  # 调整y轴范围让曲线更明显
    
    # 标注最佳层
    best_layer = layers[np.argmax(test_accs)]
    best_acc = max(test_accs)
    ax1.annotate(f'Best: L{best_layer}\n{best_acc:.3f}', 
                xy=(best_layer, best_acc), 
                xytext=(best_layer-3, best_acc+0.05),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # === 图2: F1-Score曲线 ===
    ax2 = axes[0, 1]
    ax2.plot(layers, val_f1s, 'o-', linewidth=2.5, markersize=8,
             label='Validation', color='#2E86AB', alpha=0.8)
    ax2.plot(layers, test_f1s, 's-', linewidth=2.5, markersize=8,
             label='Test', color='#A23B72', alpha=0.8)
    ax2.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1-Score (Macro)', fontsize=12, fontweight='bold')
    ax2.set_title('Macro F1-Score Across Layers', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.35, 0.85])
    
    # === 图3: 按Bucket分析（使用最佳层）===
    ax3 = axes[1, 0]
    bucket_data = results[best_layer]['per_bucket']
    buckets = list(bucket_data.keys())
    accs = [bucket_data[b]['acc'] for b in buckets]
    f1s = [bucket_data[b]['f1'] for b in buckets]
    
    x = np.arange(len(buckets))
    width = 0.35
    bars1 = ax3.bar(x - width/2, accs, width, label='Accuracy', 
                    color='#F18F01', alpha=0.8, edgecolor='black')
    bars2 = ax3.bar(x + width/2, f1s, width, label='F1-Score',
                    color='#C73E1D', alpha=0.8, edgecolor='black')
    
    ax3.set_xlabel('Protein Category', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title(f'Performance by Category (Layer {best_layer})', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['α-helix', 'β-sheet', 'α+β'], fontsize=11)
    ax3.legend(fontsize=11, frameon=True, shadow=True)
    ax3.set_ylim([0.4, 0.9])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上显示数值
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # === 图4: Layer-wise improvement ===
    ax4 = axes[1, 1]
    improvements = [test_accs[i] - test_accs[0] if i > 0 else 0 for i in range(len(layers))]
    colors = ['green' if x > 0 else 'red' for x in improvements]
    ax4.bar(layers, improvements, color=colors, alpha=0.6, edgecolor='black')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy Δ from Layer 0', fontsize=12, fontweight='bold')
    ax4.set_title('Layer-wise Performance Gain', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存到: {save_path}")
    plt.show()


def print_results_table(results):
    """打印一个漂亮的结果表格"""
    print("\n" + "="*80)
    print("📊 PROBING RESULTS SUMMARY")
    print("="*80)
    print(f"{'Layer':<8} {'Val Acc':<12} {'Test Acc':<12} {'Val F1':<12} {'Test F1':<12}")
    print("-"*80)
    
    for layer in sorted(results.keys()):
        r = results[layer]
        print(f"L{layer:<7} {r['val_acc']:<12.4f} {r['test_acc']:<12.4f} "
              f"{r['val_f1']:<12.4f} {r['test_f1']:<12.4f}")
    
    # 找出最佳层
    best_layer = max(results.keys(), key=lambda l: results[l]['test_acc'])
    print("="*80)
    print(f"🏆 BEST LAYER: {best_layer} with Test Acc = {results[best_layer]['test_acc']:.4f}")
    print("="*80 + "\n")
