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
        layers_to_probe = [0,6,12,18,24,30,33]
    embedder = ESM2Embedder()

    # 训练集按残基统计类别权重（照顾 E）
    cw = None
    if class_weight:
        from collections import Counter
        cnt = Counter([r.label_hec for r in residues_by_split['train']])
        tot = sum(cnt.values()) + 1e-6
        inv = {k: tot/(v+1e-6) for k,v in cnt.items()}
        mean = np.mean(list(inv.values()))
        cw = {0:float(inv.get('H',1.0)/mean), 1:float(inv.get('E',1.0)/mean), 2:float(inv.get('C',1.0)/mean)}

    results = {}
    for layer in layers_to_probe:
        Xtr, ytr, _, _ = build_xy_for_split_layer(residues_by_split['train'], proteins, embedder, layer)
        Xva, yva, _, _ = build_xy_for_split_layer(residues_by_split['val'],   proteins, embedder, layer)
        Xte, yte, bte, _ = build_xy_for_split_layer(residues_by_split['test'], proteins, embedder, layer)
        if len(ytr)==0 or len(yte)==0:
            results[layer] = {'accuracy':0.0,'f1_macro':0.0,'n_train':int(len(ytr)),'n_test':int(len(yte))}
            continue
        clf = LogisticRegression(max_iter=2000, random_state=42)
        if cw is not None:
            clf.set_params(class_weight=cw)
        pipe = Pipeline([('scaler', StandardScaler(with_mean=False)), ('clf', clf)])
        pipe.fit(Xtr, ytr)
        # 验证（可选）
        if len(yva)>0:
            yvap = pipe.predict(Xva)
            print(f"[L{layer}] val acc={accuracy_score(yva,yvap):.3f} f1={f1_score(yva,yvap,average='macro'):.3f}")
        # 测试
        ytep = pipe.predict(Xte)
        acc = accuracy_score(yte, ytep)
        f1m = f1_score(yte, ytep, average='macro')
        # 按桶
        per_bucket = {}
        for bucket in ['alpha','beta','alpha_beta']:
            idx = [i for i,b in enumerate(bte) if b==bucket]
            if idx:
                per_bucket[bucket] = {
                    'acc': accuracy_score(yte[idx], ytep[idx]),
                    'f1_macro': f1_score(yte[idx], ytep[idx], average='macro')
                }
        results[layer] = {'accuracy':acc, 'f1_macro':f1m, 'per_bucket':per_bucket,
                          'n_train':int(len(ytr)), 'n_test':int(len(yte))}
        print(f"[L{layer}] test acc={acc:.3f} f1_macro={f1m:.3f} (n={len(ytr)}/{len(yte)})")
    return results


def example_usage():
    protein_csv = "ss_multi_dataset.split.proteins.csv"
    residue_csv = "ss_multi_dataset.split.residues.csv"
    layers = [0,6,12,18,24,30,33]
    results = run_real_dataset_probing(protein_csv, residue_csv, layers_to_probe=layers, class_weight=True)
    plot_layer_performance(results, metric_name='accuracy', title='Token-level SS accuracy (test) across layers')
    plt.show()

if __name__ == "__main__":
  example_usage()
