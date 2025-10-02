# Layer-wise Probing for physical Constraints Analysis
import torch
import numpy as np
from transformers import AutoTokenizer, EsmModel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


#Extract embeddings in particular Layers
class ESM2LayerExtractor:
    
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D"):
       
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def extract_embeddings(self, sequences, layers=None, pool='mean'):
        
        if layers is None:
            layers = list(range(self.model.config.num_hidden_layers + 1))
        
        all_embeddings = {layer: [] for layer in layers}
        
        with torch.no_grad():
            for seq in sequences:
                # Tokenize
                inputs = self.tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=1024)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.model(**inputs)
                hidden_states = outputs.hidden_states  # (num_layers+1, batch, seq_len, hidden_dim)
                
                # Extract specified layers
                for layer in layers:
                    layer_output = hidden_states[layer][0]  # (seq_len, hidden_dim)
                    
                    # Pool to sequence-level representation
                    if pool == 'mean':
                        # 跳过特殊token <cls> 和 <eos>
                        embedding = layer_output[1:-1].mean(dim=0).cpu().numpy()
                    elif pool == 'cls':
                        embedding = layer_output[0].cpu().numpy()  # 使用<cls> token
                    
                    all_embeddings[layer].append(embedding)
        
        # Convert to numpy arrays
        return {layer: np.array(embs) for layer, embs in all_embeddings.items()}  

# Probing Classifiers
class ProteinProbe:
    def __init__(self, task_type='classification'):
        self.task_type = task_type
        if task_type == 'classification':  # 用 ==，不是 =
            self.model = Pipeline([
                ('scaler', StandardScaler(with_mean=False)),
                ('clf', LogisticRegression(max_iter=2000, random_state=42))
            ])
        else:
            self.model = Pipeline([
                ('scaler', StandardScaler(with_mean=False)),
                ('reg', Ridge(alpha=1.0))
            ])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        if self.task_type == 'classification':
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            return {'task': 'classification', 'accuracy': acc, 'f1_score': f1}
        else:
            mse = mean_squared_error(y_test, y_pred)    
            corr = pearsonr(y_test, y_pred)[0]
            return {'task': 'regression', 'mse': mse, 'pearson_r': corr}

#Task 1: Prediction of second structure
class SecondaryStructureTask:
    """Prediction of Secondary Structure"""

    @staticmethod
    def check_physical_validity(predictions):
        """
        predictions: 可迭代，如 'H','E','C' 序列；只用到了 'H' 和 'E'。
        规则（可按需改）：
          - 螺旋短段: 长度 < 3
          - 螺旋超长: 长度 > 40
          - 孤立 β 链: 单个 'E'（没有相邻的 E）
        """
        if not predictions:
            return {
                'short_helix': 0,
                'long_helix': 0,
                'isolated_strand': 0,
                'violation_rate': 0.0,
                'n_helix_segments': 0,
                'n_strand_segments': 0
            }

        # 统计 H 段
        helix_runs = []
        run = 0
        for ss in predictions:
            if ss == 'H':
                run += 1
            else:
                if run > 0:
                    helix_runs.append(run)
                    run = 0
        if run > 0:  # 收尾
            helix_runs.append(run)

        # 统计 E 段
        strand_runs = []
        run = 0
        for ss in predictions:
            if ss == 'E':
                run += 1
            else:
                if run > 0:
                    strand_runs.append(run)
                    run = 0
        if run > 0:
            strand_runs.append(run)

        # 违例计数
        short_helix = sum(1 for l in helix_runs if l < 3)
        long_helix  = sum(1 for l in helix_runs if l > 40)
        isolated_strand = sum(1 for l in strand_runs if l == 1)  # 也可用 l < 2

        # 分母避免 0：若没有任何 H 段，用 1 做兜底
        denom = max(len(helix_runs), 1)
        violation_rate = (short_helix + long_helix) / denom

        return {
            'short_helix': short_helix,
            'long_helix': long_helix,
            'isolated_strand': isolated_strand,
            'violation_rate': violation_rate,
            'n_helix_segments': len(helix_runs),
            'n_strand_segments': len(strand_runs)
        }
    

HYDRO = set("AILVFMWY")          # 疏水
POLAR = set("STNQCH")            # 极性（含Cys）
POS   = set("KR")                # 正电
NEG   = set("DE")                # 负电
PRO   = "P"
GLY   = "G"

def _charge(a):
    if a in POS: return +1
    if a in NEG: return -1
    return 0

def _is_core(sasa, cutoff=20.0):  # Å^2，粗略阈值
    return sasa is not None and sasa < cutoff

def _volume(aa):
    # 粗略体积序（Å^3），只要相对大小
    tbl = dict(A=88,R=173,N=114,D=111,C=108,Q=143,E=138,G=60,H=153,I=166,L=166,K=168,
               M=162,F=189,P=112,S=89,T=116,W=227,Y=193,V=140)
    return tbl.get(aa, 130)

#Task 2: 去稳定检查
class StabilityTask:
    @staticmethod
    def energy_consistency_with_structure(items):
        """
        items: 列表[dict,...]，每个 dict 至少包含：
          {
            'pred_ddg': float,
            'true_ddg': float or None,      # 可选
            'from': 'A', 'to': 'V',
            'sasa': float,                  # 单位 Å^2（突变位点侧链 SASA）
            'ss': 'H'|'E'|'C',              # 二级结构
            'disulfide': bool,              # 是否参与二硫键
            'salt_bridge': bool,            # 是否参与盐桥（基于结构分析/注释）
          }
        返回：总体统计 + 各规则命中的一致率
        """
        rules_hits = {  
            'core_polarization': 0,
            'surface_hydrophobization': 0,
            'helix_proline_break': 0,
            'strand_gly_flexible': 0,
            'volume_clash_core': 0,
            'salt_bridge_break': 0,
            'disulfide_break': 0
        }
        rules_consistent = {k: 0 for k in rules_hits}

        preds, trues = [], []

        for x in items:
            pred = x['pred_ddg']
            true = x.get('true_ddg', None)
            a, b = x['from'], x['to']
            ss = x.get('ss', 'C')
            sasa = x.get('sasa', None)
            core = _is_core(sasa)
            disulf = x.get('disulfide', False)
            sb = x.get('salt_bridge', False)

            # 1) core→极性化：一般去稳定（ΔΔG>0）
            if core and (a in HYDRO) and (b in POLAR):
                rules_hits['core_polarization'] += 1
                if pred > 0: rules_consistent['core_polarization'] += 1

            # 2) surface→疏水化：常去稳定（暴露疏水不利），ΔΔG>0（经验上更常见）
            if (not core) and (a in POLAR|NEG|POS) and (b in HYDRO):
                rules_hits['surface_hydrophobization'] += 1
                if pred > 0: rules_consistent['surface_hydrophobization'] += 1

            # 3) α-螺旋中引入 Pro：破坏主链构象，ΔΔG>0
            if ss == 'H' and b == PRO:
                rules_hits['helix_proline_break'] += 1
                if pred > 0: rules_consistent['helix_proline_break'] += 1

            # 4) β-链中 Gly → 过度柔性，常去稳定（视环境），这里弱规则：ΔΔG≥0 更合理
            if ss == 'E' and (b == GLY):
                rules_hits['strand_gly_flexible'] += 1
                if pred >= 0: rules_consistent['strand_gly_flexible'] += 1

            # 5) core 体积显著增大：易拥挤，ΔΔG>0；显著减小可能造成空穴（多为 >0）
            dv = _volume(b) - _volume(a)
            if core and abs(dv) >= 30:  # 粗阈值
                rules_hits['volume_clash_core'] += 1
                if pred > 0: rules_consistent['volume_clash_core'] += 1

            # 6) 打断盐桥：若位点参与盐桥且改成中性，通常 ΔΔG>0
            if sb and (_charge(a) != 0) and (_charge(b) == 0):
                rules_hits['salt_bridge_break'] += 1
                if pred > 0: rules_consistent['salt_bridge_break'] += 1

            # 7) 破坏二硫键：Cys→非Cys，多数 ΔΔG>0
            if disulf and a == 'C' and b != 'C':
                rules_hits['disulfide_break'] += 1
                if pred > 0: rules_consistent['disulfide_break'] += 1

            # 记录整体统计
            preds.append(pred)
            if true is not None:
                trues.append(true)

        # 汇总
        out = {}
        # 逐规则一致率
        for k in rules_hits:
            n = max(rules_hits[k], 1)
            out[f'{k}_rate'] = rules_consistent[k] / n
            out[f'{k}_count'] = rules_hits[k]

        # 全局数值统计（可与结构一致性并列报告）
        preds = np.asarray(preds)
        out['extreme_predictions'] = int(np.sum(np.abs(preds) > 20))

        if len(trues) > 1:
            trues = np.asarray(trues)
            out['sign_consistency'] = float(np.mean(np.sign(preds[:len(trues)]) == np.sign(trues)))
            # 相关性
            out['pearson_r'] = float(pearsonr(preds[:len(trues)], trues)[0])
        else:
            out['sign_consistency'] = np.nan
            out['pearson_r'] = np.nan

        return out
    
#Task 3: Residues hydrophobe buried/exposed prediction
#Hydropho->buried; polar->exposed
class HydrophobicityTask:
    HYDROPHOBIC = set(list("AVILMFWP"))
    POLAR = set(list("STNQHDEKR"))

    @staticmethod
    def _is_buried_label(x):
          #x可能是字符串也可能是概率：Prob>0.5 -> buried
        if isinstance(x, str):
            return x.lower().startswith('b')
        return float(x) > 0.5

    @staticmethod
    def check_hydrophobic_pattern(sequences, burial_predictions):
        """
        sequences: [str,...]，氨基酸序列
        burial_predictions: 与 sequences 对齐的列表，每个元素是
            - 与序列等长的标签列表：'buried'/'exposed' 或
            - 与序列等长的概率列表：buried 概率 ∈ [0,1]
        返回：
            - hydrophobic_consistency: 疏水→埋藏 的平均一致率
            - polar_consistency: 极性→暴露 的平均一致率
            - overall_consistency: 二者加权平均
        """
        hydro_scores, polar_scores = [], []

        for seq, burial in zip(sequences, burial_predictions):
            buried_flags = [HydrophobicityTask._is_buried_label(b) for b in burial]

            # 疏水→埋藏
            hydro_idx = [i for i, aa in enumerate(seq) if aa in HydrophobicityTask.HYDROPHOBIC]
            if len(hydro_idx) > 0:
                hydro_hit = sum(buried_flags[i] for i in hydro_idx)
                hydro_scores.append(hydro_hit / len(hydro_idx))

            # 极性→暴露
            polar_idx = [i for i, aa in enumerate(seq) if aa in HydrophobicityTask.POLAR]
            if len(polar_idx) > 0:
                polar_hit = sum((not buried_flags[i]) for i in polar_idx)
                polar_scores.append(polar_hit / len(polar_idx))

        hydro_cons = float(np.mean(hydro_scores)) if hydro_scores else np.nan
        polar_cons = float(np.mean(polar_scores)) if polar_scores else np.nan

        # overall：对可用项取平均
        parts = [x for x in [hydro_cons, polar_cons] if not np.isnan(x)]
        overall = float(np.mean(parts)) if parts else np.nan

        return {
            'hydrophobic_consistency': hydro_cons,
            'polar_consistency': polar_cons,
            'overall_consistency': overall,
            'n_sequences_used_hydro': len(hydro_scores),
            'n_sequences_used_polar': len(polar_scores)
        }
          

#Train Probe
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from functools import partial
import numpy as np

def run_layer_wise_probing(
    sequences, labels, task_name,
    task_type='classification',
    layers_to_probe=None,
    physical_check_fn=None,
    pool='mean'
):
    """
    所有层上运行probing实验
    sequences: List[str]
    labels: np.array / List[...]
    task_type: 'classification' + 'regression'
    physical_check_fn: callable(y_pred) -> dict ou None
    """
    print("\n" + "="*60)
    print(f"Running probing for: {task_name}")
    print("="*60 + "\n")

    #1)split index: X/y对齐
    n = len(labels)
    all_idx = np.arange(n)
    train_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=42)
    y_train = np.asarray(labels)[train_idx]
    y_test = np.asarray(labels)[test_idx]

    #2)Extract chaque layer of embedding
    extractor = ESM2LayerExtractor()
    embeddings_dict = extractor.extract_embeddings(sequences, layers=layers_to_probe, pool=pool)

    results = {}

    #3)逐层training/evaluate
    for layer, embs in embeddings_dict.items():
        print(f"Probing layer {layer}...")

        X_train = embs[train_idx]
        X_test = embs[test_idx]

        probe = ProteinProbe(task_type=task_type)
        probe.train(X_train, y_train)
        metrics = probe.evaluate(X_test, y_test)

        #约束检查
        if physical_check_fn is not None:
           y_pred = probe.model.predict(X_test)
           phys = physical_check_fn(y_pred)
           if phys: metrics.update(phys)


        results[layer] = metrics
        print(f" Layer {layer}: {metrics}")

    return results

#Visualisation
def plot_layer_performance(results, metric_name='accuracy', title=None):
    """
    单个任务不同层的性能曲线
    """
    layers = sorted(results.keys())
    scores = [results[layer][metric_name] for layer in layers]

    plt.figure(figsize=(10, 6))
    plt.plot(layers, scores, marker='o', linestyle='-', color='tab:blue',
             linewidth=2, markersize=6, label=metric_name)
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    plt.title(title or f'{metric_name} across layers', fontsize=14, fontweight='bold')
    plt.xticks(layers)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return plt


def plot_multi_task_comparison(all_results, tasks):
    """
    tasks: list of (task_name, metric_name)
    """
    fig, axes = plt.subplot(1, len(tasks), figsize=(6*len(tasks), 5), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, (task_name, metric_name) in zip(axes, tasks):
        results = all_results[task_name]
        layers = sorted(results.keys())
        scores = [results[layer][metric_name] for layer in layers]
        
        ax.plot(layers, scores, marker='o', linestyle='-', linewidth=2,
                markersize=6, label=f"{task_name} ({metric_name})")
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_title(task_name, fontsize=13, fontweight='bold')
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    return fig

#测试
# 需要额外导入
from Bio import SeqIO
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_fasta(path):
    return [str(rec.seq) for rec in SeqIO.parse(path, "fasta")]

def example_usage():
    """
    以 10 条同类蛋白（Hemoglobin α 链）做分类为例
    目标：对不同物种的 α 链做分类 → 观察不同层的线性可分性
    """
    # 1) 准备数据（把我给你的 hemoglobin_alpha.fasta 存到工程里）
    sequences = load_fasta("hemoglobin_alpha.fasta")
    labels = np.array([
        "Human","Chimpanzee","Mouse","Rat","Rabbit",
        "Cow","Pig","Horse","Chicken","Frog"
    ])

    # 2) 跑逐层 probing（分类任务，不做物理检查）
    results = run_layer_wise_probing(
        sequences=sequences,
        labels=labels,
        task_name="Hemoglobin species classification",
        task_type='classification',
        layers_to_probe=[0, 6, 12, 18, 24, 33],   # 对 esm2_t33 来说是合理采样点
        physical_check_fn=None,                  # 序列级分类，无需物理检查
    )

    # 3) 可视化
    plot_layer_performance(
        results, metric_name='accuracy',
        title='Species classification accuracy across ESM-2 layers'
    )
    plt.show()

    
if __name__ == "__main__":
    example_usage()