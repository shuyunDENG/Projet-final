import requests, pandas as pd

def uniprot_search_ids(query, size=200, reviewed=True):
    base = "https://rest.uniprot.org/uniprotkb/search"
    q = query + (" AND reviewed:true" if reviewed else "")
    r = requests.get(base, params={"query": q, "fields": "accession", "size": size})
    r.raise_for_status()
    # 简单解析 accession（返回的是 TSV）
    ids = [line.strip().split("\t")[0] for line in r.text.splitlines()[1:]]
    return ids

UNIPROT_IDS = uniprot_search_ids('hemoglobin alpha', size=200, reviewed=True)
len(UNIPROT_IDS), UNIPROT_IDS[:5]

import requests
from Bio import SeqIO
from io import StringIO

def fetch_uniprot_fasta(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    r = requests.get(url); r.raise_for_status()
    rec = next(SeqIO.parse(StringIO(r.text), "fasta"))
    return str(rec.seq)

uniprot_seq = {uid: fetch_uniprot_fasta(uid) for uid in UNIPROT_IDS}
len(uniprot_seq)

import requests, json
from tqdm import tqdm

def rcsb_best_struct(uniprot_id, max_hits=3):
    # 用 RCSB 搜索与该 UniProt 相关的条目，按分辨率排序
    # GraphQL 简明做法：用聚合端点更灵活，这里用 REST 简化
    q = {
      "query": {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession", "operator": "exact_match", "value": uniprot_id}
      },
      "return_type": "entry",
      "request_options": {"paginate": {"start": 0, "rows": 50},
                          "scoring_strategy": "combined"}
    }
    r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q)
    if r.status_code!=200: return []
    ids = [x["identifier"] for x in r.json().get("result_set",[])]
    # 简单返回前几个，后面再筛分辨率
    return ids[:max_hits]

def rcsb_entry_info(pdb_id):
    r = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}")
    if r.status_code!=200: return {}
    return r.json()

def choose_representative_pdb(uniprot_id):
    cands = rcsb_best_struct(uniprot_id)
    # 取最小分辨率（若有）
    best = None; best_res = 1e9
    for pid in cands:
        info = rcsb_entry_info(pid)
        res = info.get("rcsb_entry_info",{}).get("resolution_combined", [None])[0]
        if res is not None and res < best_res:
            best, best_res = pid, res
    return best

import requests, json
from tqdm import tqdm

def rcsb_best_struct(uniprot_id, max_hits=3):
    # 用 RCSB 搜索与该 UniProt 相关的条目，按分辨率排序
    # GraphQL 简明做法：用聚合端点更灵活，这里用 REST 简化
    q = {
      "query": {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession", "operator": "exact_match", "value": uniprot_id}
      },
      "return_type": "entry",
      "request_options": {"paginate": {"start": 0, "rows": 50},
                          "scoring_strategy": "combined"}
    }
    r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q)
    if r.status_code!=200: return []
    ids = [x["identifier"] for x in r.json().get("result_set",[])]
    # 简单返回前几个，后面再筛分辨率
    return ids[:max_hits]

def rcsb_entry_info(pdb_id):
    r = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}")
    if r.status_code!=200: return {}
    return r.json()

def choose_representative_pdb(uniprot_id):
    cands = rcsb_best_struct(uniprot_id)
    # 取最小分辨率（若有）
    best = None; best_res = 1e9
    for pid in cands:
        info = rcsb_entry_info(pid)
        res = info.get("rcsb_entry_info",{}).get("resolution_combined", [None])[0]
        if res is not None and res < best_res:
            best, best_res = pid, res
    return best

def fetch_sifts_mapping(pdb_id):
    # PDBe SIFTS residue-level mapping（json）
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot_segments/{pdb_id.lower()}"
    r = requests.get(url)
    if r.status_code!=200: return {}
    return r.json()

# 为每个 UniProt 选一个代表性结构并取 SIFTS 映射
PDB_PICK = {}
SIFTS = {}
for uid in tqdm(UNIPROT_IDS):
    pid = choose_representative_pdb(uid)
    if pid:
        PDB_PICK[uid] = pid
        SIFTS[(uid,pid)] = fetch_sifts_mapping(pid)
len(PDB_PICK), list(PDB_PICK.items())[:5]

import os, subprocess, pandas as pd
from pathlib import Path
from Bio.PDB import MMCIFParser
from Bio.PDB.DSSP import DSSP

DATA_DIR = Path("pdb_mmCIF"); DATA_DIR.mkdir(exist_ok=True)

def download_mmcif(pdb_id, outdir=DATA_DIR):
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    p = outdir / f"{pdb_id}.cif"
    if not p.exists():
        r = requests.get(url)
        if r.status_code==200:
            p.write_bytes(r.content)
    return p

def run_dssp_on_chain(pdb_cif_path, chain_id):
    # 用 Biopython 的 DSSP 接口（内部会调用 mkdssp）
    parser = MMCIFParser(QUIET=True)
    struct = parser.get_structure(pdb_cif_path.stem, pdb_cif_path)
    model = struct[0]
    dssp = DSSP(model, str(pdb_cif_path))  # 返回一个可索引对象
    # 收集该链的 (resseq, aa, dssp_code)
    out = []
    for (ch, res_id), v in dssp.property_dict.items():
        if ch != chain_id: 
            continue
        aa = v["AA"]           # 单字母氨基酸
        ss8 = v["SS"] or "C"   # 8-state, 空置当作 C
        out.append((res_id[1], aa, ss8))
    out.sort(key=lambda x: x[0])
    return out  # list of (pdb_resseq, aa, ss8)

def ss8_to_hec(ss8):
    # 常用折叠：H/G/I→H；E/B→E；其他→C
    if ss8 in ("H","G","I"): return "H"
    if ss8 in ("E","B"): return "E"
    return "C"


def build_uniprot_ss(uniprot_id, pdb_id, uniprot_seq, sifts_json):
    # 1) 解析出：该 pdb 里和此 uniprot 对应的链/区段映射
    m = sifts_json.get(pdb_id.lower(), {})
    target = None
    for db in m.get("UniProt", {}).values():
        if db.get("identifier") == uniprot_id:
            target = db
            break
    if not target:
        return None

    # 取所有链
    chain_map = target["mappings"]  # 列表：每一项含 pdb_id/chain_id/residue_number 与 unp_start/unp_end
    chains = sorted(set(x["chain_id"] for x in chain_map))

    # 2) 每条链：下载 mmCIF、DSSP
    pdb_path = download_mmcif(pdb_id)
    hec_by_unp_pos = [""] * len(uniprot_seq)  # 先空，稍后填充
    for chain in chains:
        dssp_rows = run_dssp_on_chain(pdb_path, chain)
        dssp_dict = {resseq: ss8_to_hec(ss8) for (resseq, aa, ss8) in dssp_rows}

        # 3) 对每段映射，把 PDB 残基号区间映射到 UniProt 区间
        for seg in chain_map:
            if seg["chain_id"] != chain:
                continue
            unp_start = seg["unp_start"]
            unp_end   = seg["unp_end"]
            pdb_start = seg["start"]["residue_number"]
            pdb_end   = seg["end"]["residue_number"]
            # 线性配准：通常是一一对应（注意可能有缺失/插入）
            L = min(unp_end - unp_start + 1, pdb_end - pdb_start + 1)
            for k in range(L):
                unp_pos = unp_start + k
                pdb_pos = pdb_start + k
                if pdb_pos in dssp_dict:
                    hec_by_unp_pos[unp_pos - 1] = dssp_dict[pdb_pos]

    # 4) 生成等长的 H/E/C 字符串（无标签位点用 'C' 或者用 '-' 再过滤都可以）
    # 为了喂探针，建议仅取“有标签”的连续片段，简单起见这里填空缺为 'C'
    hec = "".join(h if h else "C" for h in hec_by_unp_pos)
    return hec


import pandas as pd
rows = []
for uid, seq in tqdm(uniprot_seq.items()):
    pid = PDB_PICK.get(uid)
    if not pid: 
        continue
    hec = build_uniprot_ss(uid, pid, seq, SIFTS[(uid, pid)])
    if hec and len(hec) == len(seq):
        rows.append({"sequence": seq, "ss": hec})

df = pd.DataFrame(rows)
df.to_csv("toy_ss.csv", index=False)
df.head(), len(df)
