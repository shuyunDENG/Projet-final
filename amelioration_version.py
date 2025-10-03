# ==================== 第1步: UniProt搜索 ====================
def uniprot_search_ids(query, size=200, reviewed=True):
    """搜索UniProt ID"""
    base = "https://rest.uniprot.org/uniprotkb/search"
    q = query + (" AND reviewed:true" if reviewed else "")
    params = {
        "query": q,
        "fields": "accession",
        "size": size,
        "format": "tsv"
    }
    
    try:
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        ids = [line.strip().split("\t")[0] 
               for line in r.text.splitlines()[1:] if line.strip()]
        return ids
    except Exception as e:
        print(f"❌ UniProt搜索失败: {e}")
        return []


def fetch_uniprot_fasta(uniprot_id):
    """获取UniProt序列"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        rec = next(SeqIO.parse(StringIO(r.text), "fasta"))
        return str(rec.seq)
    except Exception as e:
        print(f"❌ 获取UniProt序列失败: {e}")
        return None

# ==================== 第2步: RCSB结构搜索 ====================
def get_pdbs_from_uniprot(uniprot_id):
    """从UniProt API获取PDB列表 - 最可靠的方法"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return []
        
        data = r.json()
        pdb_list = []
        
        xrefs = data.get("uniProtKBCrossReferences", [])
        for xref in xrefs:
            if xref.get("database") == "PDB":
                pdb_id = xref.get("id")
                if pdb_id:
                    pdb_list.append(pdb_id)
        
        return pdb_list
    except Exception as e:
        print(f"❌ UniProt PDB搜索失败: {e}")
        return []

def rcsb_search_structures(uniprot_id, max_hits=5):
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
                        "operator": "exact_match",
                        "value": "UniProt"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                        "operator": "exact_match",
                        "value": uniprot_id
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": max_hits}}
    }
    
    try:
        r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", 
                         json=query, timeout=30)
        if r.status_code != 200:
            return []
        result = r.json().get("result_set", [])
        return [x["identifier"] for x in result]
    except Exception as e:
        print(f"❌ RCSB搜索失败: {e}")
        return []


def get_pdb_resolution(pdb_id):
    """获取PDB分辨率"""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        res_list = data.get("rcsb_entry_info", {}).get("resolution_combined", [])
        return res_list[0] if res_list else None
    except Exception as e:
        print(f"❌ 获取PDB分辨率失败: {e}")
        return None

def choose_best_pdb(uniprot_id):
    """选择分辨率最高的PDB结构"""
    pdb_list = rcsb_search_structures(uniprot_id)
    if not pdb_list:
        return None
    
    best_pdb = None
    best_res = 999.0
    
    for pdb_id in pdb_list:
        res = get_pdb_resolution(pdb_id)
        if res and res < best_res:
            best_pdb = pdb_id
            best_res = res
    
    return best_pdb


def choose_best_pdb_with_validation(uniprot_id, max_candidates=50):
    """
    选择最佳PDB - 优先用UniProt API, RCSB作为备选
    """
    print(f"🔍 搜索 UniProt {uniprot_id} 的PDB结构...")
    
    # 方法1: UniProt API (优先)
    pdb_list = get_pdbs_from_uniprot(uniprot_id)
    
    # 方法2: 如果失败,尝试RCSB (备选)
    if not pdb_list:
        print("  ⚠️  UniProt API无结果,尝试RCSB...")
        pdb_list = rcsb_search_structures(uniprot_id, max_hits=max_candidates)
    
    if not pdb_list:
        print("❌ 两种方法都未找到PDB结构")
        return None, None, None
    
    # 限制数量
    pdb_list = pdb_list[:max_candidates]
    print(f"✓ 找到 {len(pdb_list)} 个候选PDB")
    print(f"\n开始验证 (共 {len(pdb_list)} 个)...")
    print("-" * 60)
    
    # 验证并选择最佳
    valid_pdbs = []
    
    for idx, pdb_id in enumerate(pdb_list, 1):
        # 显示进度
        print(f"[{idx}/{len(pdb_list)}] 检查 {pdb_id}...", end=" ")
        
        resolution = get_pdb_resolution(pdb_id)
        if resolution is None:
            print("无分辨率数据")
            continue
        
        sifts_data = fetch_sifts_mapping(pdb_id)
        mappings = extract_chain_mappings(sifts_data, pdb_id, uniprot_id)
        
        if mappings:
            coverage = sum(m[2] - m[1] + 1 for m in mappings)
            valid_pdbs.append((pdb_id, resolution, coverage, mappings))
            print(f"✓ 分辨率={resolution:.2f}Å, 覆盖={coverage}aa, {len(mappings)}链")
        else:
            print("✗ 无SIFTS映射")
    
    print("-" * 60)
    print(f"验证完成: {len(valid_pdbs)}/{len(pdb_list)} 个PDB有效\n")
    
    if not valid_pdbs:
        print("❌ 所有候选PDB都没有有效的SIFTS映射!")
        return None, None, None
    
    # 选择最佳
    valid_pdbs.sort(key=lambda x: (x[1], -x[2]))
    best = valid_pdbs[0]
    
    print(f"🏆 最佳PDB: {best[0]}")
    print(f"   分辨率: {best[1]:.2f} Å")
    print(f"   覆盖长度: {best[2]} aa")
    print(f"   链数: {len(best[3])}")
    
    return best[0], best[1], best[3]

# ==================== 第3步: SIFTS映射 ====================
def fetch_sifts_mapping(pdb_id):
    """获取PDB到UniProt的残基映射"""
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot_segments/{pdb_id.lower()}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        print(f"❌ 获取SIFTS映射失败: {e}")
        return None

def get_uniprot_entry_name(uniprot_id):
    """获取UniProt的entry name (如P04637 → P53_HUMAN)"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            return data.get("uniProtkbId")  # 这是entry name
    except:
        pass
    return None


def extract_chain_mappings(sifts_json, pdb_id, uniprot_id):
    """
    从SIFTS JSON提取链映射信息
    支持UniProt ID (P04637) 和 Entry Name (P53_HUMAN)
    """
    if not sifts_json:
        return []
    
    # 获取entry name
    entry_name = get_uniprot_entry_name(uniprot_id)
    
    mappings = []
    data = sifts_json.get(pdb_id.lower(), {})
    
    for uniprot_data in data.get("UniProt", {}).values():
        # 检查两种标识符
        identifier = uniprot_data.get("identifier")
        if identifier not in [uniprot_id, entry_name]:
            continue
        
        for seg in uniprot_data.get("mappings", []):
            chain_id = seg.get("chain_id")
            unp_start = seg.get("unp_start")
            unp_end = seg.get("unp_end")
            pdb_start = seg.get("start", {}).get("residue_number")
            pdb_end = seg.get("end", {}).get("residue_number")
            
            if all([chain_id, unp_start, unp_end, pdb_start, pdb_end]):
                mappings.append((chain_id, unp_start, unp_end, pdb_start, pdb_end))
    
    return mappings

# ==================== 第4步: DSSP标注 ====================
def download_mmcif(pdb_id, cache_dir="pdb_cache"):
    """下载mmCIF文件"""
    Path(cache_dir).mkdir(exist_ok=True)
    cif_path = Path(cache_dir) / f"{pdb_id}.cif"

    if cif_path.exists():
        return cif_path

    urls = [
        f"https://files.rcsb.org/download/{pdb_id}.cif",
        f"https://www.ebi.ac.uk/pdbe/entry-files/download/{pdb_id}.cif"
    ]
    for url in urls:
        try:
           r = requests.get(url, timeout=30)
           if r.status_code == 200:
              cif_path.write_bytes(r.content)
              return cif_path
        except Exception as e:
          print(f"  ⚠️ 下载{pdb_id}失败: {e}")
    return None


def run_dssp(cif_path, chain_id):
    """
    运行DSSP获取二级结构
    返回: {pdb_resseq: ss_code, ...}
    """
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(cif_path.stem, cif_path)
        model = structure[0]

        # 尝试mkdssp，失败则尝试dssp
        try:
            dssp = DSSP(model, str(cif_path), dssp='mkdssp')
        except Exception:
            dssp = DSSP(model, str(cif_path), dssp='dssp')

        ss_dict = {}
        for key, value in dssp.property_dict.items():
            ch, res_id = key
            if ch != chain_id:
                continue
            ss_code = value[2] or 'C'  # DSSP secondary structure code
            pdb_resseq = res_id[1]
            ss_dict[pdb_resseq] = ss_code

        return ss_dict

    except Exception as e:
        print(f"  ⚠️ DSSP失败: {e}")
        return {}


def ss8_to_hec(ss8):
    """将8态二级结构转换为3态(H/E/C)"""
    if ss8 in ('H', 'G', 'I'):  # α-helix, 310-helix, π-helix
        return 'H'
    elif ss8 in ('E', 'B'):  # β-sheet, β-bridge
        return 'E'
    else:
        return 'C'  # Coil/loop


def build_residue_mapping(pdb_id, uniprot_id, entry_name=None):
    """
    构建PDB残基到UniProt位置的精确映射
    返回: {(chain_id, pdb_resseq): uniprot_idx, ...}
    """
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            print(f"❌ 获取PDBe残基映射失败，状态码: {r.status_code}")
            return {}
        data = r.json()
    except Exception as e:
        print(f"❌ 获取PDBe残基映射失败: {e}")
        return {}

    residue_map = {}
    pdb_data = data.get(pdb_id.lower(), {})
    mappings = pdb_data.get("mappings", [])
    for mapping in mappings:
        uid = mapping.get("uniprot_id")
        if uid not in [uniprot_id, entry_name]:
            continue
        residues = mapping.get("residue_mapping", [])
        for res in residues:
            chain = res.get("auth_asym_id")                  # 链ID（author）
            pdb_resseq = res.get("author_residue_number")    # 作者编号（带插码的数字部分）
            unp_pos = res.get("unp_residue_number")          # UniProt位置（1-based）
            if chain and pdb_resseq is not None and unp_pos is not None:
                residue_map[(chain, int(pdb_resseq))] = int(unp_pos) - 1  # 存0-based
    if not residue_map:
        print(f"⚠️ {pdb_id}×{uniprot_id} 未得到任何残基映射（可能无对应链或API字段变更）")
    return residue_map


def map_ss_to_uniprot(uniprot_seq, pdb_id, uniprot_id, cache_dir="pdb_cache"):
    """
    将PDB的DSSP结果映射回UniProt序列（使用精确的残基映射）
    返回: HEC字符串（与uniprot_seq等长）
    """
    cif_path = download_mmcif(pdb_id, cache_dir)
    if not cif_path:
        return None

    # 初始化为全C（未知区域）
    hec_array = ['C'] * len(uniprot_seq)

    # 构建精确的残基映射
    entry_name = get_uniprot_entry_name(uniprot_id)
    residue_map = build_residue_mapping(pdb_id, uniprot_id, entry_name)
    if not residue_map:
        return None

    # 获取所有涉及的链
    chains = set(chain for chain, _ in residue_map.keys())

    # 对每条链运行DSSP
    for chain_id in chains:
        ss_dict = run_dssp(cif_path, chain_id)
        if not ss_dict:
            continue

        # 使用精确映射来赋值
        for (ch, pdb_res), unp_idx in residue_map.items():
            if ch != chain_id:
                continue

            if pdb_res in ss_dict and 0 <= unp_idx < len(uniprot_seq):
                ss8 = ss_dict[pdb_res]
                hec_array[unp_idx] = ss8_to_hec(ss8)

    return ''.join(hec_array)
