import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO
from Bio.PDB import MMCIFParser
from Bio.PDB.DSSP import DSSP
from io import StringIO
import time
import warnings
warnings.filterwarnings('ignore')

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


def choose_best_pdb_with_validation(uniprot_id, max_candidates=20):
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


def build_residue_mapping_from_sifts(sifts_json, pdb_id, uniprot_id):
    """
    从SIFTS JSON构建精确的残基映射
    返回: {(chain_id, author_pdb_resseq): uniprot_idx, ...}
    """
    if not sifts_json:
        return {}

    entry_name = get_uniprot_entry_name(uniprot_id)
    residue_map = {}
    data = sifts_json.get(pdb_id.lower(), {})

    for uniprot_data in data.get("UniProt", {}).values():
        identifier = uniprot_data.get("identifier")
        if identifier not in [uniprot_id, entry_name]:
            continue

        for seg in uniprot_data.get("mappings", []):
            chain_id = seg.get("chain_id")
            unp_start = seg.get("unp_start")
            unp_end = seg.get("unp_end")

            pdb_start_info = seg.get("start", {})
            pdb_end_info = seg.get("end", {})
            pdb_start = pdb_start_info.get("residue_number")
            pdb_end = pdb_end_info.get("residue_number")

            if not all([chain_id, unp_start, unp_end, pdb_start, pdb_end]):
                continue

            unp_len = unp_end - unp_start + 1
            pdb_len = pdb_end - pdb_start + 1

            if unp_len != pdb_len:
                print(f"  ⚠️ 警告: 链{chain_id}映射长度不一致 (UniProt:{unp_len} vs PDB:{pdb_len})")
                map_len = min(unp_len, pdb_len)
            else:
                map_len = unp_len

            for i in range(map_len):
                pdb_resseq = pdb_start + i
                unp_idx = (unp_start - 1) + i
                residue_map[(chain_id, pdb_resseq)] = unp_idx

    return residue_map


def ss8_to_hec(ss8):
    """将8态二级结构转换为3态(H/E/C)"""
    if ss8 in ('H', 'G', 'I'):
        return 'H'
    elif ss8 in ('E', 'B'):
        return 'E'
    else:
        return 'C'


def get_residue_numbering_from_mmcif(cif_path, chain_id):
    """
    从mmCIF文件中提取author编号到label编号的映射
    返回: {author_seq_id: label_seq_id, ...}
    """
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('struct', cif_path)

        mmcif_dict = parser._mmcif_dict

        mapping = {}

        if '_atom_site.auth_asym_id' in mmcif_dict:
            auth_chains = mmcif_dict['_atom_site.auth_asym_id']
            auth_seq_ids = mmcif_dict['_atom_site.auth_seq_id']
            label_seq_ids = mmcif_dict['_atom_site.label_seq_id']

            for i in range(len(auth_chains)):
                if auth_chains[i] == chain_id:
                    try:
                        auth_num = int(auth_seq_ids[i])
                        label_num = int(label_seq_ids[i])
                        mapping[auth_num] = label_num
                    except (ValueError, KeyError):
                        continue

        return mapping

    except Exception as e:
        print(f"  ⚠️ 读取mmCIF编号映射失败: {e}")
        return {}


def run_dssp(cif_path, chain_id):
    """
    运行DSSP获取二级结构
    返回: {label_seq_id: ss_code, ...}
    """
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(cif_path.stem, cif_path)
        model = structure[0]

        try:
            dssp = DSSP(model, str(cif_path), dssp='mkdssp')
        except Exception:
            dssp = DSSP(model, str(cif_path), dssp='dssp')

        ss_dict = {}
        for key, value in dssp.property_dict.items():
            ch, res_id = key
            if ch != chain_id:
                continue
            ss_code = value[2] or 'C'
            label_seq_id = res_id[1]
            ss_dict[label_seq_id] = ss_code

        return ss_dict

    except Exception as e:
        print(f"  ⚠️ DSSP失败: {e}")
        return {}


# Helper: 检测DSSP编号体系
def _detect_dssp_numbering(dssp_dict, auth_to_label):
    """
    检测DSSP键使用的是author编号还是label编号。
    返回 'author' 或 'label'。
    逻辑：与auth_to_label的key(autor)与value(label)分别做交集，谁大用谁。
    """
    if not dssp_dict:
        return 'label'
    dssp_keys = set(dssp_dict.keys())
    author_ids = set(auth_to_label.keys())
    label_ids = set(auth_to_label.values())
    inter_author = len(dssp_keys & author_ids)
    inter_label = len(dssp_keys & label_ids)
    # 平手时更常见的是DSSP返回author编号
    if inter_author >= inter_label:
        return 'author'
    return 'label'


def map_ss_to_uniprot_fixed(uniprot_seq, pdb_id, uniprot_id,
                             sifts_json=None, cache_dir="pdb_cache"):
    """
    修复版：正确处理author编号和label编号的转换
    """
    print(f"\n{'='*70}")
    print(f"开始映射 {pdb_id} → UniProt {uniprot_id}")
    print(f"{'='*70}")

    # 下载PDB文件
    cif_path = download_mmcif(pdb_id, cache_dir)
    if not cif_path:
        print(f"❌ 无法下载PDB {pdb_id}")
        return None
    print(f"✓ PDB文件已下载")

    # 获取SIFTS映射
    if sifts_json is None:
        print(f"  → 获取SIFTS映射...")
        sifts_json = fetch_sifts_mapping(pdb_id)

    if not sifts_json:
        print(f"❌ 无法获取SIFTS映射")
        return None

    # SIFTS的残基映射（author编号 -> UniProt位置）
    residue_map = build_residue_mapping_from_sifts(sifts_json, pdb_id, uniprot_id)

    if not residue_map:
        print(f"❌ 未能构建残基映射")
        return None

    print(f"✓ SIFTS映射了 {len(residue_map)} 个残基")

    # 初始化
    hec_array = ['C'] * len(uniprot_seq)
    mask_array = ['0'] * len(uniprot_seq)  # 1=有DSSP真值, 0=无真值(未覆盖)
    chains = set(chain for chain, _ in residue_map.keys())
    print(f"✓ 发现 {len(chains)} 条链: {', '.join(sorted(chains))}")

    mapped_count = 0
    for chain_id in sorted(chains):
        print(f"\n{'-'*70}")
        print(f"处理链 {chain_id}")
        print(f"{'-'*70}")

        # 1. 获取该链的SIFTS映射（author编号 -> UniProt）
        chain_sifts = {pdb_res: unp_idx for (ch, pdb_res), unp_idx in residue_map.items()
                       if ch == chain_id}
        sifts_nums = sorted(chain_sifts.keys())
        print(f"  [SIFTS] {len(chain_sifts)} 个残基")
        print(f"          Author编号范围: {min(sifts_nums)} - {max(sifts_nums)}")
        print(f"          前5个: {sifts_nums[:5]}")

        # 2. 读取author->label编号映射
        auth_to_label = get_residue_numbering_from_mmcif(cif_path, chain_id)
        if not auth_to_label:
            print(f"  [映射] ⚠️ 无法获取编号映射，假设author=label")
            auth_to_label = {k: k for k in chain_sifts.keys()}
        else:
            print(f"  [映射] {len(auth_to_label)} 个残基的author→label映射")
            # 显示几个映射示例
            sample_keys = sorted(auth_to_label.keys())[:5]
            print(f"          示例: ", end="")
            for k in sample_keys:
                print(f"{k}→{auth_to_label[k]}", end=" ")
            print()

        # 3. 运行DSSP（获得label编号的结果）
        dssp_dict = run_dssp(cif_path, chain_id)
        if not dssp_dict:
            print(f"  [DSSP] ❌ 失败")
            continue
        dssp_nums = sorted(dssp_dict.keys())
        print(f"  [DSSP] {len(dssp_dict)} 个残基")
        print(f"         Label编号范围: {min(dssp_nums)} - {max(dssp_nums)}")
        print(f"         前5个: {dssp_nums[:5]}")
        # 2.5 检测DSSP编号体系（author vs label）
        numbering_mode = _detect_dssp_numbering(dssp_dict, auth_to_label)
        print(f"  [编号] 检测到DSSP使用: {numbering_mode} 编号")

        # 4. 转换并映射
        chain_mapped = 0
        mismatches = []

        for auth_num, unp_idx in chain_sifts.items():
            # author -> label
            label_num = auth_to_label.get(auth_num)

            if numbering_mode == 'author':
                dssp_key = auth_num
            else:
                if label_num is None:
                    mismatches.append(f"author={auth_num}无label映射")
                    continue
                dssp_key = label_num

            # 取DSSP的8态，转换到HEC，再写回到对应UniProt索引
            if dssp_key in dssp_dict:
                if 0 <= unp_idx < len(uniprot_seq):
                    ss8 = dssp_dict[dssp_key]
                    hec_array[unp_idx] = ss8_to_hec(ss8)
                    mask_array[unp_idx] = '1'
                    chain_mapped += 1
                else:
                    mismatches.append(f"UniProt索引{unp_idx}越界")
            else:
                mismatches.append(f"DSSP无该编号({numbering_mode}={dssp_key})")

        print(f"\n  [结果] ✓ 成功映射: {chain_mapped}/{len(chain_sifts)} 个残基 ({chain_mapped/len(chain_sifts)*100:.1f}%)")

        if mismatches and len(mismatches) <= 10:
            print(f"  [问题] 未映射的残基: {', '.join(mismatches[:10])}")
        elif len(mismatches) > 10:
            print(f"  [问题] {len(mismatches)} 个残基未映射 (仅显示前10个): {', '.join(mismatches[:10])}")

        mapped_count += chain_mapped

    # 统计
    coverage = sum(1 for c in hec_array if c != 'C')
    coverage_pct = coverage / len(uniprot_seq) * 100

    h_count = sum(1 for c in hec_array if c == 'H')
    e_count = sum(1 for c in hec_array if c == 'E')
    c_count = sum(1 for c in hec_array if c == 'C')

    print(f"\n{'='*70}")
    print(f"最终统计")
    print(f"{'='*70}")
    print(f"  UniProt序列长度: {len(uniprot_seq)} aa")
    print(f"  总映射残基数: {mapped_count}")
    print(f"  结构覆盖: {coverage}/{len(uniprot_seq)} ({coverage_pct:.1f}%)")
    print(f"  二级结构分布:")
    print(f"    H (螺旋): {h_count} ({h_count/len(uniprot_seq)*100:.1f}%)")
    print(f"    E (折叠): {e_count} ({e_count/len(uniprot_seq)*100:.1f}%)")
    print(f"    C (无规): {c_count} ({c_count/len(uniprot_seq)*100:.1f}%)")

    return ''.join(hec_array), ''.join(mask_array)

# ==================== 第5步: 主流程 ====================
def generate_ss_dataset(query="hemoglobin alpha",
                       max_proteins=100,
                       output_csv="ss_dataset.csv"):
    """
    完整流程：生成二级结构数据集
    """
    print("="*60)
    print("🧬 二级结构数据集生成器")
    print("="*60)

    # Step 1: 搜索UniProt
    print(f"\n📍 步骤1: 搜索UniProt (query='{query}')")
    uniprot_ids = uniprot_search_ids(query, size=max_proteins)
    print(f"   找到 {len(uniprot_ids)} 个UniProt ID")

    if not uniprot_ids:
        print("❌ 没有找到任何UniProt ID，退出")
        return

    # Step 2-5: 逐个处理
    results = []
    failed_count = 0

    for i, uid in enumerate(tqdm(uniprot_ids, desc="处理中"), 1):
        try:
            # 获取序列
            seq = fetch_uniprot_fasta(uid)
            if not seq:
                failed_count += 1
                continue

            # 找最佳PDB
            pdb_id = choose_best_pdb(uid)
            if not pdb_id:
                failed_count += 1
                continue

            # 获取SIFTS映射
            sifts = fetch_sifts_mapping(pdb_id)
            mappings = extract_chain_mappings(sifts, pdb_id, uid)
            if not mappings:
                failed_count += 1
                continue

            # DSSP标注
            hec, mask = map_ss_to_uniprot_fixed(seq, pdb_id, uid)
            if not hec or len(hec) != len(seq) or len(mask) != len(seq):
                failed_count += 1
                continue

            # 保存结果
            results.append({
                'uniprot_id': uid,
                'pdb_id': pdb_id,
                'sequence': seq,
                'ss': hec,
                'length': len(seq),
                'mask': mask,
                'coverage': mask.count('1') / len(seq)
            })

            # 避免请求过快
            time.sleep(0.5)

        except Exception as e:
            print(f"\n  ❌ {uid} 处理失败: {e}")
            failed_count += 1
            continue

    # 保存CSV
    if results:
        df = pd.DataFrame(results)
        # 构建按残基展开的长表（token-level），仅包含有DSSP真值的位置
        long_rows = []
        for row in results:
            uid = row['uniprot_id']
            pdb = row['pdb_id']
            seq = row['sequence']
            ss  = row['ss']
            mask = row['mask']
            for i, m in enumerate(mask):
                if m == '1':
                    long_rows.append({
                        'uniprot_id': uid,
                        'pdb_id': pdb,
                        'position': i + 1,
                        'aa': seq[i],
                        'label_hec': ss[i]
                    })
        long_df = pd.DataFrame(long_rows)
        df.to_csv(output_csv, index=False)
        long_csv = output_csv.replace('.csv', '.residues.csv')
        long_df.to_csv(long_csv, index=False)

        print("\n" + "="*60)
        print("✅ 数据集生成完成！")
        print("="*60)
        print(f"📊 成功: {len(results)} 条")
        print(f"❌ 失败: {failed_count} 条")
        print(f"💾 保存到: {output_csv}")
        print(f"🧪 残基级CSV: {long_csv}")
        print("\n数据预览:")
        print(df[['uniprot_id', 'pdb_id', 'length']].head())

        # 统计二级结构分布
        all_ss = ''.join(df['ss'])
        h_pct = all_ss.count('H') / len(all_ss) * 100
        e_pct = all_ss.count('E') / len(all_ss) * 100
        c_pct = all_ss.count('C') / len(all_ss) * 100
        print(f"\n二级结构分布:")
        print(f"  H (Helix): {h_pct:.1f}%")
        print(f"  E (Sheet): {e_pct:.1f}%")
        print(f"  C (Coil):  {c_pct:.1f}%")

        return df
    else:
        print("\n❌ 没有成功处理任何数据！")
        return None


def generate_multiquery_dataset(queries,
                                max_proteins_per_query=30,
                                output_prefix="ss_multi_dataset.csv"):
    """
    多关键词收集器：依次搜索多个 query，合并去重，生成两份 CSV
    - queries: list[str]
    - max_proteins_per_query: 每个 query 抓取的 UniProt 上限
    - output_prefix: 主CSV文件名；残基级CSV会自动用 .residues.csv
    说明：
    * 为了提高成功率，这里使用 choose_best_pdb_with_validation（带SIFTS覆盖检查）
    * 每条记录会多一个 'source_query' 字段，标注来自哪个关键词
    """
    print("="*60)
    print("🧬 多关键词二级结构数据集生成器")
    print("="*60)

    results = []
    failed = 0
    seen_uids = set()  # 跨query去重

    for q_idx, query in enumerate(queries, 1):
        print(f"\n📍 Query[{q_idx}/{len(queries)}]: '{query}'")
        uids = uniprot_search_ids(query, size=max_proteins_per_query)
        print(f"   找到 {len(uids)} 个UniProt ID (限制 {max_proteins_per_query})")

        for uid in tqdm(uids, desc=f"处理中[{query}]"):
            if uid in seen_uids:
                continue
            try:
                seq = fetch_uniprot_fasta(uid)
                if not seq:
                    failed += 1
                    continue

                # 过长序列直接跳过（超大复合物/低分辨率结构常见）
                if len(seq) > 800:
                    print(f"  ⚠️ 序列过长({len(seq)} aa) — 跳过")
                    failed += 1
                    continue

                # 使用带覆盖验证的PDB选择器
                pdb_id, reso, spans = choose_best_pdb_with_validation(uid, max_candidates=20)
                if not pdb_id:
                    failed += 1
                    continue

                sifts = fetch_sifts_mapping(pdb_id)
                mappings = extract_chain_mappings(sifts, pdb_id, uid)
                if not mappings:
                    failed += 1
                    continue

                # 估算覆盖比例（按UniProt区间）
                est_cov = 0
                for (_chain, unp_start, unp_end, _pdb_start, _pdb_end) in mappings:
                    if unp_start is not None and unp_end is not None:
                        est_cov += (unp_end - unp_start + 1)
                cov_ratio = est_cov / len(seq)
                if cov_ratio < 0.4:
                    print(f"  ⚠️ 覆盖偏低({cov_ratio:.2f}) — 跳过")
                    failed += 1
                    continue

                hec, mask = map_ss_to_uniprot_fixed(seq, pdb_id, uid)
                if not hec or len(hec) != len(seq) or len(mask) != len(seq):
                    failed += 1
                    continue

                results.append({
                    'uniprot_id': uid,
                    'pdb_id': pdb_id,
                    'sequence': seq,
                    'ss': hec,
                    'length': len(seq),
                    'mask': mask,
                    'coverage': mask.count('1') / len(seq),
                    'source_query': query
                })
                seen_uids.add(uid)

                time.sleep(0.8)  # 轻限流

            except Exception as e:
                print(f"\n  ❌ {uid} 处理失败: {e}")
                failed += 1
                continue

    if not results:
        print("\n❌ 没有成功处理任何数据！")
        return None

    # 汇总与保存
    import pandas as pd
    df = pd.DataFrame(results)

    # 残基级长表
    long_rows = []
    for row in results:
        uid = row['uniprot_id']
        pdb = row['pdb_id']
        seq = row['sequence']
        ss  = row['ss']
        mask = row['mask']
        src = row['source_query']
        for i, m in enumerate(mask):
            if m == '1':
                long_rows.append({
                    'uniprot_id': uid,
                    'pdb_id': pdb,
                    'position': i + 1,
                    'aa': seq[i],
                    'label_hec': ss[i],
                    'source_query': src
                })
    long_df = pd.DataFrame(long_rows)

    # 保存
    main_csv = output_prefix
    res_csv  = output_prefix.replace('.csv', '.residues.csv')
    df.to_csv(main_csv, index=False)
    long_df.to_csv(res_csv, index=False)

    print("\n" + "="*60)
    print("✅ 多关键词数据集生成完成！")
    print("="*60)
    print(f"📊 成功: {len(results)} 条 | 失败: {failed} 条 | 去重后UID: {len(seen_uids)}")
    print(f"💾 主CSV: {main_csv}")
    print(f"🧪 残基级CSV: {res_csv}")

    # 简要分布
    all_ss = ''.join(df['ss'])
    h_pct = all_ss.count('H') / len(all_ss) * 100
    e_pct = all_ss.count('E') / len(all_ss) * 100
    c_pct = all_ss.count('C') / len(all_ss) * 100
    print(f"\n二级结构分布: H {h_pct:.1f}% | E {e_pct:.1f}% | C {c_pct:.1f}%")

    print("\n来源概览（每个关键词收集到的条目数）:")
    print(df['source_query'].value_counts())

    return df

# ==================== Colab入口 ====================
if __name__ == "__main__":
    # 可在此调整：max_proteins_per_query、候选PDB上限在 choose_best_pdb_with_validation(max_candidates=20)、长度阈值(>800跳过)、覆盖阈值( <0.4 跳过 )
    # 在Colab中运行前先安装依赖
    print("检查依赖...")
    try:
        from Bio.PDB.DSSP import DSSP
        print("✅ Biopython已安装")
    except Exception:
        print("⚠️ 正在安装Biopython...")
        import subprocess
        subprocess.check_call(['pip', 'install', '-q', 'biopython'])

    # 检查DSSP
    import subprocess
    try:
        subprocess.run(['mkdssp', '--version'], capture_output=True, check=True)
        print("✅ DSSP已安装")
    except Exception:
        print("⚠️ 正在安装DSSP...")
        subprocess.check_call(['apt-get', 'install', '-y', '-qq', 'dssp'])

    print("\n开始生成数据集...\n")

    # 多关键词示例（可在此编辑关键词列表）
    queries = [
        # 全α
        "hemoglobin",
        "myoglobin",
        # 高β
        "green fluorescent protein",
        "porin",
        "immunoglobulin domain",
        "beta propeller",
        # α/β
        "triosephosphate isomerase",
        "enolase",
        "aldolase",
    ]
    df = generate_multiquery_dataset(
        queries=queries,
        max_proteins_per_query=50,
        output_prefix="ss_multi_dataset.csv"
    )

    # 显示示例
    if df is not None:
        print("\n示例数据:")
        for idx, row in df.head(3).iterrows():
            print(f"\n序列 {idx+1}:")
            print(f"  UniProt: {row['uniprot_id']}")
            print(f"  PDB: {row['pdb_id']}")
            print(f"  长度: {row['length']}")
            print(f"  序列: {row['sequence'][:50]}...")
            print(f"  结构: {row['ss'][:50]}...")
