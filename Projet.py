import sqlite3
# ❗注意：本项目中 protein.faa 文件仅用于提取蛋白ID 与 序列长度以计算 coverage，
# 在生成 dotplot 和分析 synteny 时不会直接使用这些序列。
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class GenomeComparisonSystem:
    def __init__(self, db_path: str):
        """Initialize the genome comparison system with a database connection."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.connect_db()
    
    def connect_db(self):
        """Connect to the SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
    
    def close_db(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
    
    def init_database(self):
        """Initialize the database with schema from SQL file."""
        with open('genome_comparison.sql', 'r') as f:
            sql_script = f.read()
        self.cursor.executescript(sql_script)
        self.conn.commit()
    
    def parse_fasta(self, fasta_file: str) -> Dict[str, str]:
        """Parse a FASTA file and return a dictionary of sequences."""
        sequences = {}
        current_id = None
        current_seq = []
        
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id:
                        sequences[current_id] = ''.join(current_seq)
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
        
        if current_id:
            sequences[current_id] = ''.join(current_seq)
        
        return sequences
    
    def insert_protein_sequences(self, fasta_file: str, version: str):
        """Insert protein sequences from a FASTA file into the database."""
        sequences = self.parse_fasta(fasta_file)
        
        for protein_id, sequence in sequences.items():
            self.cursor.execute("""
                INSERT OR IGNORE INTO protein_sequences (protein_id, version, sequence, sequence_length)
                VALUES (?, ?, ?, ?)
            """, (protein_id, version, sequence, len(sequence)))
        
        self.conn.commit()
    
    def parse_feature_table(self, feature_file: str, version: str) -> Dict[str, Dict]:
        """Parse a feature table file and return protein features."""
        features = {}

        with open(feature_file, 'r') as f:
            for idx, line in enumerate(f, 1):
                if line.startswith("#"):
                   continue
                parts = line.strip().split("\t")
                if len(parts) < 11:
                   continue
                if parts[0] == "CDS" and parts[1] == "with_protein":
                    try:
                        protein_id = parts[10]
                        start = int(parts[7])
                        end = int(parts[8])
                        product = parts[13] if len(parts) > 13 else ""
                        features[protein_id] = {
                        "start": start,
                        "end": end,
                        "product": product,
                        "rank": idx
                       }
                    except Exception as e:
                        print("跳过行（解析失败）:", e, "--", line)
            return features
    
    def insert_features(self, feature_file: str, version: str):
        """Insert protein features from a feature table into the database."""
        features = self.parse_feature_table(feature_file, version)
        
        for protein_id, info in features.items():
            self.cursor.execute("""
                INSERT OR IGNORE INTO features (protein_id, version, start_position, end_position, product, rank)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (protein_id, version, info['start'], info['end'], info['product'], info['rank']))
        
        self.conn.commit()
    
    def parse_blast_output(self, blast_file: str) -> List[Dict]:
        """Parse tabular BLAST output (outfmt 6) and return hit information."""
        blast_hits = []
        with open(blast_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 12:
                    continue
                blast_hits.append({
                    'query_id': parts[0],
                    'subject_id': parts[1],
                    'pident': float(parts[2]),
                    'alignment_length': int(parts[3]),
                    'mismatches': int(parts[4]),
                    'gap_opens': int(parts[5]),
                    'qstart': int(parts[6]),
                    'qend': int(parts[7]),
                    'sstart': int(parts[8]),
                    'send': int(parts[9]),
                    'evalue': float(parts[10]),
                    'bitscore': float(parts[11])
                })
        return blast_hits
    
    def insert_blast_hits(self, blast_file: str):
        """Insert BLAST hits from a BLASTP output file into the database."""
        blast_hits = self.parse_blast_output(blast_file)
        
        for hit in blast_hits:
            self.cursor.execute("""
                INSERT OR IGNORE INTO blast_hits (query_id, subject_id, evalue, bitscore)
                VALUES (?, ?, ?, ?)
            """, (hit['query_id'], hit['subject_id'], hit['evalue'], hit['bitscore']))
        
        self.conn.commit()
    
    def calculate_best_hits(self, identity_thresh: float = 30.0, evalue_thresh: float = 1e-5, 
                          bitscore_thresh: float = 50.0):
        """Calculate and store best hits based on criteria."""
        self.cursor.execute("""
            INSERT OR REPLACE INTO best_hits (query_id, subject_id, pident, evalue, bitscore, coverage)
            SELECT 
                bh.query_id,
                bh.subject_id,
                bh.pident,
                bh.evalue,
                bh.bitscore,
                CAST(bh.alignment_length AS FLOAT) / NULLIF(ps.sequence_length, 0) AS coverage
            FROM blast_hits bh
            JOIN protein_sequences ps ON bh.query_id = ps.protein_id
            WHERE bh.pident >= ? AND bh.evalue <= ? AND bh.bitscore >= ?
        """, (identity_thresh, evalue_thresh, bitscore_thresh))
        
        self.conn.commit()
    
    def parse_cog_output(self, cog_file: str) -> List[Dict]:
        """Parse COG search output file and return COG assignments."""
        cog_assignments = []
        
        with open(cog_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    protein_id = parts[0]
                    cog_info = parts[1]
                    evalue_str = parts[2]
                    
                    try:
                        evalue = float(evalue_str)
                    except ValueError:
                        evalue = None
                    cod_id = cog_info.split(',')[0] 
                    category = cog_info.split('[')[-1].split(']')[0] if '[' in cog_info else ''
                    annotation = cog_info    
                    cog_assignments.append({
                        'protein_id': query_id,
                        'cog_id': cog_id,
                        'evalue': evalue,
                        'bitscore': 0.0, #No Bitscore in this fiche
                        'category': category,
                        'annotation': annotation
                    })
        
        return cog_assignments
    
    def insert_cog_assignments(self, cog_file: str):
        """Insert COG assignments from a COG search output file into the database."""
        cog_assignments = self.parse_cog_output(cog_file)
        
        for assignment in cog_assignments:
            self.cursor.execute("""
                INSERT OR IGNORE INTO cog_assignments 
                (protein_id, cog_id, evalue, bitscore, cog_category, functional_annotation)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (assignment['protein_id'], assignment['cog_id'], assignment['evalue'],
                  assignment['bitscore'], assignment['category'], assignment['annotation']))
        
        self.conn.commit()
    
    def detect_synteny_blocks(self, genome_a: str, genome_b: str, window_size: int = 5):
        """Detect synteny blocks between two genomes."""
        self.cursor.execute("""
            SELECT 
                f1.protein_id AS gene_a,
                f2.protein_id AS gene_b,
                f1.rank AS rank_a,
                f2.rank AS rank_b
            FROM best_hits bh
            JOIN features f1 ON bh.query_id = f1.protein_id
            JOIN features f2 ON bh.subject_id = f2.protein_id
            WHERE f1.version = ? AND f2.version = ?
            ORDER BY f1.rank, f2.rank
        """, (genome_a, genome_b))
        
        hits = self.cursor.fetchall()
        synteny_blocks = []
        current_block = []
        
        for i in range(len(hits)):
            if i == 0 or self._is_in_same_block(hits[i], hits[i-1], window_size):
                current_block.append(hits[i])
            else:
                if len(current_block) >= 3:  # Minimum block size
                    synteny_blocks.append(current_block)
                current_block = [hits[i]]
        
        if len(current_block) >= 3:
            synteny_blocks.append(current_block)
        
        return synteny_blocks
    
    def _is_in_same_block(self, hit1, hit2, window_size: int) -> bool:
        """Check if two hits belong to the same synteny block."""
        rank_diff_a = abs(hit1[2] - hit2[2])
        rank_diff_b = abs(hit1[3] - hit2[3])
        return rank_diff_a <= window_size and rank_diff_b <= window_size
    
    def insert_synteny_blocks(self, genome_a: str, genome_b: str, synteny_blocks: List[List]):
        """Insert detected synteny blocks into the database."""
        for idx, block in enumerate(synteny_blocks, 1):
            start_a = min(hit[2] for hit in block)
            end_a = max(hit[2] for hit in block)
            start_b = min(hit[3] for hit in block)
            end_b = max(hit[3] for hit in block)
            orientation = 'forward' if block[0][3] < block[-1][3] else 'reverse'
            
            self.cursor.execute("""
                INSERT INTO synteny_blocks 
                (block_id, genome_a, genome_b, start_a, end_a, start_b, end_b, gene_count, orientation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (idx, genome_a, genome_b, start_a, end_a, start_b, end_b, len(block), orientation))
        
        self.conn.commit()
    
    def generate_dotplot(self, genome_a: str, genome_b: str, output_file: str = None,
                        identity_thresh: float = 30.0, evalue_thresh: float = 1e-5):
        """Generate a dotplot visualization for two genomes."""
        self.cursor.execute("""
            SELECT x_position, y_position, pident
            FROM dotplot_view
            WHERE query_id IN (SELECT protein_id FROM features WHERE version = ?)
            AND subject_id IN (SELECT protein_id FROM features WHERE version = ?)
            AND pident >= ? AND evalue <= ?
        """, (genome_a, genome_b, identity_thresh, evalue_thresh))
        
        points = self.cursor.fetchall()
        
        if not points:
            print(f"No points found for genomes {genome_a} and {genome_b} with given thresholds")
            return
        
        x_coords = [pt[0] for pt in points]
        y_coords = [pt[1] for pt in points]
        colors = [pt[2]/100.0 for pt in points]  # Normalize to 0-1 for colormap
        
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(x_coords, y_coords, c=colors, cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(scatter, label='Percent Identity')
        plt.xlabel(f'{genome_a} gene rank')
        plt.ylabel(f'{genome_b} gene rank')
        plt.title(f'Dotplot: {genome_a} vs {genome_b}')
        
        # Highlight synteny blocks
        self.cursor.execute("""
            SELECT start_a, end_a, start_b, end_b
            FROM synteny_blocks
            WHERE genome_a = ? AND genome_b = ?
        """, (genome_a, genome_b))
        
        synteny_blocks = self.cursor.fetchall()
        for block in synteny_blocks:
            plt.plot([block[0], block[1]], [block[2], block[3]], 'r-', linewidth=2, alpha=0.5)
        
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()
    
    def get_homologous_groups(self) -> Dict[int, List[str]]:
        """Retrieve homologous gene groups from the database."""
        self.cursor.execute("""
            SELECT group_id, protein_id
            FROM homologous_genes
            ORDER BY group_id
        """)
        
        groups = {}
        for group_id, protein_id in self.cursor.fetchall():
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(protein_id)
        
        return groups
    
    def get_protein_info(self, protein_id: str) -> Dict:
        """Get detailed information about a protein."""
        self.cursor.execute("""
            SELECT ps.sequence, f.start_position, f.end_position, f.product, f.rank
            FROM protein_sequences ps
            JOIN features f ON ps.protein_id = f.protein_id
            WHERE ps.protein_id = ?
        """, (protein_id,))
        
        result = self.cursor.fetchone()
        if result:
            return {
                'sequence': result[0],
                'start': result[1],
                'end': result[2],
                'product': result[3],
                'rank': result[4]
            }
        return None
    
    def get_cog_info(self, protein_id: str) -> List[Dict]:
        """Get COG assignments for a protein."""
        self.cursor.execute("""
            SELECT cog_id, evalue, bitscore, cog_category, functional_annotation
            FROM cog_assignments
            WHERE protein_id = ?
        """, (protein_id,))
        
        cog_hits = []
        for row in self.cursor.fetchall():
            cog_hits.append({
                'cog_id': row[0],
                'evalue': row[1],
                'bitscore': row[2],
                'category': row[3],
                'annotation': row[4]
            })
        
        return cog_hits
    
    def get_similarity_stats(self, genome_a: str, genome_b: str) -> Dict:
        """Calculate similarity statistics between two genomes."""
        self.cursor.execute("""
            SELECT 
                COUNT(*) as total_matches,
                AVG(pident) as avg_identity,
                MIN(pident) as min_identity,
                MAX(pident) as max_identity,
                AVG(bitscore) as avg_bitscore
            FROM best_hits bh
            JOIN features f1 ON bh.query_id = f1.protein_id
            JOIN features f2 ON bh.subject_id = f2.protein_id
            WHERE f1.version = ? AND f2.version = ?
        """, (genome_a, genome_b))
        
        row = self.cursor.fetchone()
        
        self.cursor.execute("""
            SELECT COUNT(*) as total_genes
            FROM features
            WHERE version = ?
        """, (genome_a,))
        genes_a = self.cursor.fetchone()[0]
        
        self.cursor.execute("""
            SELECT COUNT(*) as total_genes
            FROM features
            WHERE version = ?
        """, (genome_b,))
        genes_b = self.cursor.fetchone()[0]
        
        return {
            'total_matches': row[0],
            'avg_identity': row[1],
            'min_identity': row[2],
            'max_identity': row[3],
            'avg_bitscore': row[4],
            'genes_a': genes_a,
            'genes_b': genes_b,
            'coverage_a': row[0] / genes_a if genes_a > 0 else 0,
            'coverage_b': row[0] / genes_b if genes_b > 0 else 0
        }

if __name__ == "__main__":
    # 📦 Initialize the comparison system and database
    gcs = GenomeComparisonSystem("comparaison.db")
    gcs.init_database()

    # ✅ Insert data for genome A: IAI1
    gcs.insert_protein_sequences("protein/IAI1.faa", "GCA_000026265.1_ASM2626v1")
    gcs.insert_features("Feature_table/GCA_000026265.1_ASM2626v1_feature_table.txt", "GCA_000026265.1_ASM2626v1")
    gcs.insert_cog_assignments("COG/IAI1_COG.out")

    # ✅ Insert data for genome B: MG1655
    gcs.insert_protein_sequences("protein/MG1655.faa", "GCA_000005845.2_ASM584v2")
    gcs.insert_features("Feature_table/GCA_000005845.2_ASM584v2_feature_table.txt", "GCA_000005845.2_ASM584v2")
    gcs.insert_cog_assignments("COG/MG1655_COG.out")

    # 🔄 Insert BLAST hits and compute best hits
    gcs.insert_blast_hits("Blastp/IAI1_vs_MG1655.out")
    gcs.calculate_best_hits()

    # 🔎 Detect synteny blocks
    blocks = gcs.detect_synteny_blocks("GCA_000026265.1_ASM2626v1", "GCA_000005845.2_ASM584v2")
    gcs.insert_synteny_blocks("GCA_000026265.1_ASM2626v1", "GCA_000005845.2_ASM584v2", blocks)

    # 📊 Generate dotplot
    gcs.generate_dotplot("GCA_000026265.1_ASM2626v1", "GCA_000005845.2_ASM584v2")

    # 📈 Print similarity statistics
    stats = gcs.get_similarity_stats("GCA_000026265.1_ASM2626v1", "GCA_000005845.2_ASM584v2")
    print("🧬 Similarity Stats:", stats)

    # ❌ Close connection
    gcs.close_db()