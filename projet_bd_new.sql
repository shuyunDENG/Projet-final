-- Clean up existing tables
DROP TABLE IF EXISTS synteny_blocks;
DROP TABLE IF EXISTS homologous_genes;
DROP TABLE IF EXISTS best_hits;
DROP TABLE IF EXISTS blast_hits;
DROP TABLE IF EXISTS features;
DROP TABLE IF EXISTS protein_sequences;
DROP TABLE IF EXISTS cog_assignments;
DROP TABLE IF EXISTS cdsearch;
DROP TABLE IF EXISTS blastp;
DROP TABLE IF EXISTS proteomes;
DROP TABLE IF EXISTS cog_categories;

-- Table for proteome information
CREATE TABLE proteomes (
    assembly_id TEXT UNIQUE,
    version TEXT PRIMARY KEY,
    file_name TEXT,
    organism_name TEXT,
    taxonomy_level TEXT,
    taxonomy_id INTEGER
);

-- Table for protein sequences
CREATE TABLE protein_sequences (
    protein_id TEXT PRIMARY KEY,
    version TEXT,
    sequence TEXT,
    sequence_length INTEGER,
    FOREIGN KEY (version) REFERENCES proteomes(version)
);

-- Table for protein features (from feature tables)
CREATE TABLE features (
    protein_id TEXT PRIMARY KEY,
    version TEXT,
    start_position INTEGER,
    end_position INTEGER,
    product TEXT,
    rank INTEGER,
    FOREIGN KEY (version) REFERENCES proteomes(version)
);

-- Table for all blast hits
CREATE TABLE blast_hits (
    query_id TEXT,
    subject_id TEXT,
    pident REAL,
    alignment_length INTEGER,
    mismatch INTEGER,
    gaps INTEGER,
    qstart INTEGER,
    qend INTEGER,
    sstart INTEGER,
    send INTEGER,
    evalue REAL,
    bitscore REAL,
    PRIMARY KEY (query_id, subject_id)
);

-- Table for best hits (filtered by your criteria)
CREATE TABLE best_hits (
    query_id TEXT,
    subject_id TEXT,
    pident REAL,
    evalue REAL,
    bitscore REAL,
    coverage REAL,
    PRIMARY KEY (query_id, subject_id),
    FOREIGN KEY (query_id) REFERENCES protein_sequences(protein_id),
    FOREIGN KEY (subject_id) REFERENCES protein_sequences(protein_id)
);

-- Table for BLASTP files
CREATE TABLE blastp (
    query_version TEXT,
    subject_version TEXT,
    output_file TEXT PRIMARY KEY,
    FOREIGN KEY (query_version) REFERENCES proteomes(version),
    FOREIGN KEY (subject_version) REFERENCES proteomes(version)
);

-- Table for COG categories
CREATE TABLE cog_categories (
    category_code TEXT PRIMARY KEY,
    category_name TEXT,
    functional_description TEXT
);

-- Table for CDsearch results
CREATE TABLE cdsearch (
    version TEXT,
    output_file TEXT PRIMARY KEY,
    FOREIGN KEY (version) REFERENCES proteomes(version)
);

-- Table for COG assignments
CREATE TABLE cog_assignments (
    protein_id TEXT,
    cog_id TEXT,
    evalue REAL,
    bitscore REAL,
    cog_category TEXT,
    functional_annotation TEXT,
    PRIMARY KEY (protein_id, cog_id),
    FOREIGN KEY (protein_id) REFERENCES protein_sequences(protein_id),
    FOREIGN KEY (cog_category) REFERENCES cog_categories(category_code)
);

-- Table for homologous genes
CREATE TABLE homologous_genes (
    group_id INTEGER,
    protein_id TEXT,
    proteome_version TEXT,
    homology_type TEXT,
    confidence_score REAL,
    PRIMARY KEY (group_id, protein_id),
    FOREIGN KEY (protein_id) REFERENCES protein_sequences(protein_id),
    FOREIGN KEY (proteome_version) REFERENCES proteomes(version)
);

-- Table for synteny blocks
CREATE TABLE synteny_blocks (
    block_id INTEGER PRIMARY KEY,
    genome_a TEXT,
    genome_b TEXT,
    start_a INTEGER,
    end_a INTEGER,
    start_b INTEGER,
    end_b INTEGER,
    gene_count INTEGER,
    orientation TEXT,
    FOREIGN KEY (genome_a) REFERENCES proteomes(version),
    FOREIGN KEY (genome_b) REFERENCES proteomes(version)
);

-- View for dotplot visualization
CREATE VIEW dotplot_view AS
SELECT 
    b.query_id,
    b.subject_id,
    f1.rank AS x_position,
    f2.rank AS y_position,
    b.pident,
    b.evalue,
    b.bitscore
FROM best_hits b
JOIN features f1 ON b.query_id = f1.protein_id
JOIN features f2 ON b.subject_id = f2.protein_id;

-- View for combined homology information
CREATE VIEW homology_summary AS
SELECT 
    h.group_id,
    h.protein_id,
    h.proteome_version,
    p.sequence,
    f.product,
    f.start_position,
    f.end_position,
    f.rank,
    c.cog_id,
    c.functional_annotation,
    c.cog_category
FROM homologous_genes h
JOIN protein_sequences p ON h.protein_id = p.protein_id
JOIN features f ON h.protein_id = f.protein_id
LEFT JOIN cog_assignments c ON h.protein_id = c.protein_id;

-- View for synteny analysis
CREATE VIEW synteny_analysis AS
SELECT 
    s.block_id,
    s.genome_a,
    s.genome_b,
    s.start_a,
    s.end_a,
    s.start_b,
    s.end_b,
    s.gene_count,
    s.orientation,
    p_a.organism_name AS organism_a,
    p_b.organism_name AS organism_b
FROM synteny_blocks s
JOIN proteomes p_a ON s.genome_a = p_a.version
JOIN proteomes p_b ON s.genome_b = p_b.version;

-- Create indexes for better performance
CREATE INDEX idx_blast_qseqid ON blast_hits(query_id);
CREATE INDEX idx_blast_sseqid ON blast_hits(subject_id);
CREATE INDEX idx_features_version ON features(version);
CREATE INDEX idx_protein_sequences_version ON protein_sequences(version);
CREATE INDEX idx_cog_assignments_cog_id ON cog_assignments(cog_id);
CREATE INDEX idx_homologous_genes_group ON homologous_genes(group_id);
CREATE INDEX idx_synteny_blocks_genomes ON synteny_blocks(genome_a, genome_b);
