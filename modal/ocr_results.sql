-- ocr_results definition
CREATE TABLE
    ocr_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_identifier TEXT UNIQUE NOT NULL,
        extracted_text TEXT,
        source_type TEXT NOT NULL, -- 'S3' or 'LOCAL'
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );

-- Index for faster lookups on extracted_text
CREATE INDEX IF NOT EXISTS idx_ocr_results_extracted_text ON ocr_results (extracted_text);