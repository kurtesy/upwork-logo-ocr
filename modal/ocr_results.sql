-- ocr_results definition
CREATE TABLE
    ocr_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_identifier TEXT UNIQUE NOT NULL,
        extracted_text TEXT,
        source_type TEXT NOT NULL, -- 'S3' or 'LOCAL'
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );