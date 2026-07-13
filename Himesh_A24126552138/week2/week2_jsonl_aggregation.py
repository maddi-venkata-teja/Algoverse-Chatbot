"""
WEEK 2: JSONL AGGREGATION
Convert ChromaDB data to JSONL format and aggregate from multiple sources
"""

import json
import jsonlines
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import csv
from data_structures_chromadb import DataStructuresKB


class JSONLAggregator:
    """
    Aggregate data from multiple sources into JSONL format
    Perfect for feeding data into training pipelines
    """
    
    def __init__(self, output_dir: str = "./jsonl_data"):
        """Initialize JSONL aggregator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"✓ JSONL output directory: {self.output_dir}")
    
    # ==================== EXPORT FROM CHROMADB ====================
    
    def export_chromadb_to_jsonl(self,
                                collection_name: str,
                                output_filename: str = "chromadb_export.jsonl") -> Dict:
        """
        Export all documents from ChromaDB to JSONL format
        """
        kb = DataStructuresKB()
        
        # Get all documents
        all_docs = kb.collection.get()
        
        output_path = self.output_dir / output_filename
        
        with jsonlines.open(output_path, mode='w') as writer:
            for doc_id, document, metadata in zip(
                all_docs['ids'],
                all_docs['documents'],
                all_docs['metadatas']
            ):
                jsonl_record = {
                    "id": doc_id,
                    "text": document,
                    "metadata": metadata,
                    "source": metadata.get("source", "unknown"),
                    "source_type": metadata.get("source_type", "unknown"),
                    "topic": metadata.get("topic", "unknown"),
                    "difficulty": metadata.get("difficulty", "unknown"),
                    "exported_at": datetime.now().isoformat()
                }
                writer.write(jsonl_record)
        
        print(f"✓ Exported {len(all_docs['ids'])} documents to {output_path}")
        return {
            "status": "success",
            "documents": len(all_docs['ids']),
            "output_file": str(output_path)
        }
    
    # ==================== AGGREGATE MULTIPLE SOURCES ====================
    
    def aggregate_from_sources(self,
                              source_files: Dict[str, str],
                              output_filename: str = "aggregated.jsonl") -> Dict:
        """
        Aggregate data from multiple source files into single JSONL
        
        Args:
            source_files: Dict of {source_type: file_path}
                         e.g., {"textbook": "textbook.json", "online": "online.csv"}
        """
        output_path = self.output_dir / output_filename
        total_records = 0
        
        with jsonlines.open(output_path, mode='w') as writer:
            for source_type, file_path in source_files.items():
                file_path = Path(file_path)
                
                if not file_path.exists():
                    print(f"⚠️  File not found: {file_path}")
                    continue
                
                # Handle different file formats
                if file_path.suffix == '.json':
                    records = self._read_json(file_path, source_type)
                elif file_path.suffix == '.csv':
                    records = self._read_csv(file_path, source_type)
                elif file_path.suffix == '.txt':
                    records = self._read_txt(file_path, source_type)
                else:
                    continue
                
                for record in records:
                    writer.write(record)
                    total_records += 1
                
                print(f"✓ Aggregated {len(records)} records from {source_type}")
        
        print(f"✓ Total aggregated records: {total_records}")
        return {
            "status": "success",
            "total_records": total_records,
            "output_file": str(output_path)
        }
    
    # ==================== DATA CLEANING & VALIDATION ====================
    
    def clean_and_validate_jsonl(self,
                                input_file: str,
                                output_file: str = "cleaned.jsonl") -> Dict:
        """
        Clean and validate JSONL data
        Removes duplicates, handles missing fields, etc.
        """
        input_path = Path(input_file)
        output_path = self.output_dir / output_file
        
        seen_texts = set()
        valid_records = 0
        invalid_records = 0
        duplicates = 0
        
        with jsonlines.open(output_path, mode='w') as writer:
            with jsonlines.open(input_path) as reader:
                for record in reader:
                    # Validate required fields
                    if not self._validate_record(record):
                        invalid_records += 1
                        continue
                    
                    # Check for duplicates
                    text = record.get('text', '')
                    if text in seen_texts:
                        duplicates += 1
                        continue
                    
                    seen_texts.add(text)
                    
                    # Clean and normalize
                    cleaned = self._clean_record(record)
                    writer.write(cleaned)
                    valid_records += 1
        
        print(f"✓ Valid records: {valid_records}")
        print(f"⚠️  Invalid records: {invalid_records}")
        print(f"⚠️  Duplicates removed: {duplicates}")
        
        return {
            "status": "success",
            "valid_records": valid_records,
            "invalid_records": invalid_records,
            "duplicates_removed": duplicates,
            "output_file": str(output_path)
        }
    
    # ==================== DATASET SPLITTING ====================
    
    def split_train_val_test(self,
                            input_file: str,
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15) -> Dict:
        """
        Split JSONL data into train/val/test sets
        """
        input_path = Path(input_file)
        
        # Read all records
        records = []
        with jsonlines.open(input_path) as reader:
            records = list(reader)
        
        total = len(records)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        # Split data
        train_data = records[:train_size]
        val_data = records[train_size:train_size + val_size]
        test_data = records[train_size + val_size:]
        
        # Save splits
        splits = {
            "train": ("train.jsonl", train_data),
            "val": ("val.jsonl", val_data),
            "test": ("test.jsonl", test_data)
        }
        
        output_files = {}
        for split_name, (filename, data) in splits.items():
            output_path = self.output_dir / filename
            with jsonlines.open(output_path, mode='w') as writer:
                for record in data:
                    writer.write(record)
            
            output_files[split_name] = str(output_path)
            print(f"✓ {split_name.upper()}: {len(data)} records → {filename}")
        
        return {
            "status": "success",
            "total_records": total,
            "train": {"count": len(train_data), "file": output_files["train"]},
            "val": {"count": len(val_data), "file": output_files["val"]},
            "test": {"count": len(test_data), "file": output_files["test"]}
        }
    
    # ==================== FORMATTING & CONVERSION ====================
    
    def format_for_training(self,
                           input_file: str,
                           output_file: str = "training_format.jsonl",
                           instruction_template: str = None) -> Dict:
        """
        Format JSONL data for training pipelines
        Adds instruction/response fields for fine-tuning
        """
        input_path = Path(input_file)
        output_path = self.output_dir / output_file
        
        if not instruction_template:
            instruction_template = "Based on the following information, answer: {text}"
        
        with jsonlines.open(output_path, mode='w') as writer:
            with jsonlines.open(input_path) as reader:
                for record in reader:
                    training_record = {
                        "instruction": instruction_template.format(
                            text=record.get('text', '')[:100]
                        ),
                        "input": "",
                        "output": record.get('text', ''),
                        "metadata": record.get('metadata', {}),
                        "source": record.get('source', 'unknown')
                    }
                    writer.write(training_record)
        
        print(f"✓ Formatted for training: {output_path}")
        return {"status": "success", "output_file": str(output_path)}
    
    # ==================== STATISTICS & ANALYSIS ====================
    
    def analyze_jsonl(self, input_file: str) -> Dict:
        """
        Analyze JSONL file for data quality and statistics
        """
        input_path = Path(input_file)
        
        stats = {
            "total_records": 0,
            "avg_text_length": 0,
            "sources": {},
            "topics": {},
            "difficulties": {},
            "missing_fields": {},
            "text_lengths": []
        }
        
        with jsonlines.open(input_path) as reader:
            for record in reader:
                stats["total_records"] += 1
                
                # Track text lengths
                text_len = len(record.get('text', ''))
                stats["text_lengths"].append(text_len)
                
                # Count sources
                source = record.get('source', 'unknown')
                stats["sources"][source] = stats["sources"].get(source, 0) + 1
                
                # Count topics
                topic = record.get('topic', 'unknown')
                stats["topics"][topic] = stats["topics"].get(topic, 0) + 1
                
                # Count difficulties
                difficulty = record.get('difficulty', 'unknown')
                stats["difficulties"][difficulty] = stats["difficulties"].get(difficulty, 0) + 1
                
                # Check for missing fields
                for field in ['text', 'metadata', 'source']:
                    if field not in record or not record[field]:
                        stats["missing_fields"][field] = stats["missing_fields"].get(field, 0) + 1
        
        # Calculate average length
        if stats["text_lengths"]:
            stats["avg_text_length"] = sum(stats["text_lengths"]) / len(stats["text_lengths"])
        
        return stats
    
    # ==================== PRIVATE METHODS ====================
    
    @staticmethod
    def _validate_record(record: Dict) -> bool:
        """Check if record has required fields"""
        required_fields = ['text']
        return all(field in record and record[field] for field in required_fields)
    
    @staticmethod
    def _clean_record(record: Dict) -> Dict:
        """Clean and normalize a record"""
        return {
            "id": record.get('id', ''),
            "text": record.get('text', '').strip(),
            "metadata": record.get('metadata', {}),
            "source": record.get('source', 'unknown'),
            "source_type": record.get('source_type', 'unknown'),
            "topic": record.get('topic', 'unknown'),
            "difficulty": record.get('difficulty', 'unknown')
        }
    
    @staticmethod
    def _read_json(file_path: Path, source_type: str) -> List[Dict]:
        """Read JSON file"""
        with open(file_path) as f:
            data = json.load(f)
        
        records = []
        if isinstance(data, list):
            for item in data:
                records.append({
                    "text": item.get('text', str(item)),
                    "source": source_type,
                    "source_type": source_type,
                    "metadata": item.get('metadata', {})
                })
        return records
    
    @staticmethod
    def _read_csv(file_path: Path, source_type: str) -> List[Dict]:
        """Read CSV file"""
        records = []
        with open(file_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "text": row.get('text', row.get('content', str(row))),
                    "source": source_type,
                    "source_type": source_type,
                    "metadata": dict(row)
                })
        return records
    
    @staticmethod
    def _read_txt(file_path: Path, source_type: str) -> List[Dict]:
        """Read TXT file (one document per line)"""
        records = []
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append({
                        "text": line,
                        "source": source_type,
                        "source_type": source_type,
                        "metadata": {}
                    })
        return records


# ==================== USAGE EXAMPLES ====================

def main():
    """Example usage"""
    
    print("="*80)
    print("WEEK 2: JSONL AGGREGATION EXAMPLES")
    print("="*80)
    
    aggregator = JSONLAggregator()
    
    # Example 1: Export from ChromaDB
    print("\n[Example 1] Export ChromaDB to JSONL")
    result = aggregator.export_chromadb_to_jsonl("data_structures")
    print(f"Exported: {result}")
    
    # Example 2: Analyze the exported data
    print("\n[Example 2] Analyze JSONL Data")
    stats = aggregator.analyze_jsonl("./jsonl_data/chromadb_export.jsonl")
    print(f"Total Records: {stats['total_records']}")
    print(f"Avg Text Length: {stats['avg_text_length']:.0f} chars")
    print(f"Sources: {stats['sources']}")
    print(f"Topics: {stats['topics']}")
    print(f"Difficulties: {stats['difficulties']}")
    
    # Example 3: Clean and validate
    print("\n[Example 3] Clean and Validate Data")
    result = aggregator.clean_and_validate_jsonl("./jsonl_data/chromadb_export.jsonl")
    print(f"Valid: {result['valid_records']}")
    print(f"Invalid: {result['invalid_records']}")
    
    # Example 4: Split into train/val/test
    print("\n[Example 4] Split Data (Train/Val/Test)")
    result = aggregator.split_train_val_test("./jsonl_data/cleaned.jsonl")
    print(f"Train: {result['train']['count']}")
    print(f"Val: {result['val']['count']}")
    print(f"Test: {result['test']['count']}")
    
    # Example 5: Format for training
    print("\n[Example 5] Format for Training Pipeline")
    result = aggregator.format_for_training("./jsonl_data/cleaned.jsonl")
    print(f"Formatted: {result['output_file']}")


if __name__ == "__main__":
    main()
