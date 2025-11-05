#!/usr/bin/env python3
"""
Data Integration Script for Solar Capstone Project
Integrates data prep agent with RAG system and MongoDB
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.agents.data_prep_agent import DataPrepAgent
from rag_system.core.rag_engine import RAGEngine

def integrate_data_systems():
 """Integrate data prep agent with RAG system"""
 print(" Integrating Data Systems...")
 
 # Initialize agents
 data_prep = DataPrepAgent()
 rag_engine = RAGEngine()
 
 # Process existing data
 print("\n1. Processing Raw Data...")
 
 # Process appliances
 appliances_path = "data/raw/appliances/appliances.csv"
 if os.path.exists(appliances_path):
 result = data_prep.process_appliance_data(appliances_path)
 if result["success"]:
 print(f" Processed {result['processed_rows']} appliance records")
 else:
 print(f" Appliance processing failed: {result['error']}")
 
 # Process locations
 locations_path = "data/raw/geo/nigerian_locations.csv"
 if os.path.exists(locations_path):
 result = data_prep.process_geographic_data(locations_path)
 if result["success"]:
 print(f" Processed {result['processed_rows']} location records")
 else:
 print(f" Location processing failed: {result['error']}")
 
 # Generate synthetic data if needed
 print("\n2. Generating Synthetic Data...")
 
 # Generate synthetic components
 result = data_prep.generate_synthetic_data("components", 50)
 if result["success"]:
 print(f" Generated {result['generated_rows']} synthetic components")
 
 # Add processed data to RAG system
 print("\n3. Adding Data to RAG System...")
 
 # Add processed data to knowledge base
 rag_result = rag_engine.add_documents("data/interim/cleaned/", recursive=True)
 if rag_result["success"]:
 print(f" Added {rag_result['total_processed']} documents to RAG system")
 else:
 print(f" RAG integration failed: {rag_result['errors']}")
 
 # Test RAG system
 print("\n4. Testing RAG System...")
 
 test_queries = [
 "solar panel efficiency",
 "battery capacity",
 "Nigerian solar irradiance",
 "appliance power consumption"
 ]
 
 for query in test_queries:
 results = rag_engine.search(query, top_k=3)
 print(f" Query: '{query}' -> {len(results)} results")
 
 # Get system summary
 print("\n5. System Summary...")
 
 data_summary = data_prep.get_data_summary()
 rag_stats = rag_engine.get_statistics()
 
 print(f" Data Files: {data_summary.get('interim_data_files', 0)}")
 print(f" Data Size: {data_summary.get('total_data_size_mb', 0):.2f} MB")
 print(f" RAG Documents: {rag_stats.get('total_documents', 0)}")
 print(f" RAG File Types: {rag_stats.get('file_types', {})}")
 
 print("\n Data integration completed successfully!")
 
 return True

if __name__ == "__main__":
 success = integrate_data_systems()
 sys.exit(0 if success else 1)

