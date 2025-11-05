# Comprehensive DataPrepAgent - Cleans all datasets for the solar recommender system
import pandas as pd
        # Import tools for working with data tables
import numpy as np
        # Import tools for mathematical calculations
import os
        # Import tools needed for this file
import logging
        # Import tools needed for this file
from datetime import datetime
        # Import tools needed for this file
from typing import Dict, List, Tuple, Optional
        # Import tools needed for this file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPrepAgent:
    # This class represents dataprepagent
    """
    Comprehensive data cleaning agent for solar recommender system.
    Handles: Geo data, Appliances, Marketplace (scraped), Synthetic components
    """
    
    def __init__(self):
        # This stores data information
        self.raw_data_dir = "data/raw"
        # This stores data information
        self.cleaned_data_dir = "data/interim/cleaned"
        # This stores data information
        self.features_data_dir = "data/interim/features"
        
        # Create output directories
        os.makedirs(self.cleaned_data_dir, exist_ok=True)
        os.makedirs(self.features_data_dir, exist_ok=True)
        
        # Data quality metrics
        # This stores important information
        self.quality_report = {}
        
    def clean_all_datasets(self) -> Dict:
    # This function performs a specific task for the solar system
        """
        Main method to clean all datasets
        Returns: Quality report with cleaning statistics
        """
        logger.info("=== STARTING COMPREHENSIVE DATA CLEANING ===")
        
        try:
        # Try to execute the code safely
            # Clean each dataset
            geo_quality = self.clean_geo_data()
            appliances_quality = self.clean_appliances_data()
            # Temporarily skip scraped marketplace cleaning per current plan
            marketplace_quality = {"status": "skipped", "reason": "using_synthetic_only_for_now"}
            synthetic_quality = self.clean_synthetic_data()
            
            # Create unified component catalog
            catalog_quality = self.create_unified_catalog()
            
            # Generate quality report
        # This stores important information
            self.quality_report = {
                "geo_data": geo_quality,
                "appliances_data": appliances_quality,
                "marketplace_data": marketplace_quality,
                "synthetic_data": synthetic_quality,
                "unified_catalog": catalog_quality,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save quality report
            self.save_quality_report()
            
            logger.info("=== DATA CLEANING COMPLETED SUCCESSFULLY ===")
            return self.quality_report
        # Send the result back to the caller
            
        except Exception as e:
        # Handle any errors that might occur
            logger.error(f"Data cleaning failed: {str(e)}")
            raise
    
    def clean_geo_data(self) -> Dict:
    # This function performs a specific task for the solar system
        """Clean geo/weather data with Nigerian context"""
        logger.info("Cleaning geo data...")
        
        input_path = os.path.join(self.raw_data_dir, "geo", "nigeria_cities_weather_data.csv")
        output_path = os.path.join(self.cleaned_data_dir, "geo_cleaned.csv")
        
        if not os.path.exists(input_path):
        # Check a condition
            logger.warning(f"Geo data not found at {input_path}")
            return {"status": "skipped", "reason": "file_not_found"}
        # Send the result back to the caller
        
        # Read data
        df = pd.read_csv(input_path)
        original_rows = len(df)
        
        # Handle missing values
        df = df.dropna(subset=["latitude", "longitude", "city"])
        
        # Convert date column
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        
        # Add Nigerian seasonal classification
        df["season"] = df["date"].apply(self._get_nigerian_season)
        
        # Calculate sun hours from weather data (simplified approach)
        df["avg_sun_hours"] = self._calculate_sun_hours(df)
        
        # Clean numeric columns
        numeric_cols = ["temp", "temp_min", "temp_max", "humidity", "wind_speed", "cloud"]
        for col in numeric_cols:
        # Go through each item in the list
            if col in df.columns:
        # Check if something is present
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Remove outliers (extreme weather values)
        df = self._remove_weather_outliers(df)
        
        # Keep essential columns (remove country column)
        essential_cols = [
            "city", "latitude", "longitude", 
            "temp", "humidity", "wind_speed", "cloud",
            "region", "population", "date", "season", "avg_sun_hours"
        ]
        
        df_clean = df[[col for col in essential_cols if col in df.columns]]
        
        # Sort by region and city alphabetically (handling special characters)
        df_clean = df_clean.sort_values(["region", "city"], key=lambda x: x.str.lower().str.replace(r'[^\w\s-]', '', regex=True))
        
        # Save cleaned data with proper CSV formatting for Excel
        df_clean.to_csv(output_path, index=False, encoding='utf-8-sig', quoting=1)
        
        quality_metrics = {
            "status": "success",
            "original_rows": original_rows,
            "cleaned_rows": len(df_clean),
            "data_loss": f"{((original_rows - len(df_clean)) / original_rows * 100):.1f}%",
            "columns_kept": len(df_clean.columns),
            "date_range": f"{df_clean['date'].min()} to {df_clean['date'].max()}"
        }
        
        logger.info(f"Geo data cleaned: {len(df_clean)} rows saved")
        return quality_metrics
        # Send the result back to the caller
    
    def clean_appliances_data(self) -> Dict:
    # This function performs a specific task for the solar system
        """Clean appliances data with continuation row handling and sorting"""
        logger.info("Cleaning appliances data...")
        
        input_path = os.path.join(self.raw_data_dir, "appliances", "appliances.csv")
        output_path = os.path.join(self.cleaned_data_dir, "appliances_cleaned.csv")
        
        if not os.path.exists(input_path):
        # Check a condition
            logger.warning(f"Appliances data not found at {input_path}")
            return {"status": "skipped", "reason": "file_not_found"}
        # Send the result back to the caller
        
        # Read data
        df = pd.read_csv(input_path)
        original_rows = len(df)
        
        # Handle continuation rows (forward fill Category and Appliance)
        df["Category"] = df["Category"].ffill()
        df["Appliance"] = df["Appliance"].ffill()
        
        # Remove rows where both Category and Appliance are still NaN
        df = df.dropna(subset=["Category", "Appliance"])
        
        # Convert power columns to numeric
        df["min_power_w"] = pd.to_numeric(df["Min Power (W)"], errors="coerce")
        df["max_power_w"] = pd.to_numeric(df["Max Power (W)"], errors="coerce")
        
        # Parse hours ranges robustly (unicode dashes) and keep both min and max
        hours_parsed = df["Typical Hours/day"].apply(self._parse_hours_range)
        df["hours_per_day_min"] = hours_parsed.apply(lambda t: t[0] if t is not None else None)
        df["hours_per_day_max"] = hours_parsed.apply(lambda t: t[1] if t is not None else None)
        
        # Clean surge factor
        df["surge_factor"] = pd.to_numeric(df["Surge Factor"], errors="coerce")
        
        # Remove rows with missing critical data
        df = df.dropna(subset=["min_power_w", "max_power_w", "surge_factor"])
        
        # Validate power ranges (min should be <= max)
        invalid_power = df["min_power_w"] > df["max_power_w"]
        if invalid_power.any():
        # Check a condition
            logger.warning(f"Found {invalid_power.sum()} rows with min_power > max_power, fixing...")
            df.loc[invalid_power, "min_power_w"] = df.loc[invalid_power, "max_power_w"]
        
        # Keep essential columns (drop original power/surge/hours columns, keep processed ones)
        essential_cols = [
            "Category", "Appliance", "Type", 
            "min_power_w", "max_power_w", "hours_per_day_min", "hours_per_day_max", "surge_factor", "Notes"
        ]
        
        df_clean = df[[col for col in essential_cols if col in df.columns]]
        
        # Reorder columns to put Notes beside surge_factor
        column_order = [
            "Category", "Appliance", "Type", 
            "min_power_w", "max_power_w", "hours_per_day_min", "hours_per_day_max", "surge_factor", "Notes"
        ]
        df_clean = df_clean[column_order]
        
        # Sort by Category and Appliance alphabetically (no regex, keep hyphens untouched)
        df_clean = df_clean.sort_values(["Category", "Appliance"])
        
        # Save cleaned data with proper CSV formatting for Excel
        df_clean.to_csv(output_path, index=False, encoding='utf-8-sig', quoting=1)
        
        quality_metrics = {
            "status": "success",
            "original_rows": original_rows,
            "cleaned_rows": len(df_clean),
            "data_loss": f"{((original_rows - len(df_clean)) / original_rows * 100):.1f}%",
            "unique_appliances": df_clean["Appliance"].nunique(),
            "unique_categories": df_clean["Category"].nunique(),
            "power_range": f"{df_clean['min_power_w'].min()}-{df_clean['max_power_w'].max()}W"
        }
        
        logger.info(f"Appliances data cleaned: {len(df_clean)} rows saved")
        return quality_metrics
        # Send the result back to the caller
    
    def clean_marketplace_data(self) -> Dict:
    # This function performs a specific task for the solar system
        """Skip scraped marketplace cleaning for now (manual cleaning required)."""
        logger.info("Skipping scraped marketplace cleaning; using synthetic for now.")
        return {"status": "skipped", "reason": "using_synthetic_only_for_now"}
        # Send the result back to the caller
    
    def clean_synthetic_data(self) -> Dict:
    # This function performs a specific task for the solar system
        """Clean synthetic component data with proper column handling and sorting"""
        logger.info("Cleaning synthetic data...")
        
        synthetic_dir = os.path.join(self.raw_data_dir, "ml", "components")
        quality_metrics = {}
        
        if not os.path.exists(synthetic_dir):
        # Check a condition
            logger.warning(f"Synthetic data directory not found at {synthetic_dir}")
            return {"status": "skipped", "reason": "directory_not_found"}
        # Send the result back to the caller
        
        # Define expected columns for each component type
        component_schemas = {
            "batteries_synth.csv": {
                "required_cols": ["battery_id", "brand", "type", "capacity_ah", "voltage", "derating_factor"],
                "sort_cols": ["brand", "type"]
            },
            "charge_controllers_synth.csv": {
                "required_cols": ["controller_id", "brand", "type", "max_voltage_V", "max_current_A", "derating_factor"],
                "sort_cols": ["brand", "type"]
            },
            "inverters_synth.csv": {
                "required_cols": ["inverter_id", "brand", "rated_power_w", "voltage_input", "voltage_output", "mode", "derating_factor"],
                "sort_cols": ["brand", "mode"]
            },
            "solar_panels_synth.csv": {
                "required_cols": ["panel_id", "brand", "panel_type", "rated_power_w", "voltage", "derating_factor", "cabling_loss", "safety_factor", "length_m", "width_m", "area_m2"],
                "sort_cols": ["brand", "panel_type"]
            }
        }
        
        for file in os.listdir(synthetic_dir):
        # Go through each item in the list
            if file.endswith(('.csv', '.parquet')):
        # Check a condition
                input_path = os.path.join(synthetic_dir, file)
                output_path = os.path.join(self.cleaned_data_dir, f"synthetic_{file}")
                
                try:
        # Try to execute the code safely
                    # Read synthetic data
                    if file.endswith('.csv'):
        # Check a condition
                        df = pd.read_csv(input_path)
                    else:
                        df = pd.read_parquet(input_path)
                    
                    original_rows = len(df)
                    
                    # Skip adding data_source column (not needed)

                    # Normalize column names to a consistent set
                    df.columns = [c.strip() for c in df.columns]

                    # Add price ranges if price_NGN is present
                    price_col = None
                    for cand in ["price_NGN", "price", "Price_NGN", "Price"]:
        # Go through each item in the list
                        if cand in df.columns:
        # Check if something is present
                            price_col = cand
                            break

                    if price_col is not None:
        # Check if the data exists
                        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
                        # Use medium market range multipliers
                        df["price_min"] = (df[price_col] * 0.75).round(0)
                        df["price_max"] = (df[price_col] * 1.25).round(0)
                        # Remove original price column after creating ranges
                        df = df.drop(columns=[price_col])
                    
                    # Handle component-specific cleaning
                    if file in component_schemas:
        # Check if something is present
                        schema = component_schemas[file]
                        
                        # Keep only required columns + price columns (no data_source)
                        required_cols = schema["required_cols"] + ["price_min", "price_max"]
                        available_cols = [col for col in required_cols if col in df.columns]
                        
                        # Remove any columns not in the required list
                        df_clean = df[available_cols].copy()
                        
                        # Sort by specified columns (handling special characters)
                        sort_cols = [col for col in schema["sort_cols"] if col in df_clean.columns]
                        if sort_cols:
        # Check a condition
                            df_clean = df_clean.sort_values(sort_cols, key=lambda x: x.str.lower().str.replace(r'[^\w\s-]', '', regex=True))
                    else:
                        # For unknown files, keep all columns
                        df_clean = df.copy()
                    
                    # Basic cleaning
                    df_clean = df_clean.dropna(how='all')  # Remove completely empty rows
                    
                    # Save cleaned
                    df_clean.to_csv(output_path, index=False)
                    
                    quality_metrics[file] = {
                        "status": "success",
                        "original_rows": original_rows,
                        "cleaned_rows": len(df_clean),
                        "columns": len(df_clean.columns),
                        "kept_columns": list(df_clean.columns)
                    }
                    
                    logger.info(f"Synthetic {file} cleaned: {len(df_clean)} rows saved with {len(df_clean.columns)} columns")
                    
                except Exception as e:
        # Handle any errors that might occur
                    logger.error(f"Failed to clean {file}: {str(e)}")
                    quality_metrics[file] = {"status": "error", "error": str(e)}
        
        return quality_metrics
        # Send the result back to the caller
    
    def create_unified_catalog(self) -> Dict:
    # This function creates new data or objects
        """Create unified component catalog from all sources"""
        logger.info("Creating unified component catalog...")
        
        try:
        # Try to execute the code safely
            # For now, build catalog from synthetic-only
                synthetic_files = [f for f in os.listdir(self.cleaned_data_dir) if f.startswith("synthetic_")]
            if synthetic_files:
        # Check a condition
                unified_list = []
                for file in synthetic_files:
        # Go through each item in the list
                    synthetic_path = os.path.join(self.cleaned_data_dir, file)
                    synthetic_df = pd.read_csv(synthetic_path)
                    
                    # Add component type and standardize columns
                    component_type = file.replace("synthetic_", "").replace(".csv", "")
                    synthetic_df["component_type"] = component_type
                    
                    # Add a unique ID for each component
                    if "battery_id" in synthetic_df.columns:
        # Check if something is present
                        synthetic_df["component_id"] = synthetic_df["battery_id"]
                    elif "controller_id" in synthetic_df.columns:
                        synthetic_df["component_id"] = synthetic_df["controller_id"]
                    elif "inverter_id" in synthetic_df.columns:
                        synthetic_df["component_id"] = synthetic_df["inverter_id"]
                    elif "panel_id" in synthetic_df.columns:
                        synthetic_df["component_id"] = synthetic_df["panel_id"]
                    
                    unified_list.append(synthetic_df)
                
                # Concatenate all components
                unified_df = pd.concat(unified_list, ignore_index=True)
                
                # Create a clean unified catalog with proper column structure
                clean_catalog = []
                
                for component_type in unified_df["component_type"].unique():
        # Go through each item in the list
                    component_df = unified_df[unified_df["component_type"] == component_type].copy()
                    
                    # Base columns for all components (no data_source)
                    base_cols = ["component_id", "brand", "component_type", "price_min", "price_max"]
                    
                    # Add component-specific columns
                    if component_type == "batteries_synth":
        # Check a condition
                        specific_cols = ["type", "capacity_ah", "voltage", "derating_factor"]
                    elif component_type == "charge_controllers_synth":
                        specific_cols = ["type", "max_voltage_V", "max_current_A", "derating_factor"]
                    elif component_type == "inverters_synth":
                        specific_cols = ["rated_power_w", "voltage_input", "voltage_output", "mode", "derating_factor"]
                    elif component_type == "solar_panels_synth":
                        specific_cols = ["panel_type", "rated_power_w", "voltage", "derating_factor", "cabling_loss", "safety_factor", "area_m2"]
                    else:
                        specific_cols = []
                    
                    # Keep only available columns
                    all_cols = base_cols + specific_cols
                    available_cols = [col for col in all_cols if col in component_df.columns]
                    component_clean = component_df[available_cols]
                    
                    clean_catalog.append(component_clean)
                
                # Combine all components
                unified_df = pd.concat(clean_catalog, ignore_index=True)
                
                # Sort by component_type and brand (handling special characters)
                unified_df = unified_df.sort_values(["component_type", "brand"], key=lambda x: x.str.lower().str.replace(r'[^\w\s-]', '', regex=True))
                
                # Save unified catalog
                output_path = os.path.join(self.cleaned_data_dir, "unified_components_catalog.csv")
                unified_df.to_csv(output_path, index=False)
                
                quality_metrics = {
                    "status": "success",
                    "total_components": len(unified_df),
                    "component_types": unified_df["component_type"].nunique()
                }
                
                logger.info(f"Unified catalog created: {len(unified_df)} total components")
                return quality_metrics
        # Send the result back to the caller
            else:
                logger.warning("No component data found to create unified catalog")
                return {"status": "skipped", "reason": "no_data"}
        # Send the result back to the caller
                
        except Exception as e:
        # Handle any errors that might occur
            logger.error(f"Failed to create unified catalog: {str(e)}")
            return {"status": "error", "error": str(e)}
        # Send the result back to the caller
    
    def _get_nigerian_season(self, date) -> str:
    # This function gets information from the system
        """Determine Nigerian season based on date"""
        if pd.isna(date):
        # Check a condition
            return "unknown"
        # Send the result back to the caller
        month = date.month
        if month in [11, 12, 1, 2, 3]:  # Nov-Mar
        # Check if something is present
            return "dry"
        # Send the result back to the caller
        else:  # Apr-Oct
            return "rainy"
        # Send the result back to the caller
    
    def _calculate_sun_hours(self, df: pd.DataFrame) -> pd.Series:
    # This function calculates solar system requirements
        """Calculate sun hours from weather data"""
        # Simplified calculation based on cloud cover and season
        base_sun_hours = np.where(
            df["season"] == "dry", 
            np.random.uniform(5.0, 6.5, len(df)),  # Dry season: more sun
            np.random.uniform(3.5, 5.0, len(df))   # Rainy season: less sun
        )
        
        # Adjust for cloud cover if available
        if "cloud" in df.columns:
        # Check if something is present
            cloud_factor = 1 - (df["cloud"] / 100) * 0.5  # Reduce sun hours by cloud cover
            return base_sun_hours * cloud_factor
        # Send the result back to the caller
        
        return base_sun_hours
        # Send the result back to the caller
    
    def _normalize_dashes(self, s: str) -> str:
    # This function performs a specific task for the solar system
        """Normalize unicode dashes to ASCII hyphen."""
        if pd.isna(s):
        # Check a condition
            return s
        # Send the result back to the caller
        return str(s).replace("\u2013", "-").replace("\u2014", "-").replace("–", "-").replace("—", "-")
        # Send the result back to the caller

    def _parse_hours_range(self, hours_str) -> Tuple[Optional[float], Optional[float]]:
    # This function performs a specific task for the solar system
        """Parse hours range and return (min, max). Handles single values and unicode dashes."""
        if pd.isna(hours_str):
        # Check a condition
            return (None, None)
        # Send the result back to the caller
        s = self._normalize_dashes(hours_str)
        if "-" in s:
        # Check if something is present
            parts = s.split("-")
            if len(parts) == 2:
        # Check the number of items
                try:
        # Try to execute the code safely
                    lo = float(parts[0].strip())
                except:
                    lo = None
                try:
        # Try to execute the code safely
                    hi = float(parts[1].strip())
                except:
                    hi = None
                return (lo, hi)
        # Send the result back to the caller
        else:
            # Single value
            try:
        # Try to execute the code safely
                v = float(s)
                return (v, v)
        # Send the result back to the caller
            except:
                return (None, None)
        # Send the result back to the caller
    
    def _remove_weather_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
    # This function performs a specific task for the solar system
        """Remove extreme weather outliers"""
        weather_cols = ["temp", "humidity", "wind_speed"]
        
        for col in weather_cols:
        # Go through each item in the list
            if col in df.columns:
        # Check if something is present
                # Remove values outside 1st and 99th percentiles
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df = df[(df[col] >= q1) & (df[col] <= q99)]
        
        return df
        # Send the result back to the caller
    
    def save_quality_report(self):
    # This function performs a specific task for the solar system
        """Save data quality report"""
        report_path = os.path.join(self.features_data_dir, "data_quality_report.json")
        
        import json
        # Import tools needed for this file
        with open(report_path, 'w') as f:
            json.dump(self.quality_report, f, indent=2, default=str)
        
        logger.info(f"Quality report saved to {report_path}")

# Main execution
if __name__ == "__main__":
        # Check a condition
    # Initialize and run data cleaning
    data_prep = DataPrepAgent()
    quality_report = data_prep.clean_all_datasets()
    
    # Print summary
    print("\n=== DATA CLEANING SUMMARY ===")
        # Display information to the user
    for dataset, metrics in quality_report.items():
        # Go through each item in the list
        if isinstance(metrics, dict) and "status" in metrics:
        # Check if something is present
            print(f"{dataset}: {metrics['status']}")
        # Display information to the user
            if metrics["status"] == "success":
        # Check a condition
                print(f"  - Rows: {metrics.get('cleaned_rows', 'N/A')}")
        # Display information to the user
                print(f"  - Data loss: {metrics.get('data_loss', 'N/A')}")
        # Display information to the user
        elif isinstance(metrics, dict):
            for component, comp_metrics in metrics.items():
        # Go through each item in the list
                if isinstance(comp_metrics, dict) and "status" in comp_metrics:
        # Check if something is present
                    print(f"{dataset}.{component}: {comp_metrics['status']}")        # Display information to the user
