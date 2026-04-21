import os
import pandas as pd
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class CSVSchemaAnalyzer:
    """
    Comprehensive CSV schema analyzer that extracts detailed metadata
    including column descriptions, data types, and table relationships.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.schema = {}
        self.sample_data = {}
        
        # Column descriptions for Instacart dataset
        self.column_descriptions = {
            # Products table
            "product_id": "Unique identifier for each product",
            "product_name": "Name of the product (e.g., 'Organic Bananas')",
            "aisle_id": "Foreign key referencing the aisle where product is located",
            "department_id": "Foreign key referencing the department the product belongs to",
            
            # Aisles table
            "aisle": "Name of the aisle (e.g., 'fresh fruits', 'baking ingredients')",
            
            # Departments table  
            "department": "Name of the department (e.g., 'produce', 'dairy eggs')",
            
            # Orders table
            "order_id": "Unique identifier for each customer order",
            "user_id": "Unique identifier for each customer",
            "eval_set": "Dataset split indicator (train/test)",
            "order_number": "Sequential order number for each user",
            "order_dow": "Day of week order was placed (0=Sunday, 6=Saturday)",
            "order_hour_of_day": "Hour of day order was placed (0-23)",
            "days_since_prior_order": "Days since user's previous order (NULL for first order)",
            
            # Order Products table
            "add_to_cart_order": "Sequence number of product added to cart",
            "reordered": "Whether product was reordered (1=reordered, 0=first time)",
            
            # Users table
            "order_id": "Order identifier (same as in orders table)",
            "product_id": "Product identifier (same as in products table)",
            "add_to_cart_order": "Order in which product was added to cart",
            "reordered": "Binary indicator if product was reordered"
        }
        
        # Table descriptions
        self.table_descriptions = {
            "products": "Contains product information including names, aisles, and departments",
            "aisles": "Product categorization - aisles within departments",
            "departments": "Top-level product categorization (produce, dairy, etc.)",
            "orders": "Customer order metadata including timing and sequence",
            "order_products": "Individual products within each order",
            "order_products_all": "Combined order products with all metadata"
        }
        
        # Data type mappings
        self.sql_type_mapping = {
            'int64': 'INTEGER',
            'float64': 'FLOAT', 
            'object': 'VARCHAR',
            'bool': 'BOOLEAN',
            'datetime64': 'TIMESTAMP'
        }
    
    def analyze_all_csvs(self) -> Dict[str, Any]:
        """
        Analyze all CSV files in the data directory.
        
        Returns:
            Dictionary containing comprehensive schema information
        """
        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory {self.data_dir} does not exist")
            return {}
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        logger.info(f"Found {len(csv_files)} CSV files to analyze")
        
        for csv_file in csv_files:
            table_name = csv_file.replace('.csv', '')
            file_path = os.path.join(self.data_dir, csv_file)
            
            try:
                self.analyze_csv(table_name, file_path)
                logger.info(f"Successfully analyzed {csv_file}")
            except Exception as e:
                logger.error(f"Failed to analyze {csv_file}: {e}")
        
        return self.schema
    
    def analyze_csv(self, table_name: str, file_path: str):
        """
        Analyze a single CSV file and extract comprehensive metadata.
        
        Args:
            table_name: Name of the table
            file_path: Path to the CSV file
        """
        # Read CSV with type inference
        df = pd.read_csv(file_path, nrows=1000)  # Sample for performance
        
        # Get basic info
        row_count = len(pd.read_csv(file_path))  # Full count for row count
        
        # Analyze columns
        columns = {}
        for col in df.columns:
            columns[col] = self._analyze_column(df, col, table_name)
        
        # Generate sample data
        sample_rows = df.head(3).to_dict('records')
        
        # Create comprehensive table info
        self.schema[table_name] = {
            'description': self.table_descriptions.get(table_name, f"Table {table_name}"),
            'row_count': row_count,
            'columns': columns,
            'sample_data': sample_rows,
            'file_path': file_path
        }
        
        self.sample_data[table_name] = sample_rows
    
    def _analyze_column(self, df: pd.DataFrame, column: str, table_name: str) -> Dict[str, Any]:
        """
        Analyze a single column and extract detailed metadata.
        
        Args:
            df: DataFrame containing the column
            column: Column name
            table_name: Name of the table
            
        Returns:
            Dictionary with column metadata
        """
        series = df[column]
        
        # Basic stats
        column_info = {
            'data_type': str(series.dtype),
            'sql_type': self.sql_type_mapping.get(str(series.dtype), 'VARCHAR'),
            'description': self.column_descriptions.get(column, f"Column {column}"),
            'nullable': series.isnull().any(),
            'unique_values': series.nunique(),
            'null_count': series.isnull().sum()
        }
        
        # Add data-specific analysis
        if series.dtype == 'object':
            # Text columns
            column_info.update({
                'max_length': series.astype(str).str.len().max(),
                'avg_length': series.astype(str).str.len().mean(),
                'sample_values': series.dropna().unique()[:5].tolist()
            })
        elif series.dtype in ['int64', 'float64']:
            # Numeric columns
            column_info.update({
                'min_value': series.min(),
                'max_value': series.max(),
                'mean_value': series.mean() if series.dtype != 'int64' or series.nunique() > 1 else None,
                'std_value': series.std() if series.dtype != 'int64' or series.nunique() > 1 else None
            })
        
        # Add relationship hints based on column names
        column_info['relationships'] = self._detect_relationships(column, table_name)
        
        return column_info
    
    def _detect_relationships(self, column: str, table_name: str) -> List[str]:
        """
        Detect potential relationships between tables based on column names.
        
        Args:
            column: Column name
            table_name: Table name
            
        Returns:
            List of relationship hints
        """
        relationships = []
        
        # Common foreign key patterns
        if column.endswith('_id'):
            base_name = column[:-3]
            if base_name != table_name:  # Don't reference self
                relationships.append(f"References {base_name} table")
        
        # Specific known relationships
        if column == 'product_id':
            relationships.append("Foreign key to products table")
        elif column == 'aisle_id':
            relationships.append("Foreign key to aisles table") 
        elif column == 'department_id':
            relationships.append("Foreign key to departments table")
        elif column == 'order_id':
            relationships.append("Foreign key to orders table")
        elif column == 'user_id':
            relationships.append("Foreign key to users table")
        
        return relationships
    
    def generate_comprehensive_schema_text(self) -> str:
        """
        Generate comprehensive schema text for the system prompt.
        
        Returns:
            Formatted schema text with all metadata
        """
        if not self.schema:
            return "No CSV files found for schema analysis."
        
        lines = []
        lines.append("=== COMPREHENSIVE CSV SCHEMA ANALYSIS ===\n")
        
        for table_name, table_info in self.schema.items():
            lines.append(f"📋 TABLE: {table_name.upper()}")
            lines.append(f"   Description: {table_info['description']}")
            lines.append(f"   Row Count: {table_info['row_count']:,}")
            lines.append(f"   File: {table_info['file_path']}")
            lines.append("")
            
            lines.append("   📊 COLUMNS:")
            for col_name, col_info in table_info['columns'].items():
                lines.append(f"   • {col_name}")
                lines.append(f"     Type: {col_info['sql_type']}")
                lines.append(f"     Description: {col_info['description']}")
                
                if col_info.get('relationships'):
                    for rel in col_info['relationships']:
                        lines.append(f"     Relationship: {rel}")
                
                if col_info['nullable']:
                    lines.append(f"     Nullable: Yes ({col_info['null_count']} nulls)")
                
                if 'sample_values' in col_info and col_info['sample_values']:
                    lines.append(f"     Sample Values: {col_info['sample_values']}")
                
                lines.append("")
            
            lines.append("   🎯 SAMPLE DATA (first 3 rows):")
            for i, row in enumerate(table_info['sample_data'][:3], 1):
                lines.append(f"   Row {i}: {row}")
            
            lines.append("\n" + "="*60 + "\n")
        
        return "\n".join(lines)
    
    def get_table_names(self) -> List[str]:
        """Get list of all table names."""
        return list(self.schema.keys())
    
    def get_column_names(self, table_name: str) -> List[str]:
        """Get column names for a specific table."""
        if table_name in self.schema:
            return list(self.schema[table_name]['columns'].keys())
        return []
    
    def get_sample_rows_text(self) -> str:
        """Get sample rows text for all tables."""
        if not self.sample_data:
            return "No sample data available."
        
        lines = []
        for table_name, samples in self.sample_data.items():
            if samples:
                lines.append(f"{table_name.upper()} (sample):")
                for i, row in enumerate(samples[:3], 1):
                    lines.append(f"  Row {i}: {row}")
                lines.append("")
        
        return "\n".join(lines)