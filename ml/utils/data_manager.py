import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import sqlite3

logger = logging.getLogger(__name__)

class DataManager:
    """
    Manages data storage and retrieval for tumor prediction results.
    """
    
    def __init__(self, db_path: str = "tumor_predictions.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database for storing predictions."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    is_tumor BOOLEAN NOT NULL,
                    non_tumor_prob REAL NOT NULL,
                    tumor_prob REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    model_version TEXT DEFAULT 'ResNet50_v1',
                    processing_time REAL,
                    image_size TEXT,
                    notes TEXT
                )
            ''')
            
            # Create analysis sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    total_images INTEGER DEFAULT 0,
                    tumor_detected INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def save_prediction(self, image_path: str, prediction_result: Dict, 
                       user_id: str = None, session_id: str = None,
                       processing_time: float = None) -> int:
        """
        Save prediction result to database.
        
        Args:
            image_path: Path to the analyzed image
            prediction_result: Dictionary containing prediction results
            user_id: Optional user identifier
            session_id: Optional session identifier
            processing_time: Time taken for processing in seconds
            
        Returns:
            ID of the saved record
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            image_name = os.path.basename(image_path)
            
            cursor.execute('''
                INSERT INTO predictions 
                (image_path, image_name, predicted_class, confidence, is_tumor,
                 non_tumor_prob, tumor_prob, risk_level, user_id, processing_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                image_path,
                image_name,
                prediction_result['predicted_class'],
                prediction_result['confidence'],
                prediction_result['is_tumor'],
                prediction_result['probabilities']['non_tumor'],
                prediction_result['probabilities']['tumor'],
                prediction_result['risk_level'],
                user_id,
                processing_time
            ))
            
            record_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Prediction saved with ID: {record_id}")
            return record_id
            
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            raise
    
    def get_predictions(self, user_id: str = None, limit: int = 100) -> List[Dict]:
        """
        Retrieve predictions from database.
        
        Args:
            user_id: Optional filter by user ID
            limit: Maximum number of records to return
            
        Returns:
            List of prediction records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if user_id:
                cursor.execute('''
                    SELECT * FROM predictions 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (user_id, limit))
            else:
                cursor.execute('''
                    SELECT * FROM predictions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving predictions: {str(e)}")
            return []
    
    def get_prediction_stats(self, user_id: str = None) -> Dict:
        """
        Get statistics about predictions.
        
        Args:
            user_id: Optional filter by user ID
            
        Returns:
            Dictionary with statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            base_query = '''
                SELECT 
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN is_tumor = 1 THEN 1 ELSE 0 END) as tumor_detected,
                    AVG(confidence) as avg_confidence,
                    MIN(timestamp) as first_prediction,
                    MAX(timestamp) as last_prediction
                FROM predictions
            '''
            
            if user_id:
                cursor.execute(base_query + ' WHERE user_id = ?', (user_id,))
            else:
                cursor.execute(base_query)
            
            result = cursor.fetchone()
            
            stats = {
                'total_predictions': result[0] or 0,
                'tumor_detected': result[1] or 0,
                'non_tumor_detected': (result[0] or 0) - (result[1] or 0),
                'avg_confidence': result[2] or 0,
                'first_prediction': result[3],
                'last_prediction': result[4]
            }
            
            if stats['total_predictions'] > 0:
                stats['tumor_rate'] = stats['tumor_detected'] / stats['total_predictions']
            else:
                stats['tumor_rate'] = 0
            
            conn.close()
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}
    
    def export_predictions_csv(self, output_path: str, user_id: str = None) -> bool:
        """
        Export predictions to CSV file.
        
        Args:
            output_path: Path for the output CSV file
            user_id: Optional filter by user ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            predictions = self.get_predictions(user_id=user_id, limit=None)
            if not predictions:
                logger.warning("No predictions to export")
                return False
            
            df = pd.DataFrame(predictions)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Predictions exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting predictions: {str(e)}")
            return False
    
    def create_analysis_session(self, user_id: str = None) -> str:
        """
        Create a new analysis session.
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Session ID
        """
        try:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO analysis_sessions (session_id, user_id)
                VALUES (?, ?)
            ''', (session_id, user_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Analysis session created: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            return None

def save_prediction_report(predictions: List[Dict], output_path: str, 
                          include_summary: bool = True) -> bool:
    """
    Save detailed prediction report as JSON.
    
    Args:
        predictions: List of prediction results
        output_path: Path for the output file
        include_summary: Whether to include summary statistics
        
    Returns:
        True if successful, False otherwise
    """
    try:
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_predictions': len(predictions),
            'predictions': predictions
        }
        
        if include_summary and predictions:
            tumor_count = sum(1 for p in predictions if p.get('is_tumor', False))
            avg_confidence = np.mean([p.get('confidence', 0) for p in predictions])
            
            report['summary'] = {
                'tumor_detected': tumor_count,
                'non_tumor_detected': len(predictions) - tumor_count,
                'tumor_rate': tumor_count / len(predictions),
                'average_confidence': float(avg_confidence),
                'high_risk_cases': sum(1 for p in predictions 
                                     if p.get('risk_level') == 'High Risk')
            }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving report: {str(e)}")
        return False

def load_prediction_report(file_path: str) -> Optional[Dict]:
    """
    Load prediction report from JSON file.
    
    Args:
        file_path: Path to the JSON report file
        
    Returns:
        Report dictionary or None if error
    """
    try:
        with open(file_path, 'r') as f:
            report = json.load(f)
        
        logger.info(f"Report loaded from {file_path}")
        return report
        
    except Exception as e:
        logger.error(f"Error loading report: {str(e)}")
        return None

def validate_prediction_data(prediction_result: Dict) -> bool:
    """
    Validate prediction result data structure.
    
    Args:
        prediction_result: Prediction result dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        'predicted_class', 'confidence', 'is_tumor', 
        'probabilities', 'risk_level'
    ]
    
    try:
        # Check required fields
        for field in required_fields:
            if field not in prediction_result:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Check probabilities structure
        probabilities = prediction_result.get('probabilities', {})
        if not isinstance(probabilities, dict):
            logger.error("Probabilities must be a dictionary")
            return False
        
        if 'non_tumor' not in probabilities or 'tumor' not in probabilities:
            logger.error("Missing probability values")
            return False
        
        # Check confidence range
        confidence = prediction_result.get('confidence', 0)
        if not (0 <= confidence <= 1):
            logger.error("Confidence must be between 0 and 1")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating prediction data: {str(e)}")
        return False