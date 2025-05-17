"""
Advanced Pattern Recognition Module

This module provides stock pattern recognition capabilities using statistical methods
and technical analysis techniques to identify common patterns in price charts.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from scipy import signal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatternRecognizer:
    """
    Class for recognizing common chart patterns in stock price data
    
    Supported patterns:
    - Head and shoulders / Inverse head and shoulders
    - Double top / Double bottom
    - Triple top / Triple bottom
    - Rectangle (trading range)
    - Triangle (ascending, descending, symmetrical)
    - Cup and handle
    - Flag / Pennant
    - Wedge (rising, falling)
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the pattern recognizer with configuration parameters
        
        Args:
            config: Dictionary with configuration parameters
                - smoothing_period: Window size for smoothing price data (default: 5)
                - peak_distance: Minimum distance between peaks/troughs (default: 15)
                - threshold_pct: Percentage threshold for pattern recognition (default: 2.0)
                - pattern_window: Window size for pattern detection (default: 120)
                - confidence_level: Confidence level for statistical tests (default: 0.95)
        """
        self.config = config or {}
        self.smoothing_period = self.config.get('smoothing_period', 5)
        self.peak_distance = self.config.get('peak_distance', 15)
        self.threshold_pct = self.config.get('threshold_pct', 2.0)
        self.pattern_window = self.config.get('pattern_window', 120)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        
    def _smooth_data(self, prices: np.ndarray) -> np.ndarray:
        """
        Apply smoothing to price data to reduce noise
        
        Args:
            prices: Array of price data
            
        Returns:
            Smoothed price data
        """
        return pd.Series(prices).rolling(window=self.smoothing_period, center=True, min_periods=1).mean().values
        
    def _find_peaks_and_troughs(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find peaks and troughs in price data
        
        Args:
            prices: Array of price data
            
        Returns:
            Tuple of (peak_indices, trough_indices)
        """
        # Smooth data to reduce noise
        smoothed = self._smooth_data(prices)
        
        # Find peaks (local maxima)
        peaks, _ = signal.find_peaks(smoothed, distance=self.peak_distance)
        
        # Find troughs (local minima)
        troughs, _ = signal.find_peaks(-smoothed, distance=self.peak_distance)
        
        return peaks, troughs
        
    def _calculate_pattern_metrics(self, prices: np.ndarray, indices: List[int]) -> Dict:
        """
        Calculate metrics for pattern validation
        
        Args:
            prices: Array of price data
            indices: List of indices defining the pattern
            
        Returns:
            Dictionary with pattern metrics
        """
        if len(indices) < 2:
            return {}
            
        # Extract prices at the specified indices
        pattern_prices = prices[indices]
        
        # Calculate price differences
        price_diffs = np.diff(pattern_prices)
        
        # Calculate percentage changes
        pct_changes = price_diffs / pattern_prices[:-1] * 100
        
        # Calculate price amplitude
        amplitude = (max(pattern_prices) - min(pattern_prices)) / min(pattern_prices) * 100
        
        # Calculate the trend before the pattern
        pre_pattern = indices[0] - 20 if indices[0] >= 20 else 0
        pre_trend = (prices[indices[0]] - prices[pre_pattern]) / prices[pre_pattern] * 100 if pre_pattern > 0 else 0
        
        return {
            "price_diffs": price_diffs,
            "pct_changes": pct_changes,
            "amplitude": amplitude,
            "pre_trend": pre_trend
        }
        
    def detect_head_and_shoulders(self, data: pd.DataFrame, column: str = 'Close') -> Dict:
        """
        Detect head and shoulders pattern and inverse head and shoulders pattern
        
        Args:
            data: DataFrame with stock data
            column: Column name to analyze (default: 'Close')
            
        Returns:
            Dictionary with pattern detection results
        """
        if column not in data.columns:
            logger.error(f"Column '{column}' not found in data")
            return {"pattern": "none", "confidence": 0}
            
        # Get price array
        prices = data[column].values
        
        # Find peaks and troughs
        peaks, troughs = self._find_peaks_and_troughs(prices)
        
        # Window for detection (analyze last N periods)
        window = min(len(prices), self.pattern_window)
        prices_window = prices[-window:]
        
        # Adjust peak and trough indices for the window
        peaks = peaks[peaks >= len(prices) - window]
        peaks = peaks - (len(prices) - window)
        troughs = troughs[troughs >= len(prices) - window]
        troughs = troughs - (len(prices) - window)
        
        # Check for regular head and shoulders pattern (3 peaks with middle higher)
        h_and_s_confidence = 0
        inv_h_and_s_confidence = 0
        
        if len(peaks) >= 3:
            # Check peaks in groups of 3
            for i in range(len(peaks) - 2):
                # Get three consecutive peaks
                p1, p2, p3 = peaks[i], peaks[i+1], peaks[i+2]
                
                # Check if middle peak is highest (head and shoulders)
                if prices_window[p2] > prices_window[p1] and prices_window[p2] > prices_window[p3]:
                    # Check if shoulders are roughly at the same height
                    shoulder_diff = abs(prices_window[p1] - prices_window[p3]) / prices_window[p1] * 100
                    
                    # Check if there's a trough between each peak
                    troughs_between = [t for t in troughs if p1 < t < p3]
                    
                    if shoulder_diff < self.threshold_pct * 2 and len(troughs_between) >= 2:
                        # Calculate confidence based on clarity of pattern
                        head_height = prices_window[p2] / ((prices_window[p1] + prices_window[p3]) / 2) - 1
                        confidence = min(0.95, max(0.5, head_height * 5 * (1 - shoulder_diff/10)))
                        
                        if confidence > h_and_s_confidence:
                            h_and_s_confidence = confidence
                            h_and_s_details = {
                                "pattern": "head_and_shoulders",
                                "confidence": confidence,
                                "left_shoulder_idx": p1 + (len(prices) - window),
                                "head_idx": p2 + (len(prices) - window),
                                "right_shoulder_idx": p3 + (len(prices) - window),
                                "neckline": min(prices_window[troughs_between[0]], prices_window[troughs_between[-1]])
                            }
        
        # Check for inverse head and shoulders pattern (3 troughs with middle lower)
        if len(troughs) >= 3:
            # Check troughs in groups of 3
            for i in range(len(troughs) - 2):
                # Get three consecutive troughs
                t1, t2, t3 = troughs[i], troughs[i+1], troughs[i+2]
                
                # Check if middle trough is lowest (inverse head and shoulders)
                if prices_window[t2] < prices_window[t1] and prices_window[t2] < prices_window[t3]:
                    # Check if shoulders are roughly at the same height
                    shoulder_diff = abs(prices_window[t1] - prices_window[t3]) / prices_window[t1] * 100
                    
                    # Check if there's a peak between each trough
                    peaks_between = [p for p in peaks if t1 < p < t3]
                    
                    if shoulder_diff < self.threshold_pct * 2 and len(peaks_between) >= 2:
                        # Calculate confidence based on clarity of pattern
                        head_depth = 1 - prices_window[t2] / ((prices_window[t1] + prices_window[t3]) / 2)
                        confidence = min(0.95, max(0.5, head_depth * 5 * (1 - shoulder_diff/10)))
                        
                        if confidence > inv_h_and_s_confidence:
                            inv_h_and_s_confidence = confidence
                            inv_h_and_s_details = {
                                "pattern": "inverse_head_and_shoulders",
                                "confidence": confidence,
                                "left_shoulder_idx": t1 + (len(prices) - window),
                                "head_idx": t2 + (len(prices) - window),
                                "right_shoulder_idx": t3 + (len(prices) - window),
                                "neckline": max(prices_window[peaks_between[0]], prices_window[peaks_between[-1]])
                            }
        
        # Return the pattern with highest confidence
        if h_and_s_confidence > inv_h_and_s_confidence and h_and_s_confidence > 0.5:
            return h_and_s_details
        elif inv_h_and_s_confidence > 0.5:
            return inv_h_and_s_details
        else:
            return {"pattern": "none", "confidence": 0}
            
    def detect_double_pattern(self, data: pd.DataFrame, column: str = 'Close') -> Dict:
        """
        Detect double top and double bottom patterns
        
        Args:
            data: DataFrame with stock data
            column: Column name to analyze (default: 'Close')
            
        Returns:
            Dictionary with pattern detection results
        """
        if column not in data.columns:
            logger.error(f"Column '{column}' not found in data")
            return {"pattern": "none", "confidence": 0}
            
        # Get price array
        prices = data[column].values
        
        # Find peaks and troughs
        peaks, troughs = self._find_peaks_and_troughs(prices)
        
        # Window for detection (analyze last N periods)
        window = min(len(prices), self.pattern_window)
        prices_window = prices[-window:]
        
        # Adjust peak and trough indices for the window
        peaks = peaks[peaks >= len(prices) - window]
        peaks = peaks - (len(prices) - window)
        troughs = troughs[troughs >= len(prices) - window]
        troughs = troughs - (len(prices) - window)
        
        # Check for double top (two peaks at similar levels)
        double_top_confidence = 0
        double_bottom_confidence = 0
        
        if len(peaks) >= 2:
            # Check consecutive pairs of peaks
            for i in range(len(peaks) - 1):
                # Get two consecutive peaks
                p1, p2 = peaks[i], peaks[i+1]
                
                # Check if peaks are at similar levels
                peak_diff = abs(prices_window[p1] - prices_window[p2]) / prices_window[p1] * 100
                
                # Check time between peaks (should be significant but not too long)
                time_between = p2 - p1
                
                # Check if there's a significant trough between the peaks
                troughs_between = [t for t in troughs if p1 < t < p2]
                
                if (peak_diff < self.threshold_pct and 
                    time_between > 10 and time_between < window/2 and 
                    len(troughs_between) > 0):
                    
                    # Calculate depth of the trough between peaks
                    trough_depth = 1 - min(prices_window[troughs_between]) / ((prices_window[p1] + prices_window[p2]) / 2)
                    
                    # Calculate confidence based on clarity of pattern
                    confidence = min(0.95, max(0.5, (1 - peak_diff/self.threshold_pct) * trough_depth * 3))
                    
                    if confidence > double_top_confidence:
                        double_top_confidence = confidence
                        double_top_details = {
                            "pattern": "double_top",
                            "confidence": confidence,
                            "first_peak_idx": p1 + (len(prices) - window),
                            "second_peak_idx": p2 + (len(prices) - window),
                            "trough_idx": troughs_between[0] + (len(prices) - window),
                            "resistance_level": (prices_window[p1] + prices_window[p2]) / 2
                        }
        
        # Check for double bottom (two troughs at similar levels)
        if len(troughs) >= 2:
            # Check consecutive pairs of troughs
            for i in range(len(troughs) - 1):
                # Get two consecutive troughs
                t1, t2 = troughs[i], troughs[i+1]
                
                # Check if troughs are at similar levels
                trough_diff = abs(prices_window[t1] - prices_window[t2]) / prices_window[t1] * 100
                
                # Check time between troughs (should be significant but not too long)
                time_between = t2 - t1
                
                # Check if there's a significant peak between the troughs
                peaks_between = [p for p in peaks if t1 < p < t2]
                
                if (trough_diff < self.threshold_pct and 
                    time_between > 10 and time_between < window/2 and 
                    len(peaks_between) > 0):
                    
                    # Calculate height of the peak between troughs
                    peak_height = max(prices_window[peaks_between]) / ((prices_window[t1] + prices_window[t2]) / 2) - 1
                    
                    # Calculate confidence based on clarity of pattern
                    confidence = min(0.95, max(0.5, (1 - trough_diff/self.threshold_pct) * peak_height * 3))
                    
                    if confidence > double_bottom_confidence:
                        double_bottom_confidence = confidence
                        double_bottom_details = {
                            "pattern": "double_bottom",
                            "confidence": confidence,
                            "first_trough_idx": t1 + (len(prices) - window),
                            "second_trough_idx": t2 + (len(prices) - window),
                            "peak_idx": peaks_between[0] + (len(prices) - window),
                            "support_level": (prices_window[t1] + prices_window[t2]) / 2
                        }
        
        # Return the pattern with highest confidence
        if double_top_confidence > double_bottom_confidence and double_top_confidence > 0.5:
            return double_top_details
        elif double_bottom_confidence > 0.5:
            return double_bottom_details
        else:
            return {"pattern": "none", "confidence": 0}
            
    def detect_triangle_pattern(self, data: pd.DataFrame, column: str = 'Close') -> Dict:
        """
        Detect triangle patterns (ascending, descending, symmetrical)
        
        Args:
            data: DataFrame with stock data
            column: Column name to analyze (default: 'Close')
            
        Returns:
            Dictionary with pattern detection results
        """
        if column not in data.columns:
            logger.error(f"Column '{column}' not found in data")
            return {"pattern": "none", "confidence": 0}
            
        # Get price array
        prices = data[column].values
        
        # Window for detection (analyze last N periods)
        window = min(len(prices), self.pattern_window)
        prices_window = prices[-window:]
        
        # Find peaks and troughs
        peaks, troughs = self._find_peaks_and_troughs(prices_window)
        
        # Need at least 2 peaks and 2 troughs to form a triangle
        if len(peaks) < 2 or len(troughs) < 2:
            return {"pattern": "none", "confidence": 0}
        
        # Get the most recent peaks and troughs (last 60% of window)
        recent_cutoff = int(window * 0.4)
        recent_peaks = peaks[peaks >= recent_cutoff]
        recent_troughs = troughs[troughs >= recent_cutoff]
        
        if len(recent_peaks) < 2 or len(recent_troughs) < 2:
            return {"pattern": "none", "confidence": 0}
        
        # Calculate linear regressions for highs and lows
        peak_x = recent_peaks
        peak_y = prices_window[recent_peaks]
        trough_x = recent_troughs
        trough_y = prices_window[recent_troughs]
        
        # Linear regression for peaks
        peak_slope, peak_intercept, peak_r, peak_p, peak_stderr = stats.linregress(peak_x, peak_y)
        
        # Linear regression for troughs
        trough_slope, trough_intercept, trough_r, trough_p, trough_stderr = stats.linregress(trough_x, trough_y)
        
        # Determine triangle type based on slopes
        peak_angle = np.degrees(np.arctan(peak_slope))
        trough_angle = np.degrees(np.arctan(trough_slope))
        
        # Check for convergence (lines getting closer)
        is_converging = (peak_slope < 0 and trough_slope > 0) or (abs(peak_slope) > abs(trough_slope))
        
        # Calculate the confidence based on regression fit and convergence
        convergence_factor = 0
        if is_converging:
            # Calculate point of convergence (intersection of trend lines)
            # x = (trough_intercept - peak_intercept) / (peak_slope - trough_slope)
            # More positive score when convergence is ahead in reasonable future
            if peak_slope != trough_slope:
                x_convergence = (trough_intercept - peak_intercept) / (peak_slope - trough_slope)
                if window < x_convergence < window * 2:
                    convergence_factor = 0.3
        
        # Calculate triangle type confidence
        # Ascending triangle: flat top (horizontal resistance), rising bottom (ascending support)
        ascending_confidence = 0
        if abs(peak_slope) < 0.0005 and trough_slope > 0.0005:
            ascending_confidence = min(0.95, 0.5 + abs(trough_slope) * 100 + abs(trough_r) * 0.2 + convergence_factor)
        
        # Descending triangle: flat bottom (horizontal support), falling top (descending resistance)
        descending_confidence = 0
        if abs(trough_slope) < 0.0005 and peak_slope < -0.0005:
            descending_confidence = min(0.95, 0.5 + abs(peak_slope) * 100 + abs(peak_r) * 0.2 + convergence_factor)
        
        # Symmetrical triangle: converging trend lines with similar slopes
        symmetrical_confidence = 0
        if peak_slope < -0.0005 and trough_slope > 0.0005 and abs(abs(peak_slope) - abs(trough_slope)) < 0.001:
            symmetrical_confidence = min(0.95, 0.5 + (abs(peak_r) + abs(trough_r)) * 0.25 + convergence_factor)
        
        # Return the triangle pattern with highest confidence
        max_confidence = max(ascending_confidence, descending_confidence, symmetrical_confidence)
        
        if max_confidence >= 0.6:
            if max_confidence == ascending_confidence:
                return {
                    "pattern": "ascending_triangle",
                    "confidence": ascending_confidence,
                    "resistance_slope": peak_slope,
                    "support_slope": trough_slope,
                    "resistance_level": np.mean(peak_y),
                    "recent_peaks": peak_x + (len(prices) - window),
                    "recent_troughs": trough_x + (len(prices) - window)
                }
            elif max_confidence == descending_confidence:
                return {
                    "pattern": "descending_triangle",
                    "confidence": descending_confidence,
                    "resistance_slope": peak_slope,
                    "support_slope": trough_slope,
                    "support_level": np.mean(trough_y),
                    "recent_peaks": peak_x + (len(prices) - window),
                    "recent_troughs": trough_x + (len(prices) - window)
                }
            else:
                return {
                    "pattern": "symmetrical_triangle",
                    "confidence": symmetrical_confidence,
                    "resistance_slope": peak_slope,
                    "support_slope": trough_slope,
                    "recent_peaks": peak_x + (len(prices) - window),
                    "recent_troughs": trough_x + (len(prices) - window)
                }
        else:
            return {"pattern": "none", "confidence": 0}
    
    def detect_cup_and_handle(self, data: pd.DataFrame, column: str = 'Close') -> Dict:
        """
        Detect cup and handle pattern
        
        Args:
            data: DataFrame with stock data
            column: Column name to analyze (default: 'Close')
            
        Returns:
            Dictionary with pattern detection results
        """
        if column not in data.columns:
            logger.error(f"Column '{column}' not found in data")
            return {"pattern": "none", "confidence": 0}
            
        # Get price array
        prices = data[column].values
        
        # Window for detection (analyze last N periods)
        window = min(len(prices), self.pattern_window * 2)  # Cup and handle needs a longer window
        prices_window = prices[-window:]
        
        # Find peaks and troughs
        peaks, troughs = self._find_peaks_and_troughs(prices_window)
        
        # Need at least 2 peaks and 1 trough to form a cup
        if len(peaks) < 2 or len(troughs) < 1:
            return {"pattern": "none", "confidence": 0}
        
        # Check each pair of peaks with a significant trough in between
        cup_confidence = 0
        
        for i in range(len(peaks) - 1):
            p1, p2 = peaks[i], peaks[i+1]
            
            # The cup should cover a significant portion of the window
            cup_width = p2 - p1
            if cup_width < window * 0.3 or cup_width > window * 0.7:
                continue
            
            # Find the lowest trough between these peaks
            troughs_between = [t for t in troughs if p1 < t < p2]
            if not troughs_between:
                continue
                
            trough_idx = troughs_between[np.argmin(prices_window[troughs_between])]
            
            # Check if the trough is roughly in the middle
            trough_position = (trough_idx - p1) / cup_width
            if trough_position < 0.3 or trough_position > 0.7:
                continue
            
            # Check if peaks are at similar heights (cup rim)
            peak_diff = abs(prices_window[p1] - prices_window[p2]) / prices_window[p1] * 100
            if peak_diff > self.threshold_pct * 2:
                continue
            
            # Check for a u-shaped curve (cup)
            # Extract the cup segment
            cup_segment = prices_window[p1:p2+1]
            
            # Calculate depth of cup
            cup_height = (prices_window[p1] + prices_window[p2]) / 2
            cup_depth = cup_height - prices_window[trough_idx]
            cup_depth_pct = cup_depth / cup_height * 100
            
            # For a good cup, depth should be significant but not too deep
            if cup_depth_pct < 5 or cup_depth_pct > 30:
                continue
            
            # Check for handle (small dip after second peak)
            if p2 + 5 >= len(prices_window):
                continue
                
            # Look for a small pullback (handle) after the right peak
            handle_segment = prices_window[p2:min(p2 + int(cup_width * 0.3), len(prices_window))]
            if len(handle_segment) < 5:
                continue
                
            # Handle should be a small dip (not too deep)
            handle_low = np.min(handle_segment)
            handle_depth = prices_window[p2] - handle_low
            handle_depth_pct = handle_depth / prices_window[p2] * 100
            
            if handle_depth_pct < 2 or handle_depth_pct > 15:
                continue
            
            # Calculate confidence based on pattern clarity
            symmetry_score = 1 - abs(trough_position - 0.5) * 2  # 1.0 if perfectly centered
            rim_score = 1 - peak_diff / (self.threshold_pct * 2)  # 1.0 if perfectly flat rim
            depth_score = 1 - abs((cup_depth_pct - 15) / 15)     # 1.0 if ideal depth (15%)
            handle_score = 1 - abs((handle_depth_pct - 8) / 8)    # 1.0 if ideal handle depth (8%)
            
            confidence = min(0.95, (symmetry_score * 0.25 + rim_score * 0.25 + depth_score * 0.25 + handle_score * 0.25) * 0.8 + 0.2)
            
            if confidence > cup_confidence:
                cup_confidence = confidence
                cup_details = {
                    "pattern": "cup_and_handle",
                    "confidence": confidence,
                    "left_peak_idx": p1 + (len(prices) - window),
                    "right_peak_idx": p2 + (len(prices) - window),
                    "cup_low_idx": trough_idx + (len(prices) - window),
                    "resistance_level": (prices_window[p1] + prices_window[p2]) / 2
                }
        
        if cup_confidence > 0.6:
            return cup_details
        else:
            return {"pattern": "none", "confidence": 0}
    
    def detect_flag_pattern(self, data: pd.DataFrame, column: str = 'Close') -> Dict:
        """
        Detect flag and pennant patterns
        
        Args:
            data: DataFrame with stock data
            column: Column name to analyze (default: 'Close')
            
        Returns:
            Dictionary with pattern detection results
        """
        if column not in data.columns:
            logger.error(f"Column '{column}' not found in data")
            return {"pattern": "none", "confidence": 0}
            
        # Get price array
        prices = data[column].values
        
        # Window for detection (analyze last N periods)
        window = min(len(prices), self.pattern_window)
        prices_window = prices[-window:]
        
        # For flags/pennants, we need a clear flagpole (strong prior move)
        # Analyze first half of window for flagpole, second half for flag/pennant
        mid_point = window // 2
        
        # Check if there's a strong prior move (flagpole)
        pole_start, pole_end = 0, mid_point - 1
        pole_change = (prices_window[pole_end] - prices_window[pole_start]) / prices_window[pole_start] * 100
        
        # Flagpole should be a significant move
        is_bullish_pole = pole_change > 10
        is_bearish_pole = pole_change < -10
        
        if not (is_bullish_pole or is_bearish_pole):
            return {"pattern": "none", "confidence": 0}
        
        # Analyze the flag/pennant portion (consolidation)
        flag_segment = prices_window[mid_point:]
        
        # Calculate linear regression for the flag segment
        x = np.arange(len(flag_segment))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, flag_segment)
        
        # Flag pattern characteristics
        angle = np.degrees(np.arctan(slope))
        
        # For bullish flag, we expect slight downtrend; for bearish flag, slight uptrend
        # Pennants converge (narrowing range)
        
        # Get upper and lower bounds of the flag/pennant
        peaks, troughs = self._find_peaks_and_troughs(flag_segment)
        
        if len(peaks) < 2 or len(troughs) < 2:
            return {"pattern": "none", "confidence": 0}
        
        # Calculate trends of highs and lows
        peak_x = peaks
        peak_y = flag_segment[peaks]
        trough_x = troughs
        trough_y = flag_segment[troughs]
        
        if len(peak_x) >= 2 and len(trough_x) >= 2:
            peak_slope, peak_intercept, peak_r, peak_p, peak_stderr = stats.linregress(peak_x, peak_y)
            trough_slope, trough_intercept, trough_r, trough_p, trough_stderr = stats.linregress(trough_x, trough_y)
            
            peak_angle = np.degrees(np.arctan(peak_slope))
            trough_angle = np.degrees(np.arctan(trough_slope))
            
            # Determine pattern based on pole direction and flag/pennant shape
            pattern_confidence = 0
            
            # Bullish flag (upward pole, downward flag)
            if is_bullish_pole and -30 < angle < -5 and abs(peak_angle - trough_angle) < 10:
                confidence = min(0.95, 0.5 + abs(r_value) * 0.3 + (pole_change / 20) * 0.2)
                
                if confidence > pattern_confidence:
                    pattern_confidence = confidence
                    pattern_details = {
                        "pattern": "bullish_flag",
                        "confidence": confidence,
                        "pole_start_idx": pole_start + (len(prices) - window),
                        "pole_end_idx": pole_end + (len(prices) - window),
                        "pole_change_pct": pole_change,
                        "flag_slope": slope,
                        "flag_angle": angle
                    }
            
            # Bearish flag (downward pole, upward flag)
            elif is_bearish_pole and 5 < angle < 30 and abs(peak_angle - trough_angle) < 10:
                confidence = min(0.95, 0.5 + abs(r_value) * 0.3 + (abs(pole_change) / 20) * 0.2)
                
                if confidence > pattern_confidence:
                    pattern_confidence = confidence
                    pattern_details = {
                        "pattern": "bearish_flag",
                        "confidence": confidence,
                        "pole_start_idx": pole_start + (len(prices) - window),
                        "pole_end_idx": pole_end + (len(prices) - window),
                        "pole_change_pct": pole_change,
                        "flag_slope": slope,
                        "flag_angle": angle
                    }
            
            # Bullish pennant (upward pole, converging lines)
            elif is_bullish_pole and peak_slope < 0 and trough_slope > 0:
                convergence = min(abs(peak_slope), abs(trough_slope)) / max(abs(peak_slope), abs(trough_slope))
                confidence = min(0.95, 0.5 + convergence * 0.3 + (pole_change / 20) * 0.2)
                
                if confidence > pattern_confidence:
                    pattern_confidence = confidence
                    pattern_details = {
                        "pattern": "bullish_pennant",
                        "confidence": confidence,
                        "pole_start_idx": pole_start + (len(prices) - window),
                        "pole_end_idx": pole_end + (len(prices) - window),
                        "pole_change_pct": pole_change,
                        "upper_slope": peak_slope,
                        "lower_slope": trough_slope
                    }
            
            # Bearish pennant (downward pole, converging lines)
            elif is_bearish_pole and peak_slope > 0 and trough_slope < 0:
                convergence = min(abs(peak_slope), abs(trough_slope)) / max(abs(peak_slope), abs(trough_slope))
                confidence = min(0.95, 0.5 + convergence * 0.3 + (abs(pole_change) / 20) * 0.2)
                
                if confidence > pattern_confidence:
                    pattern_confidence = confidence
                    pattern_details = {
                        "pattern": "bearish_pennant",
                        "confidence": confidence,
                        "pole_start_idx": pole_start + (len(prices) - window),
                        "pole_end_idx": pole_end + (len(prices) - window),
                        "pole_change_pct": pole_change,
                        "upper_slope": peak_slope,
                        "lower_slope": trough_slope
                    }
            
            if pattern_confidence > 0.6:
                return pattern_details
        
        return {"pattern": "none", "confidence": 0}
    
    def detect_wedge_pattern(self, data: pd.DataFrame, column: str = 'Close') -> Dict:
        """
        Detect rising and falling wedge patterns
        
        Args:
            data: DataFrame with stock data
            column: Column name to analyze (default: 'Close')
            
        Returns:
            Dictionary with pattern detection results
        """
        if column not in data.columns:
            logger.error(f"Column '{column}' not found in data")
            return {"pattern": "none", "confidence": 0}
            
        # Get price array
        prices = data[column].values
        
        # Window for detection (analyze last N periods)
        window = min(len(prices), self.pattern_window)
        prices_window = prices[-window:]
        
        # Find peaks and troughs
        peaks, troughs = self._find_peaks_and_troughs(prices_window)
        
        if len(peaks) < 3 or len(troughs) < 3:
            return {"pattern": "none", "confidence": 0}
        
        # Calculate linear regressions for highs and lows
        peak_x = peaks
        peak_y = prices_window[peaks]
        trough_x = troughs
        trough_y = prices_window[troughs]
        
        peak_slope, peak_intercept, peak_r, peak_p, peak_stderr = stats.linregress(peak_x, peak_y)
        trough_slope, trough_intercept, trough_r, trough_p, trough_stderr = stats.linregress(trough_x, trough_y)
        
        # Convert slopes to angles for easier interpretation
        peak_angle = np.degrees(np.arctan(peak_slope))
        trough_angle = np.degrees(np.arctan(trough_slope))
        
        # Calculate the overall trend
        overall_slope, overall_intercept, overall_r, overall_p, overall_stderr = stats.linregress(
            np.arange(len(prices_window)), prices_window)
        overall_angle = np.degrees(np.arctan(overall_slope))
        
        # Check for wedge patterns
        wedge_confidence = 0
        
        # Rising wedge: both lines slope upward, but upper line has lower slope
        # (lines converge upward, bearish pattern)
        if peak_slope > 0 and trough_slope > 0 and peak_slope < trough_slope:
            # Calculate convergence point
            if peak_slope != trough_slope:
                x_convergence = (trough_intercept - peak_intercept) / (peak_slope - trough_slope)
                future_convergence = x_convergence > len(prices_window)
                
                if future_convergence:
                    confidence = min(0.95, 0.5 + abs(peak_r) * 0.2 + abs(trough_r) * 0.2 + 
                                    (min(trough_slope, peak_slope) / max(trough_slope, peak_slope)) * 0.3)
                    
                    if confidence > wedge_confidence:
                        wedge_confidence = confidence
                        wedge_details = {
                            "pattern": "rising_wedge",
                            "confidence": confidence,
                            "upper_slope": peak_slope,
                            "lower_slope": trough_slope,
                            "upper_angle": peak_angle,
                            "lower_angle": trough_angle,
                            "convergence_point": int(x_convergence)
                        }
        
        # Falling wedge: both lines slope downward, but upper line has higher slope
        # (lines converge downward, bullish pattern)
        if peak_slope < 0 and trough_slope < 0 and peak_slope > trough_slope:
            # Calculate convergence point
            if peak_slope != trough_slope:
                x_convergence = (trough_intercept - peak_intercept) / (peak_slope - trough_slope)
                future_convergence = x_convergence > len(prices_window)
                
                if future_convergence:
                    confidence = min(0.95, 0.5 + abs(peak_r) * 0.2 + abs(trough_r) * 0.2 + 
                                    (min(abs(trough_slope), abs(peak_slope)) / max(abs(trough_slope), abs(peak_slope))) * 0.3)
                    
                    if confidence > wedge_confidence:
                        wedge_confidence = confidence
                        wedge_details = {
                            "pattern": "falling_wedge",
                            "confidence": confidence,
                            "upper_slope": peak_slope,
                            "lower_slope": trough_slope,
                            "upper_angle": peak_angle,
                            "lower_angle": trough_angle,
                            "convergence_point": int(x_convergence)
                        }
        
        if wedge_confidence > 0.6:
            return wedge_details
        else:
            return {"pattern": "none", "confidence": 0}
    
    def detect_all_patterns(self, data: pd.DataFrame, column: str = 'Close') -> Dict:
        """
        Run all pattern detection methods and return the most significant pattern
        
        Args:
            data: DataFrame with stock data
            column: Column name to analyze (default: 'Close')
            
        Returns:
            Dictionary with pattern detection results
        """
        if column not in data.columns:
            logger.error(f"Column '{column}' not found in data")
            return {"pattern": "none", "confidence": 0}
        
        # Run all pattern detection methods
        head_shoulders_result = self.detect_head_and_shoulders(data, column)
        double_pattern_result = self.detect_double_pattern(data, column)
        triangle_result = self.detect_triangle_pattern(data, column)
        cup_handle_result = self.detect_cup_and_handle(data, column)
        flag_result = self.detect_flag_pattern(data, column)
        wedge_result = self.detect_wedge_pattern(data, column)
        
        # Collect results with confidence > 0
        patterns = []
        for result in [head_shoulders_result, double_pattern_result, triangle_result, 
                      cup_handle_result, flag_result, wedge_result]:
            if result.get("confidence", 0) > 0:
                patterns.append(result)
        
        # Sort by confidence (highest first)
        patterns.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        # Return all detected patterns
        return {
            "primary_pattern": patterns[0] if patterns else {"pattern": "none", "confidence": 0},
            "all_patterns": patterns
        }


def detect_patterns(ticker: str, data: pd.DataFrame, config: Dict = None) -> Dict:
    """
    Detect chart patterns in stock data
    
    Args:
        ticker: Stock ticker symbol
        data: DataFrame with stock data
        config: Configuration parameters for pattern detection
        
    Returns:
        Dictionary with pattern detection results
    """
    try:
        recognizer = PatternRecognizer(config)
        results = recognizer.detect_all_patterns(data)
        
        return {
            "ticker": ticker,
            "primary_pattern": results["primary_pattern"],
            "all_patterns": results["all_patterns"]
        }
    except Exception as e:
        logger.error(f"Error detecting patterns: {str(e)}")
        return {
            "ticker": ticker,
            "error": str(e)
        }


def detect_multi_stock_patterns(tickers: List[str], data_dict: Dict[str, pd.DataFrame], 
                               config: Dict = None) -> Dict:
    """
    Detect patterns across multiple stocks
    
    Args:
        tickers: List of stock ticker symbols
        data_dict: Dictionary with ticker symbols as keys and DataFrames as values
        config: Configuration parameters for pattern detection
        
    Returns:
        Dictionary with pattern detection results for all stocks
    """
    results = {}
    
    for ticker in tickers:
        if ticker in data_dict and not data_dict[ticker].empty:
            results[ticker] = detect_patterns(ticker, data_dict[ticker], config)
        else:
            results[ticker] = {"ticker": ticker, "error": "No data available"}
    
    # Count patterns by type
    pattern_counts = {}
    for ticker, result in results.items():
        if "primary_pattern" in result and result["primary_pattern"]["pattern"] != "none":
            pattern = result["primary_pattern"]["pattern"]
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    return {
        "stocks": results,
        "pattern_counts": pattern_counts
    }