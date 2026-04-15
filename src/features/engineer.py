"""Feature engineering for Network Intrusion Detection System."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler


class NetworkFlowFeatureEngineer:
    """Feature engineer for network flow data."""

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        scaler_type: str = "robust",
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize feature engineer.

        Args:
            categorical_columns: List of categorical column names.
            numerical_columns: List of numerical column names.
            scaler_type: Type of scaler to use ('standard' or 'robust').
            logger: Logger instance.
        """
        self.categorical_columns = categorical_columns or [
            "protocol_type", "service", "flag"
        ]
        self.numerical_columns = numerical_columns or [
            "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
            "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
            "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
            "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
        ]
        self.scaler_type = scaler_type
        self.logger = logger or logging.getLogger(__name__)

        # Initialize encoders and scalers
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = self._get_scaler()
        self.is_fitted = False

    def _get_scaler(self):
        """Get the appropriate scaler based on scaler_type.

        Returns:
            Scaler instance.
        """
        if self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {self.scaler_type}")

    def fit(self, X: pd.DataFrame) -> "NetworkFlowFeatureEngineer":
        """Fit the feature engineer on training data.

        Args:
            X: Training features DataFrame.

        Returns:
            Self for method chaining.
        """
        self.logger.info("Fitting feature engineer")

        # Fit label encoders for categorical columns
        for col in self.categorical_columns:
            if col in X.columns:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(X[col].astype(str))

        # Fit scaler for numerical columns
        numerical_cols = [col for col in self.numerical_columns if col in X.columns]
        if numerical_cols:
            self.scaler.fit(X[numerical_cols])

        self.is_fitted = True
        self.logger.info("Feature engineer fitting completed")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted encoders and scalers.

        Args:
            X: Features DataFrame to transform.

        Returns:
            Transformed features DataFrame.
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transform")

        self.logger.info("Transforming features")
        X_transformed = X.copy()

        # Transform categorical columns
        for col in self.categorical_columns:
            if col in X_transformed.columns and col in self.label_encoders:
                # Handle unknown categories
                known_categories = set(self.label_encoders[col].classes_)
                X_transformed[col] = X_transformed[col].astype(str)
                X_transformed[col] = X_transformed[col].apply(
                    lambda x: x if x in known_categories else "unknown"
                )
                X_transformed[col] = self.label_encoders[col].transform(X_transformed[col])

        # Transform numerical columns
        numerical_cols = [col for col in self.numerical_columns if col in X_transformed.columns]
        if numerical_cols:
            X_transformed[numerical_cols] = self.scaler.transform(X_transformed[numerical_cols])

        self.logger.info("Feature transformation completed")
        return X_transformed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform features in one step.

        Args:
            X: Features DataFrame.

        Returns:
            Transformed features DataFrame.
        """
        return self.fit(X).transform(X)

    def engineer_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer statistical features from network flow data.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with additional statistical features.
        """
        self.logger.info("Engineering statistical features")
        df_enhanced = df.copy()

        # Byte-based features
        if "src_bytes" in df.columns and "dst_bytes" in df.columns:
            df_enhanced["total_bytes"] = df["src_bytes"] + df["dst_bytes"]
            df_enhanced["byte_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
            df_enhanced["byte_asymmetry"] = np.abs(df["src_bytes"] - df["dst_bytes"]) / (df["total_bytes"] + 1)

        # Duration-based features
        if "duration" in df.columns:
            df_enhanced["duration_log"] = np.log1p(df["duration"])
            df_enhanced["duration_sqrt"] = np.sqrt(df["duration"])

        # Rate-based features
        if "count" in df.columns and "duration" in df.columns:
            df_enhanced["connection_rate"] = df["count"] / (df["duration"] + 1)

        if "srv_count" in df.columns and "duration" in df.columns:
            df_enhanced["service_rate"] = df["srv_count"] / (df["duration"] + 1)

        # Error rate combinations
        error_cols = ["serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate"]
        available_error_cols = [col for col in error_cols if col in df.columns]
        if available_error_cols:
            df_enhanced["total_error_rate"] = df[available_error_cols].mean(axis=1)
            df_enhanced["max_error_rate"] = df[available_error_cols].max(axis=1)

        # Service diversity features
        if "same_srv_rate" in df.columns and "diff_srv_rate" in df.columns:
            df_enhanced["service_diversity"] = df["diff_srv_rate"] / (df["same_srv_rate"] + 1)

        # Host-based features
        if "dst_host_count" in df.columns and "dst_host_srv_count" in df.columns:
            df_enhanced["host_service_ratio"] = df["dst_host_srv_count"] / (df["dst_host_count"] + 1)

        # Entropy-based features
        if "src_bytes" in df.columns and "dst_bytes" in df.columns:
            df_enhanced["byte_entropy"] = self._calculate_byte_entropy(df["src_bytes"], df["dst_bytes"])

        self.logger.info("Statistical feature engineering completed")
        return df_enhanced

    def _calculate_byte_entropy(self, src_bytes: pd.Series, dst_bytes: pd.Series) -> pd.Series:
        """Calculate entropy based on byte distribution.

        Args:
            src_bytes: Source bytes series.
            dst_bytes: Destination bytes series.

        Returns:
            Entropy values.
        """
        total_bytes = src_bytes + dst_bytes
        src_prob = src_bytes / (total_bytes + 1)
        dst_prob = dst_bytes / (total_bytes + 1)

        # Calculate entropy
        epsilon = 1e-10
        entropy = -(src_prob * np.log2(src_prob + epsilon) + dst_prob * np.log2(dst_prob + epsilon))
        
        return entropy

    def engineer_temporal_features(self, df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
        """Engineer temporal features from timestamp data.

        Args:
            df: Input DataFrame.
            time_col: Name of timestamp column.

        Returns:
            DataFrame with temporal features.
        """
        if time_col not in df.columns:
            self.logger.warning(f"Timestamp column '{time_col}' not found, skipping temporal features")
            return df

        self.logger.info("Engineering temporal features")
        df_temporal = df.copy()

        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df_temporal[time_col] = pd.to_datetime(df[time_col])

        # Extract time components
        df_temporal["hour"] = df_temporal[time_col].dt.hour
        df_temporal["day_of_week"] = df_temporal[time_col].dt.dayofweek
        df_temporal["day_of_month"] = df_temporal[time_col].dt.day
        df_temporal["month"] = df_temporal[time_col].dt.month

        # Cyclical encoding for time features
        df_temporal["hour_sin"] = np.sin(2 * np.pi * df_temporal["hour"] / 24)
        df_temporal["hour_cos"] = np.cos(2 * np.pi * df_temporal["hour"] / 24)
        df_temporal["day_sin"] = np.sin(2 * np.pi * df_temporal["day_of_week"] / 7)
        df_temporal["day_cos"] = np.cos(2 * np.pi * df_temporal["day_of_week"] / 7)

        # Business hours indicator
        df_temporal["is_business_hours"] = (
            (df_temporal["hour"] >= 9) & (df_temporal["hour"] <= 17)
        ).astype(int)

        # Weekend indicator
        df_temporal["is_weekend"] = (df_temporal["day_of_week"] >= 5).astype(int)

        self.logger.info("Temporal feature engineering completed")
        return df_temporal

    def engineer_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer network-specific features.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with network features.
        """
        self.logger.info("Engineering network features")
        df_network = df.copy()

        # Protocol-specific features
        if "protocol_type" in df.columns:
            df_network["is_tcp"] = (df["protocol_type"] == 0).astype(int)
            df_network["is_udp"] = (df["protocol_type"] == 1).astype(int)
            df_network["is_icmp"] = (df["protocol_type"] == 2).astype(int)

        # Service-specific features
        if "service" in df.columns:
            df_network["is_http"] = (df["service"] == "http").astype(int)
            df_network["is_ftp"] = (df["service"] == "ftp").astype(int)
            df_network["is_smtp"] = (df["service"] == "smtp").astype(int)
            df_network["is_ssh"] = (df["service"] == "ssh").astype(int)

        # Flag-based features
        if "flag" in df.columns:
            df_network["is_sf_flag"] = (df["flag"] == "SF").astype(int)
            df_network["is_s0_flag"] = (df["flag"] == "S0").astype(int)
            df_network["is_rej_flag"] = (df["flag"] == "REJ").astype(int)

        # Connection state features
        if "land" in df.columns:
            df_network["is_land_attack"] = df["land"].astype(int)

        # Fragment-based features
        if "wrong_fragment" in df.columns:
            df_network["has_wrong_fragments"] = (df["wrong_fragment"] > 0).astype(int)

        if "urgent" in df.columns:
            df_network["has_urgent_packets"] = (df["urgent"] > 0).astype(int)

        self.logger.info("Network feature engineering completed")
        return df_network

    def engineer_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer behavioral features based on connection patterns.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with behavioral features.
        """
        self.logger.info("Engineering behavioral features")
        df_behavioral = df.copy()

        # Connection intensity features
        if "count" in df.columns and "duration" in df.columns:
            df_behavioral["connection_intensity"] = df["count"] / (df["duration"] + 1)

        # Service usage patterns
        if "srv_count" in df.columns and "count" in df.columns:
            df_behavioral["service_concentration"] = df["srv_count"] / (df["count"] + 1)

        # Error pattern features
        error_cols = ["serror_rate", "rerror_rate"]
        available_error_cols = [col for col in error_cols if col in df.columns]
        if available_error_cols:
            df_behavioral["error_pattern_score"] = df[available_error_cols].mean(axis=1)

        # Host interaction patterns
        if "dst_host_count" in df.columns and "dst_host_srv_count" in df.columns:
            df_behavioral["host_service_efficiency"] = df["dst_host_srv_count"] / (df["dst_host_count"] + 1)

        # Anomaly indicators
        numerical_cols = ["duration", "src_bytes", "dst_bytes", "count", "srv_count"]
        available_numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        if available_numerical_cols:
            # Calculate z-scores for anomaly detection
            for col in available_numerical_cols:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df_behavioral[f"{col}_zscore"] = np.abs((df[col] - mean_val) / std_val)

        self.logger.info("Behavioral feature engineering completed")
        return df_behavioral

    def get_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from trained model.

        Args:
            model: Trained model with feature_importances_ attribute.
            feature_names: List of feature names.

        Returns:
            DataFrame with feature importance scores.
        """
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model does not have feature_importances_ attribute")

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        return importance_df

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "mutual_info",
        k: int = 20,
    ) -> List[str]:
        """Select top-k features using specified method.

        Args:
            X: Features DataFrame.
            y: Target series.
            method: Feature selection method ('mutual_info', 'chi2', 'f_score').
            k: Number of features to select.

        Returns:
            List of selected feature names.
        """
        from sklearn.feature_selection import (
            SelectKBest,
            mutual_info_classif,
            chi2,
            f_classif,
        )

        self.logger.info(f"Selecting top {k} features using {method}")

        # Choose scoring function
        if method == "mutual_info":
            score_func = mutual_info_classif
        elif method == "chi2":
            score_func = chi2
        elif method == "f_score":
            score_func = f_classif
        else:
            raise ValueError(f"Unsupported feature selection method: {method}")

        # Select features
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.logger.info(f"Selected features: {selected_features}")
        return selected_features
