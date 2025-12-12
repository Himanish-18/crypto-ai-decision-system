import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.risk.pca_factor_model import PCAFactorModel

logger = logging.getLogger("residual_model")


class ResidualReturnModel:
    """
    v24 Alpha Intelligence.
    Isolates 'Pure Alpha' by removing Market Beta and Factor Risk from returns.
    residual = raw_return - (beta * market_return + factor_loadings * factors)
    """

    def __init__(self, use_pca_factors: bool = True):
        self.use_pca = use_pca_factors
        if self.use_pca:
            self.pca_model = PCAFactorModel(n_components=3)
        self.betas = {}

    def fit_transform(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series,
        other_factors: pd.DataFrame = None,
    ) -> pd.Series:
        """
        Fit Beta model and return Residuals.
        """
        # Align data
        df = pd.DataFrame({"Asset": asset_returns, "Market": market_returns})
        if other_factors is not None:
            df = pd.concat([df, other_factors], axis=1)

        df = df.dropna()

        if df.empty:
            return asset_returns  # Fallback

        y = df["Asset"]
        X = df.drop(columns=["Asset"])

        # Fit Linear Model
        model = LinearRegression()
        model.fit(X, y)

        # Calculate Residuals
        y_pred = model.predict(X)
        residuals = y - y_pred

        # Store Betas
        self.betas = dict(zip(X.columns, model.coef_))
        self.betas["intercept"] = model.intercept_

        logger.info(
            f"Residual Model Fit. Market Beta: {self.betas.get('Market', 0):.4f}"
        )

        return residuals

    def extract_pca_residuals(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove common PCA factors from ALL assets in the dataframe.
        Returns dataframe of idiosyncratic returns.
        """
        if not self.use_pca:
            logger.warning("PCA not enabled for this instance.")
            return returns_df

        self.pca_model.fit(returns_df)

        # Reconstruct systematic component
        # X_rec = Score * Loading + Mean
        X_sys = self.pca_model.pca.inverse_transform(
            self.pca_model.pca.transform(returns_df)
        )
        X_sys = pd.DataFrame(X_sys, index=returns_df.index, columns=returns_df.columns)

        # Residual = Original - Systematic
        residuals = returns_df - X_sys

        explained = np.sum(self.pca_model.pca.explained_variance_ratio_)
        logger.info(
            f"PCA Removed {explained*100:.2f}% Systematic Variance. Remaining is Pure Alpha."
        )

        return residuals
