from src.models.multi_horizon_trainer import MultiHorizonTrainer

# Use the existing features file
# Correct path relative to where we run it (project root)
feature_path = "data/features/features_1H_advanced.parquet"

trainer = MultiHorizonTrainer(
    horizons=[1, 3, 12, 24],  # 1h, 3h, 12h, 24h prediction
    regime_aware=True,
    use_rl_filter=True,
    label_type="trend_duration"  # Predict continuation probability
)

trainer.fit(dataset=feature_path)
trainer.save("data/models/multi_horizon_v1/")
