# Data Split Information

To build and evaluate predictive models effectively, the dataset was split into training, validation, and test sets as follows:

| Dataset Partition | Percentage of Total Data | Description                                      |
|-------------------|-------------------------|------------------------------------------------|
| Training Set      | 70%                     | Used to train machine learning models.         |
| Validation Set    | 15%                     | Used for tuning hyperparameters and model selection. |
| Test Set          | 15%                     | Used to assess the final model’s performance on unseen data. |

---

# Splitting Strategy

- The data was randomly split to ensure a representative distribution of both normal and failure instances across all sets.
- Stratified sampling was applied to maintain the proportion of failure cases in each split due to class imbalance.
- Time-based splitting was considered but not applied here due to lack of explicit timestamps in the dataset.
- Data preprocessing steps such as scaling and feature engineering were fit only on the training set and applied to validation and test sets to avoid data leakage.

---

# Notes

- If the dataset contains time series or sequential data, more advanced splitting techniques (e.g., sliding window or rolling forecast origin) might be needed for real-world deployment scenarios.
- Class imbalance is addressed using techniques such as oversampling, undersampling, or class-weighted loss functions during model training.
