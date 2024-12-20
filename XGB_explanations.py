import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from xgboost import XGBClassifier
import matplotlib
from lime.lime_tabular import LimeTabularExplainer

matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


class LIMEAnalyzer:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.explanations_by_class = {0: [], 1: [], 2: []}  # Store explanations for each class
        self.all_explanations = []  # Store all explanations

    def add_explanation(self, feature_dict, predicted_class):
        """Add a new LIME explanation to the collection"""
        self.explanations_by_class[predicted_class].append(feature_dict)
        self.all_explanations.append(feature_dict)

    def compute_mean_importance(self):
        """Compute mean absolute importance for each feature across all samples"""
        feature_importances = {feature: [] for feature in self.feature_names}

        for explanation in self.all_explanations:
            for feature, importance in explanation.items():
                feature_importances[feature].append(abs(importance))

        mean_importances = {
            feature: np.mean(importances) if importances else 0
            for feature, importances in feature_importances.items()
        }

        return mean_importances

    def compute_top_k_frequency(self, k=5):
        """Compute how often each feature appears in top-k most important features"""
        feature_counts = {feature: 0 for feature in self.feature_names}
        total_explanations = len(self.all_explanations)

        for explanation in self.all_explanations:
            # Sort features by absolute importance
            sorted_features = sorted(explanation.items(),
                                     key=lambda x: abs(x[1]),
                                     reverse=True)[:k]

            for feature, _ in sorted_features:
                feature_counts[feature] += 1

        # Convert to frequencies
        feature_frequencies = {
            feature: count / total_explanations
            for feature, count in feature_counts.items()
        }

        return feature_frequencies

    def compute_class_wise_importance(self):
        """Compute mean feature importance for each class"""
        class_wise_importance = {}

        for class_label, explanations in self.explanations_by_class.items():
            if not explanations:
                continue

            feature_importances = {feature: [] for feature in self.feature_names}

            for explanation in explanations:
                for feature, importance in explanation.items():
                    feature_importances[feature].append(importance)

            class_wise_importance[class_label] = {
                feature: np.mean(importances) if importances else 0
                for feature, importances in feature_importances.items()
            }

        return class_wise_importance

    def plot_mean_importance(self):
        """Plot mean absolute importance of features"""
        mean_importances = self.compute_mean_importance()

        plt.figure(figsize=(12, 6))
        features = list(mean_importances.keys())
        importances = list(mean_importances.values())

        sns.barplot(x=importances, y=features)
        plt.title('Mean Absolute Feature Importance Across All Samples')
        plt.xlabel('Mean Absolute Importance')
        plt.tight_layout()
        plt.savefig('mean_feature_importance.png')
        plt.close()

    def plot_top_k_frequency(self, k=5):
        """Plot frequency of features appearing in top-k"""
        frequencies = self.compute_top_k_frequency(k)

        plt.figure(figsize=(12, 6))
        features = list(frequencies.keys())
        freq_values = list(frequencies.values())

        sns.barplot(x=freq_values, y=features)
        plt.title(f'Frequency of Features in Top-{k} Most Important')
        plt.xlabel('Frequency')
        plt.tight_layout()
        plt.savefig('feature_frequency.png')
        plt.close()

    def plot_class_wise_heatmap(self):
        """Plot heatmap of feature importance across classes"""
        class_wise_importance = self.compute_class_wise_importance()

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(class_wise_importance).T

        plt.figure(figsize=(12, 8))
        sns.heatmap(df, cmap='RdBu', center=0, annot=True, fmt='.3f')
        plt.title('Feature Importance by Class')
        plt.xlabel('Features')
        plt.ylabel('Class')
        plt.tight_layout()
        plt.savefig('class_wise_importance_heatmap.png')
        plt.close()

    def cluster_explanations(self, n_clusters=3):
        """Cluster explanations using K-means"""
        # Convert explanations to feature vectors
        feature_vectors = []
        for explanation in self.all_explanations:
            vector = [explanation.get(feature, 0) for feature in self.feature_names]
            feature_vectors.append(vector)

        # Perform K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(feature_vectors)

        # Analyze feature importance by cluster
        cluster_importance = {i: [] for i in range(n_clusters)}
        for vector, cluster in zip(feature_vectors, clusters):
            cluster_importance[cluster].append(vector)

        # Compute mean importance for each cluster
        cluster_means = {}
        for cluster, vectors in cluster_importance.items():
            cluster_means[cluster] = np.mean(vectors, axis=0)

        # Plot cluster patterns
        plt.figure(figsize=(15, 8))
        for cluster, means in cluster_means.items():
            plt.subplot(1, n_clusters, cluster + 1)
            sns.barplot(x=means, y=self.feature_names)
            plt.title(f'Cluster {cluster} Pattern')
            plt.xlabel('Mean Importance')
        plt.tight_layout()
        plt.savefig('cluster_patterns.png')
        plt.close()

        return clusters, cluster_means

class StockTrendPredictor:
    def __init__(self):
        self.trend_nodes = ['Trend_Up', 'Trend_Neutral', 'Trend_Down']
        self.models = {
            'XGBoost': XGBClassifier(
                objective='multi:softprob',  # for multi-class probability
                num_class=3,
                learning_rate=0.1,
                max_depth=6,
                n_estimators=100,
                eval_metric='mlogloss',  # changed from list to string
                use_label_encoder=False,  # important for newer versions
                random_state=42
            )
        }

        self.param_grids = {
            'XGBoost': {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'n_estimators': [100, 200],
                'min_child_weight': [1, 3],
                'gamma': [0, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        }

    def train_and_evaluate(self, X, y):
            """Train and evaluate all models with LIME explanations"""
            results = {}

            # Existing feature names setup
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                feature_names = [
                    'sentiment_score', 'RSI', 'MACD', 'Returns_5d', 'combined_trend_influence',
                    'sentiment_trend_up_interaction', 'sentiment_trend_neutral_interaction',
                    'sentiment_trend_down_interaction', 'day_of_week', 'month',
                    'influence_Trend_Up', 'influence_Trend_Neutral', 'influence_Trend_Down'
                ]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Convert to numpy arrays if needed
            X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

            for model_name, model in self.models.items():
                try:
                    print(f"\nTraining {model_name}...")
                    lime_explanations = []

                    # Existing grid search code...
                    if model_name == 'XGBoost':
                        grid_search = GridSearchCV(
                            estimator=model,
                            param_grid=self.param_grids[model_name],
                            cv=5,
                            scoring='accuracy',
                            n_jobs=-1,
                            verbose=1
                        )
                        y_train = y_train.astype(int)
                        y_test = y_test.astype(int)
                    else:
                        grid_search = GridSearchCV(
                            estimator=model,
                            param_grid=self.param_grids[model_name],
                            cv=5,
                            scoring='accuracy',
                            n_jobs=-1
                        )

                    # Fit the model
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_

                    # Make predictions
                    y_pred = best_model.predict(X_test)

                    # Initialize LIME analyzer
                    lime_analyzer = LIMEAnalyzer(feature_names=X_test.columns)

                    # LIME Explanations
                    try:
                        class_names = [0, 1, 2]  # For three classes: Downward, Neutral, Upward

                        # Initialize LIME explainer
                        lime_explainer = LimeTabularExplainer(
                            X_test.values,
                            class_names=class_names,
                            feature_names=X_test.columns,
                            discretize_continuous=True
                        )

                        # Analyze multiple instances
                        for idx in range(min(100, len(X_test))):
                            try:
                                print(f"\nAnalyzing instance {idx}...")

                                # Get current instance
                                instance = X_test.values[idx]

                                # Get prediction
                                predicted_class = best_model.predict(instance.reshape(1, -1))[0]
                                print(f"Predicted class: {predicted_class}")

                                # Get LIME explanation
                                print("Generating LIME explanation...")
                                explanation = lime_explainer.explain_instance(
                                    instance,
                                    best_model.predict_proba,
                                    num_features=len(X_test.columns),
                                    labels=[predicted_class]  # Only explain the predicted class
                                )
                                print("Generated LIME explanation")

                                # Get feature weights
                                print("Processing feature weights...")
                                feature_weights = explanation.as_list(label=predicted_class)  # Specify the label
                                feature_dict = {}

                                # Convert to dictionary and update feature mappings
                                print("Mapping features...")
                                for k, v in dict(feature_weights).items():
                                    for feature in X_test.columns:
                                        if k.find(feature) != -1:
                                            feature_dict[feature] = v
                                            break

                                print(f"Feature dictionary: {feature_dict}")

                                # Add explanation to analyzer
                                lime_analyzer.add_explanation(feature_dict, predicted_class)
                                print("Added explanation to analyzer")

                                # Save individual LIME explanation plot for first few instances
                                if idx < 5:
                                    print(f"Generating plot for instance {idx}...")
                                    predicted_proba = best_model.predict_proba(instance.reshape(1, -1))[0][
                                        predicted_class]
                                    fig = explanation.as_pyplot_figure(label=predicted_class)
                                    plt.title(
                                        f'Explanation - {model_name} - Instance {idx}\nPredicted Class: {predicted_class} (Prob: {predicted_proba:.2f})',
                                        fontsize=8
                                    )
                                    plt.tight_layout()
                                    plt.savefig(f'lime_explanation_{model_name}_instance_{idx}.png')
                                    plt.close()
                                    print(f"Saved plot for instance {idx}")

                            except Exception as e:
                                print(f"Error analyzing instance {idx}:")
                                print(f"Error details: {str(e)}")
                                import traceback
                                traceback.print_exc()
                                continue

                        print("\nCompleted individual instance analysis")

                        # Generate global analysis plots if we have collected any explanations
                        if lime_analyzer.all_explanations:
                            print("\nGenerating global LIME analysis plots...")

                            try:
                                # Plot mean importance
                                lime_analyzer.plot_mean_importance()
                                print("Generated mean importance plot")

                                # Plot top-k frequency
                                lime_analyzer.plot_top_k_frequency(k=5)
                                print("Generated top-k frequency plot")

                                # Plot class-wise heatmap
                                lime_analyzer.plot_class_wise_heatmap()
                                print("Generated class-wise heatmap")

                                # Perform clustering analysis only if we have enough samples
                                n_explanations = len(lime_analyzer.all_explanations)
                                n_clusters = min(3,
                                                 n_explanations)  # Adjust number of clusters based on available samples

                                if n_explanations >= 2:  # Only do clustering if we have at least 2 samples
                                    clusters, cluster_means = lime_analyzer.cluster_explanations(n_clusters=n_clusters)
                                    print(f"Generated cluster patterns plot with {n_clusters} clusters")

                                # Print insights
                                print("\nGlobal Feature Importance Analysis:")
                                mean_importance = lime_analyzer.compute_mean_importance()
                                top_features = sorted(mean_importance.items(), key=lambda x: abs(x[1]), reverse=True)

                                print("\nTop 5 Globally Important Features:")
                                for feature, importance in top_features[:5]:
                                    print(f"{feature}: {importance:.4f}")

                                # Save global analysis results
                                with open('lime_global_analysis.txt', 'w') as f:
                                    f.write("Global Feature Importance Analysis\n")
                                    f.write("--------------------------------\n")
                                    f.write("\nTop 5 Globally Important Features:\n")
                                    for feature, importance in top_features[:5]:
                                        f.write(f"{feature}: {importance:.4f}\n")

                                    f.write("\nClass-wise Important Features:\n")
                                    class_importance = lime_analyzer.compute_class_wise_importance()
                                    for class_label, importances in class_importance.items():
                                        f.write(f"\nClass {class_label}:\n")
                                        top_class_features = sorted(importances.items(), key=lambda x: abs(x[1]),
                                                                    reverse=True)[:5]
                                        for feature, importance in top_class_features:
                                            f.write(f"{feature}: {importance:.4f}\n")

                            except Exception as e:
                                print("Error generating global analysis:")
                                print(f"Error details: {str(e)}")
                                traceback.print_exc()
                        else:
                            print("No explanations were collected, skipping global analysis")

                    except Exception as e:
                        print(f"Error in LIME analysis setup for {model_name}:")
                        print(f"Error details: {str(e)}")
                        traceback.print_exc()

                    # Store all results
                    results[model_name] = {
                        'model': best_model,
                        'best_params': grid_search.best_params_,
                        'best_score': grid_search.best_score_,
                        'test_predictions': y_pred,
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test
                    }

                    # Continue with existing evaluation code...
                    print("\nClassification Report:")
                    print(classification_report(y_test, y_pred,
                                                target_names=['Downward', 'Neutral', 'Upward']))

                    accuracy = np.mean(y_pred == y_test)
                    print(f"\nTest Set Accuracy: {accuracy:.4f}")

                    # Generate existing plots...
                    self.plot_confusion_matrix(y_test, y_pred, model_name)
                    self.plot_feature_importance(best_model, X_train, model_name)
                    self.plot_performance_metrics(y_test, y_pred, model_name)

                except Exception as e:
                    print(f"Error training {model_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

            return results

    def load_data(self, news_stock_file, features_file):
            """Load and prepare the data"""
            try:
                news_stock_data = pd.read_csv(news_stock_file)
                features_data = pd.read_csv(features_file)

                threshold = 0.001
                price_change_pct = (news_stock_data['Close'] - news_stock_data['Open']) / news_stock_data['Open']
                target = np.where(price_change_pct > threshold, 2,
                                  np.where(price_change_pct < -threshold, 0, 1))

                return features_data, target
            except Exception as e:
                print(f"Error loading data: {str(e)}")
                return None, None

    def save_plot(self, fig, filename):
            """Safely save plot to file"""
            try:
                fig.savefig(filename)
                plt.close(fig)
            except Exception as e:
                print(f"Error saving plot {filename}: {str(e)}")

    def plot_confusion_matrix(self, y_true, y_pred, model_name):
            """Plot and save confusion matrix"""
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                cm = confusion_matrix(y_true, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Downward', 'Neutral', 'Upward'],
                            yticklabels=['Downward', 'Neutral', 'Upward'], ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title(f'Confusion Matrix - {model_name}')
                plt.tight_layout()
                self.save_plot(fig, f'confusion_matrix_{model_name}1.png')
            except Exception as e:
                print(f"Error plotting confusion matrix for {model_name}: {str(e)}")

    def plot_feature_importance(self, model, X_train, model_name):
            """Plot feature importance if available"""
            try:
                if hasattr(model, 'feature_importances_'):
                    fig, ax = plt.subplots(figsize=(12, 6))
                    importance_df = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': model.feature_importances_
                    })
                    importance_df = importance_df.sort_values('importance', ascending=False)
                    top_15_features = importance_df.head(15)

                    sns.barplot(x='importance', y='feature', data=top_15_features, ax=ax)
                    ax.set_title(f'Top 15 Feature Importance - {model_name}')
                    ax.set_xlabel('Importance Score')
                    ax.set_ylabel('Features')
                    plt.tight_layout()
                    self.save_plot(fig, f'feature_importance_{model_name}1.png')
            except Exception as e:
                print(f"Error plotting feature importance for {model_name}: {str(e)}")

    def plot_performance_metrics(self, y_test, y_pred, model_name):
            """Plot performance metrics heatmap"""
            try:
                performance_metrics = classification_report(y_test, y_pred,
                                                            target_names=['Downward', 'Neutral', 'Upward'],
                                                            output_dict=True)
                metrics_df = pd.DataFrame(performance_metrics).transpose()

                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(metrics_df.iloc[:-3][['precision', 'recall', 'f1-score']],
                            annot=True, cmap='YlOrRd', ax=ax)
                ax.set_title(f'Model Performance Metrics - {model_name}')
                plt.tight_layout()
                self.save_plot(fig, f'performance_metrics_{model_name}1.png')
            except Exception as e:
                print(f"Error plotting performance metrics for {model_name}: {str(e)}")

def main():
    try:
        # Initialize predictor
        predictor = StockTrendPredictor()

        # Load data
        X, y = predictor.load_data('News+Stock data1.csv', 'final_features1.csv')

        if X is None or y is None:
            print("Failed to load data. Exiting...")
            return

        # Train and evaluate all models
        results = predictor.train_and_evaluate(X, y)

        # Compare models
        comparison_df = pd.DataFrame({
            model_name: {
                'Best CV Score': results[model_name]['best_score'],
                'Best Parameters': str(results[model_name]['best_params'])
            }
            for model_name in results.keys()
        }).transpose()

        print("\nModel Comparison:")
        print(comparison_df)

        # Save comparison results
        comparison_df.to_csv('model_comparison1.csv')

    except Exception as e:
        print(f"An error occurred in main: {str(e)}")

if __name__ == "__main__":
        main()