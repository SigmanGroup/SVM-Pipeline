from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def perform_grid_search_cv(selected_kernels, param_grids, X_train_scaled_df, y_train, X_test_scaled_df, y_test, random_state, steps, k_fold):
    """
    Perform GridSearchCV with RFECV for SVM classifiers with different kernels.

    Parameters:
    selected_kernels (list): List of kernel types to evaluate.
    param_grids (dict): Dictionary containing parameter grids for each kernel.
    X_train_scaled_df (DataFrame): Scaled training data.
    y_train (Series): Training labels.
    X_test_scaled_df (DataFrame): Scaled test data.
    y_test (Series): Test labels.
    random_state (int): Random state for reproducibility.
    steps (int): Number of steps for RFECV.
    k_fold (int): Number of folds for cross-validation.
    """

    best_models = {}
    
    for kernel in selected_kernels:
        print(f"Running GridSearchCV for {kernel.strip()} kernel...")

        svm = SVC(kernel=kernel.strip(), random_state=random_state, probability=False)

        # RFECV
        rfecv = RFECV(estimator=svm, step=steps, cv=k_fold, scoring='accuracy')
        pipeline = Pipeline([
            ('rfe', rfecv),
            ('svc', svm)
        ])

        grid_search = GridSearchCV(pipeline, param_grids[kernel.strip()], cv=k_fold, scoring='accuracy')
        grid_search.fit(X_train_scaled_df, y_train)
        best_pipeline = grid_search.best_estimator_
        best_rfecv = best_pipeline.named_steps['rfe']
        selected_features_mask = best_rfecv.support_
        selected_features = [feature for feature, selected in zip(X_train_scaled_df.columns, selected_features_mask) if selected]
        
        # Store the best model for the linear kernel
        best_models[kernel.strip()] = {
            'best_pipeline': best_pipeline,
            'best_rfecv': best_rfecv,
            'best_svc_model': best_pipeline.named_steps['svc'],
            'selected_features': selected_features
        }

        y_train_pred = best_pipeline.predict(X_train_scaled_df)
        y_test_pred = best_pipeline.predict(X_test_scaled_df)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        best_index = grid_search.best_index_
        cv_results = grid_search.cv_results_

        # Print results 
        print(f"Best {kernel.capitalize()} SVM Parameters:", best_pipeline.named_steps['svc'].get_params())
        print("Training Set Accuracy:", train_accuracy)
        print("Test Set Accuracy:", test_accuracy)
        print("Cross-Validation Scores:", grid_search.cv_results_['mean_test_score'])
        print("Best Cross-Validation Score:", grid_search.best_score_)
        print("Number of Features Selected:", len(selected_features))
        print("")
        print("Cross-Validation Scores for Best Parameter Combination:")
        for i in range(k_fold):
            print(f"Fold {i+1}: {cv_results[f'split{i}_test_score'][best_index]}")
        print(f"Classification Report for {kernel.capitalize()} Kernel Training Set:")
        print(classification_report(y_train, y_train_pred))
        print(f"Classification Report for {kernel.capitalize()} Kernel Test Set:")
        print(classification_report(y_test, y_test_pred))
        print("")

    return best_models