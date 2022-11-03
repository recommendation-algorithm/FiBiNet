import tensorflow as tf
from fibinet.model.components.inputs import SparseFeat, DenseFeat, VarLenSparseFeat


def get_top_inputs_embeddings(feature_columns, features, embeddings, feature_importance_metric="dimension",
                              feature_importance_top_k=-1, return_feature_index=False):
    # Default use all inputs and embeddings
    if feature_importance_top_k == -1:
        feature_importance_top_k = len(feature_columns)

    # Get the TopK most important feature names
    sorted_feature_columns = []
    print("Feature columns: ")
    for fc in feature_columns:
        print(fc)
        sorted_feature_columns.append((fc.name, getattr(fc, feature_importance_metric)))
    sorted_feature_columns.sort(key=lambda f: f[1], reverse=True)
    print("sorted_feature_columns: ", sorted_feature_columns)
    selected_feature_columns = set([sorted_feature_columns[i][0] for i in range(feature_importance_top_k)])
    print("selected_feature_columns: ", selected_feature_columns)

    # Get the TopK most important feature inputs in order of [sparse inputs, var len inputs, dense inputs]
    top_inputs = []
    for idx, fc in enumerate(feature_columns):
        if isinstance(fc, SparseFeat) and fc.name in selected_feature_columns:
            top_inputs.append(features[fc.name])
    for idx, fc in enumerate(feature_columns):
        if isinstance(fc, VarLenSparseFeat) and fc.name in selected_feature_columns:
            top_inputs.append(features[fc.name])
    for idx, fc in enumerate(feature_columns):
        if isinstance(fc, DenseFeat) and fc.name in selected_feature_columns:
            top_inputs.append(features[fc.name])

    count_sparse_features = 0
    offsets_sparse_features = []
    count_var_len_features = 0
    offsets_var_len_features = []
    count_dense_features = 0
    offsets_dense_features = []
    for idx, fc in enumerate(feature_columns):
        if isinstance(fc, SparseFeat):
            if fc.name in selected_feature_columns:
                offsets_sparse_features.append(count_sparse_features)
            count_sparse_features += 1
        elif isinstance(fc, VarLenSparseFeat):
            if fc.name in selected_feature_columns:
                offsets_var_len_features.append(count_var_len_features)
            count_var_len_features += 1
        else:
            if fc.name in selected_feature_columns:
                offsets_dense_features.append(count_dense_features)
            count_dense_features += 1
    # embeddings = [sparse_embeddings, var_len_embeddings, dense_embeddings]
    selected_features_indexes = [idx for idx in offsets_sparse_features]
    base_var_len_features = count_sparse_features
    for offset in offsets_var_len_features:
        selected_features_indexes.append(base_var_len_features + offset)
    base_dense_features = count_sparse_features + count_var_len_features
    for offset in offsets_dense_features:
        selected_features_indexes.append(base_dense_features + offset)
    top_embeddings = tf.gather(embeddings, selected_features_indexes, axis=1)
    if return_feature_index:
        return top_inputs, top_embeddings, selected_features_indexes
    return top_inputs, top_embeddings
