import pandas as pd
import lightgbm as lgb
import logging


def select_canary_features(train_df, target, params, n_canaries=5):
    logging.info("Starting feature selection with canaries.")
    
    df = train_df.copy()
    
    # 1) inject canaries
    logging.info(f"Injecting {n_canaries} canary features.")
    for i in range(n_canaries):
        df[f'canary_{i}'] = np.random.randn(len(df))
    X = df.drop(columns=[target])
    y = df[target]
    
    # 2) lightGBM train
    logging.info("Training LightGBM model.")
    dtrain = lgb.Dataset(X, y)
    model  = lgb.train(params, dtrain, num_boost_round=200)
    
    # 3) importances
    logging.info("Calculating feature importances.")
    
    imps   = pd.Series(model.feature_importance("gain"), index=X.columns)
    thresh = imps.filter(like="canary_").max()
    logging.info(f"Canary importance threshold: {thresh}")
    
    # 4) keep only real features above that threshold
    real_feats = [f for f in X.columns if not f.startswith("canary_")]
    survivors  = [f for f in real_feats if imps[f] > thresh]
    logging.info(f"Selected {len(survivors)} features out of {len(real_feats)} real features.")
    
    return survivors