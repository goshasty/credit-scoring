from hyperopt import hp

space_params = {
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),
    "n_estimators": hp.choice("n_estimators", range(50, 200)),
    "max_depth": hp.choice("max_depth", range(3, 12)),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
    "subsample": hp.uniform("subsample", 0.5, 1.0),
    "reg_alpha": hp.uniform("reg_alpha", 0, 1),
}
