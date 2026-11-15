"""Utilties to handle estimator list"""

from sklearn.utils.metaestimators import _BaseComposition


class _BaseXComposition(_BaseComposition):
    """
    parameter handler for list of estimators
    """

    def _set_params(self, attr, named_attr, **params):
        # Ordered parameter replacement
        # 1. root parameter
        if attr in params:
            setattr(self, attr, params.pop(attr))

        # 2. resolve named replacements before removing any estimators
        items = getattr(self, named_attr)
        if items:
            names = {name for name, _ in items}
            replacements = {
                name: params.pop(name)
                for name in list(params)
                if "__" not in name and name in names
            }
            if replacements:
                estimators = [
                    replacements.get(name, estimator) for name, estimator in items
                ]
                setattr(
                    self,
                    attr,
                    [estimator for estimator in estimators if estimator is not None],
                )

        # 3. estimator parameters and other initialisation arguments
        super(_BaseXComposition, self).set_params(**params)
        return self
