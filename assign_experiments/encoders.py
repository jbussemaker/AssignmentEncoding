from assign_enc.lazy.encodings import *
from assign_enc.lazy.imputation import *
from assign_enc.eager.encodings import *
from assign_enc.eager.imputation import *

__all__ = ['EAGER_ENCODERS', 'EAGER_IMPUTERS', 'DEFAULT_EAGER_ENCODER', 'DEFAULT_EAGER_IMPUTER',
           'LAZY_ENCODERS', 'LAZY_IMPUTERS', 'DEFAULT_LAZY_ENCODER', 'DEFAULT_LAZY_IMPUTER']


EAGER_ENCODERS = [
    lambda imp: OneVarEncoder(imp),
    lambda imp: DirectMatrixEncoder(imp),
    lambda imp: DirectMatrixEncoder(imp, remove_gaps=False),
    lambda imp: ElementGroupedEncoder(imp),
    lambda imp: ElementGroupedEncoder(imp, normalize_within_group=False),

    lambda imp: AmountFirstGroupedEncoder(imp, TotalAmountGrouper(), OneVarLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, SourceAmountGrouper(), OneVarLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, SourceAmountFlattenedGrouper(), OneVarLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, TargetAmountGrouper(), OneVarLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, TargetAmountFlattenedGrouper(), OneVarLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, TotalAmountGrouper(), FlatIndexLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, TotalAmountGrouper(), RelFlatIndexLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, TotalAmountGrouper(), CoordIndexLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, TotalAmountGrouper(), RelCoordIndexLocationGrouper()),
]

DEFAULT_EAGER_ENCODER = lambda: OneVarEncoder(DEFAULT_EAGER_IMPUTER())

EAGER_IMPUTERS = [
    lambda: FirstImputer(),
    lambda: AutoModImputer(),
    lambda: AutoModImputer(reverse=True),
    lambda: ClosestImputer(),
    lambda: ClosestImputer(euclidean=False),
    lambda: ConstraintViolationImputer(),
]

DEFAULT_EAGER_IMPUTER = AutoModImputer

LAZY_ENCODERS = [
    lambda imp: LazyDirectMatrixEncoder(imp),

    lambda imp: LazyAmountFirstEncoder(imp, FlatLazyAmountEncoder(), FlatLazyConnectionEncoder()),
    lambda imp: LazyAmountFirstEncoder(imp, TotalLazyAmountEncoder(), FlatLazyConnectionEncoder()),
    lambda imp: LazyAmountFirstEncoder(imp, SourceLazyAmountEncoder(), FlatLazyConnectionEncoder()),
    lambda imp: LazyAmountFirstEncoder(imp, SourceTargetLazyAmountEncoder(), FlatLazyConnectionEncoder()),
]

DEFAULT_LAZY_ENCODER = lambda: LazyDirectMatrixEncoder(DEFAULT_LAZY_IMPUTER())

LAZY_IMPUTERS = [
    lambda: LazyFirstImputer(),
    lambda: LazyDeltaImputer(),
    lambda: LazyClosestImputer(),
    lambda: LazyClosestImputer(euclidean=False),
    lambda: LazyConstraintViolationImputer(),
]

DEFAULT_LAZY_IMPUTER = LazyDeltaImputer
