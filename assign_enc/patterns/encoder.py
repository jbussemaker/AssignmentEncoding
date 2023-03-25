import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.lazy_encoding import *

__all__ = ['InvalidPatternEncoder', 'PatternEncoderBase']


class InvalidPatternEncoder(RuntimeError):
    pass


class PatternEncoderBase(LazyEncoder):
    """
    Base class for encoders that can only be used for specific connection patterns. The advantage is that encoder is
    very fast, optimal in terms of imputation ratio and distance correlation, and has custom imputation that is better
    and faster than automatic imputation.

    However, it only applies to a limited set of connection settings. When incompatible settings are provided to
    set_settings, an InvalidPatternEncoder exception is raised.
    """

    def __init__(self, imputer):
        super().__init__(imputer)
        self._settings: Optional[MatrixGenSettings] = None
        self._effective_settings = None
        self._is_transpose = False
        self._transpose_existence_map = {}

    def set_settings(self, settings: MatrixGenSettings):
        # Get effective connection settings and check if all of them are appropriate for this pattern encoder
        try:
            self._is_transpose = False
            self._try_settings(settings)
            self._transpose_existence_map = {}

        except InvalidPatternEncoder:
            # Check if the transpose settings would work (as all patterns are implemented in one way only)
            self._is_transpose = True
            self._try_settings(settings.get_transpose_settings())
            self._transpose_existence_map = {existence: existence.get_transpose() for existence in (
                settings.existence.patterns if settings.existence is not None else [NodeExistence()])}

        # Encode
        super().set_settings(settings)

    def _try_settings(self, settings: MatrixGenSettings):
        self._settings = settings
        self._effective_settings = settings.get_effective_settings()
        if not self._is_compatible_effective(self._effective_settings):
            raise InvalidPatternEncoder(f'Invalid pattern encoder {self!r}')

    def is_compatible(self, settings: MatrixGenSettings) -> bool:
        """Returns whether this encoder is compatible with the provided matrix settings"""
        if self._is_compatible_effective(settings.get_effective_settings()):
            return True
        if self._is_compatible_effective(settings.get_transpose_settings().get_effective_settings()):
            return True
        return False

    def _is_compatible_effective(self, effective_settings) -> bool:
        initialize = True
        for effective_settings, _, _ in effective_settings.values():
            if len(effective_settings.src) == 0 or len(effective_settings.tgt) == 0:
                continue
            if not self._matches_pattern(effective_settings, initialize=initialize):
                return False
            initialize = False
        return True

    def _matches_pattern(self, effective_settings: MatrixGenSettings, initialize: bool) -> bool:
        """Returns whether this encoder can be used for these matrix settings, i.e. whether it matches the pattern"""
        raise NotImplementedError

    def _encode(self, existence: NodeExistence) -> List[DiscreteDV]:
        if self._is_transpose:
            if existence not in self._transpose_existence_map:
                return []
            existence = self._transpose_existence_map[existence]
        if existence not in self._effective_settings:
            return []

        effective_settings, _, _ = self._effective_settings[existence]
        if len(effective_settings.src) == 0 or len(effective_settings.tgt) == 0:
            return []

        return self._encode_effective(effective_settings, existence)

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[Tuple[DesignVector, np.ndarray]]:
        if self._is_transpose:
            if existence not in self._transpose_existence_map:
                return vector, self.get_empty_matrix()
            existence = self._transpose_existence_map[existence]
        if existence not in self._effective_settings:
            return vector, self.get_empty_matrix()

        effective_settings, src_map, tgt_map = self._effective_settings[existence]
        if len(effective_settings.src) == 0 or len(effective_settings.tgt) == 0:
            return vector, self.get_empty_matrix()

        vector, matrix = self._decode_effective(vector, effective_settings, existence)
        expanded_matrix = self._settings.expand_effective_matrix(matrix, src_map, tgt_map)

        if self._is_transpose:
            expanded_matrix = expanded_matrix.T

        return vector, expanded_matrix

    def _impute(self, vector, matrix, existence: NodeExistence) -> Tuple[DesignVector, np.ndarray]:
        raise RuntimeError('Pattern encoder should never (automatically) impute!')

    def _encode_effective(self, effective_settings: MatrixGenSettings, existence: NodeExistence) -> List[DiscreteDV]:
        """Encode an effective settings pattern"""
        raise NotImplementedError

    def _decode_effective(self, vector: DesignVector, effective_settings: MatrixGenSettings, existence: NodeExistence) \
            -> Tuple[DesignVector, np.ndarray]:
        """Decode and correct a design vector"""
        raise NotImplementedError

    def _pattern_name(self) -> str:
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r})'

    def __str__(self):
        return f'{self._pattern_name()} Pattern Encoder'
