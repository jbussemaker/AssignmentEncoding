import pytest
import itertools
import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.patterns.encoder import *
from assign_enc.patterns.patterns import *
from assign_enc.lazy.imputation.first import *


@pytest.fixture
def settings():
    settings_map = {
        'combining': MatrixGenSettings(src=[Node([1], repeated_allowed=False)],
                                       tgt=[Node([0, 1], repeated_allowed=False) for _ in range(3)]),
        'assigning': MatrixGenSettings(src=[Node(min_conn=0, repeated_allowed=False) for _ in range(2)],
                                       tgt=[Node(min_conn=0, repeated_allowed=False) for _ in range(2)]),
        'assigning_sur': MatrixGenSettings(src=[Node(min_conn=0, repeated_allowed=False) for _ in range(2)],
                                           tgt=[Node(min_conn=1, repeated_allowed=False) for _ in range(2)]),
        'assigning_inj': MatrixGenSettings(src=[Node(min_conn=0, repeated_allowed=False) for _ in range(2)],
                                           tgt=[Node([0, 1], repeated_allowed=False) for _ in range(2)]),
        'assigning_rep': MatrixGenSettings(src=[Node(min_conn=0) for _ in range(2)],
                                           tgt=[Node(min_conn=0) for _ in range(2)]),
        'assigning_sur_rep': MatrixGenSettings(src=[Node(min_conn=0) for _ in range(2)],
                                               tgt=[Node(min_conn=1) for _ in range(2)]),
        'assigning_inj_rep': MatrixGenSettings(src=[Node(min_conn=0) for _ in range(2)],
                                               tgt=[Node([0, 1]) for _ in range(2)]),
        'partitioning': MatrixGenSettings(src=[Node(min_conn=0, repeated_allowed=False) for _ in range(3)],
                                          tgt=[Node([1], repeated_allowed=False) for _ in range(3)]),
        'downselecting': MatrixGenSettings(src=[Node(min_conn=0, repeated_allowed=False)],
                                           tgt=[Node([0, 1], repeated_allowed=False) for _ in range(3)]),
        'connecting': MatrixGenSettings(src=[Node(min_conn=0, repeated_allowed=False) for _ in range(2)],
                                        tgt=[Node(min_conn=0, repeated_allowed=False) for _ in range(2)],
                                        excluded=[(i, j) for i in range(2) for j in range(2) if i >= j]),
        'connecting_dir': MatrixGenSettings(src=[Node(min_conn=0, repeated_allowed=False) for _ in range(2)],
                                            tgt=[Node(min_conn=0, repeated_allowed=False) for _ in range(2)],
                                            excluded=[(i, j) for i in range(2) for j in range(2) if i == j]),
        'permuting': MatrixGenSettings(src=[Node([1], repeated_allowed=False) for _ in range(3)],
                                       tgt=[Node([1], repeated_allowed=False) for _ in range(3)]),
        'unordered_combining': MatrixGenSettings(src=[Node([2])],
                                                 tgt=[Node(min_conn=0) for _ in range(3)]),
        'unordered_norepl_combining': MatrixGenSettings(src=[Node([2], repeated_allowed=False)],
                                                        tgt=[Node([0, 1], repeated_allowed=False) for _ in range(3)]),
        'nomatch': MatrixGenSettings(src=[Node([1]), Node(min_conn=0), Node([0, 1], repeated_allowed=False)],
                                     tgt=[Node([1, 2, 3], repeated_allowed=False), Node(min_conn=0)]),
    }

    for key, settings in list(settings_map.items()):
        settings_map[key+'_transpose'] = settings.get_transpose_settings()

    return settings_map


def _do_test_encoders(encoder_cls: Type[PatternEncoderBase], settings_map, match_keys):
    match_keys += [key+'_transpose' for key in match_keys]

    encoders = []
    for key, settings in settings_map.items():
        for include_empty in [False, True]:
            if include_empty:
                src, tgt = settings.src, settings.tgt
                settings = MatrixGenSettings(
                    src=src, tgt=tgt, excluded=settings.excluded, existence=NodeExistencePatterns(patterns=[
                        NodeExistence(),
                        NodeExistence(src_exists=[False]*len(src), tgt_exists=[False]*(len(tgt))),
                    ]))

            # Check that the correct encoder is matched with the settings
            try:
                encoder = encoder_cls(LazyFirstImputer())
                encoder.set_settings(settings)
                assert key in match_keys

                if not include_empty:
                    encoders.append(encoder)

            except InvalidPatternEncoder:
                assert key not in match_keys
                continue

            # Enumerate all design variables to check pattern-provided imputation
            matrix_gen = AggregateAssignmentMatrixGenerator(settings)
            agg_matrix = matrix_gen.get_agg_matrix(cache=False)[NodeExistence()]
            agg_matrix_set = {tuple(flat_matrix) for flat_matrix in
                              agg_matrix.reshape(agg_matrix.shape[0], np.prod(agg_matrix.shape[1:]))}

            try:
                seen_dvs = set()
                for des_vector in itertools.product(*[list(range(dv.n_opts+1)) for dv in encoder.design_vars]):
                    imp_dv, matrix = encoder.get_matrix(list(des_vector)+[0])
                    assert len(imp_dv) == len(encoder.design_vars)+1
                    assert tuple(matrix.ravel()) in agg_matrix_set
                    seen_dvs.add(tuple(imp_dv))

                assert len(seen_dvs) == len(agg_matrix_set)

            except:
                print(repr(encoder), key)
                raise

    return encoders


def test_combining_encoder(settings):
    _do_test_encoders(CombiningPatternEncoder, settings, ['combining'])


def test_assigning_encoder(settings):
    encoders = _do_test_encoders(
        AssigningPatternEncoder, settings, ['assigning', 'assigning_sur', 'assigning_rep', 'assigning_sur_rep'])

    def _check_settings(encoder: Any, surjective=False, repeatable=False):
        assert isinstance(encoder, AssigningPatternEncoder)
        assert encoder.surjective == surjective
        assert encoder.repeatable == repeatable
        if repeatable:
            assert encoder._n_max > 1
        else:
            assert encoder._n_max == 1

    _check_settings(encoders[0])
    _check_settings(encoders[1], surjective=True)
    _check_settings(encoders[2], repeatable=True)
    _check_settings(encoders[3], surjective=True, repeatable=True)


def test_partitioning_encoder(settings):
    _do_test_encoders(PartitioningPatternEncoder, settings, ['partitioning'])


def test_downselecting_encoder(settings):
    _do_test_encoders(DownselectingPatternEncoder, settings, ['assigning_inj', 'assigning_inj_rep', 'downselecting'])


def test_connecting_encoder(settings):
    encoders = _do_test_encoders(ConnectingPatternEncoder, settings, ['connecting', 'connecting_dir'])

    enc = encoders[0]
    assert isinstance(enc, ConnectingPatternEncoder)
    assert not enc.directed

    enc = encoders[1]
    assert isinstance(enc, ConnectingPatternEncoder)
    assert enc.directed


def test_permuting_encoder(settings):
    encoders = _do_test_encoders(PermutingPatternEncoder, settings, ['permuting'])
    assert encoders[0].get_distance_correlation()


def test_unordered_combining_encoder(settings):
    _do_test_encoders(
        UnorderedCombiningPatternEncoder, settings, ['unordered_norepl_combining', 'unordered_combining'])
