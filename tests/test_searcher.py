import pytest
import pandas as pd

from cause2e._searcher import (
    TetradSearcher,
    query_searcher_attribute_in_separate_process,
    query_searcher_attribute,
    query_searcher_method,
    _run_function_in_separate_process,
)


@pytest.fixture
def searcher_input():
    data = pd.DataFrame([1, 2, 3])
    continuous = {1}
    discrete = set()
    knowledge = None
    return (data, continuous, discrete, knowledge)


@pytest.fixture
def searcher_sprinkler():
    data = pd.read_csv('tests/fixtures/data/binary_sprinkler.csv', index_col=0)
    continuous = set()
    discrete = set(data.columns)
    knowledge = None
    return TetradSearcher(data, continuous, discrete, knowledge)


@pytest.mark.parametrize("requires_vm", [True, False])
def test_minimal_multiprocessing(requires_vm):
    output = _run_function_in_separate_process(
        func=_example_fun,
        requires_vm=requires_vm,
        word='hello',
    )
    assert output == 'hello'


def _example_fun(word):
    return word


def test_attribute_query(searcher_input):
    without_mp = query_searcher_attribute(searcher_input, '_separator')
    with_mp = query_searcher_attribute_in_separate_process(searcher_input, '_separator')
    assert without_mp == with_mp


def test_method_query_no_args(searcher_input):
    output = _run_function_in_separate_process(
        func=query_searcher_method,
        requires_vm=True,
        searcher_input=searcher_input,
        method_name='show_search_algos',
    )
    assert 'fges' in output


@pytest.mark.parametrize('method_name', ['show_search_algos', 'show_search_scores', 'show_independence_tests'])
def test_info_method_no_args(method_name):
    searcher = TetradSearcher(0, 0, 0, 0)
    func = getattr(searcher, method_name)
    output = _run_function_in_separate_process(
        func=func,
        requires_vm=True,
    )
    # test different words for algos, scores and independence tests
    assert (test_word in output for test_word in ['fges', 'degen-gauss-bic', 'fisher-z-test'])


def test_show_algo_info():
    searcher = TetradSearcher(0, 0, 0, 0)
    output = _run_function_in_separate_process(
        func=searcher.show_algo_info,
        requires_vm=True,
        algo_name='fges',
    )
    assert 'FGES is an optimized' in output


def test_show_algo_params():
    searcher = TetradSearcher(0, 0, 0, 0)
    output = _run_function_in_separate_process(
        func=searcher.show_algo_params,
        requires_vm=True,
        algo_name='fges',
        score_name='degen-gauss-bic',
    )
    assert 'faithfulnessAssumed' in output


def test_get_type_threshold_in_separate_process(searcher_sprinkler):
    threshold = _run_function_in_separate_process(
        func=searcher_sprinkler._get_type_threshold,
        requires_vm=False,
        complete=True,
    )
    assert threshold == 3  # data is binary


def test_run_search_in_separate_process(searcher_sprinkler):
    graph = _run_function_in_separate_process(
        func=searcher_sprinkler.run_search,
        requires_vm=True,
        algo='fges',
        use_knowledge=False,
        score='cg-bic-score',
        verbose=True,
        keep_vm=False,
    )
    assert "Sprinkler" in graph.nodes
