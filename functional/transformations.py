from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from itertools import (
    dropwhile,
    takewhile,
    islice,
    count,
    product,
    chain,
    starmap,
    filterfalse,
)
import collections
import types
from typing import (
    Any,
    Callable,
    Collection,
    Generator,
    Hashable,
    Iterable,
    Reversible,
    Sequence,
    overload,
)
from typing import Generic, TypeVar
import typing

if typing.TYPE_CHECKING:
    from _typeshed import SupportsRichComparison, SupportsLenAndGetItem


from functional.execution import ExecutionStrategies

_CallableT = TypeVar("_CallableT", bound=Callable)
_HashableT = TypeVar("_HashableT", bound=Hashable)
_T = TypeVar("_T")
_U = TypeVar("_U")
_K = TypeVar("_K")
_V = TypeVar("_V")
_T_co = TypeVar("_T_co", covariant=True)
_R = TypeVar("_R")
_R_co = TypeVar("_R_co", covariant=True)


#: Defines a Transformation from a name, function, and execution_strategies
@dataclass
class Transformation(Generic[_T, _R]):
    name: str
    function: Callable[[_T], _R]
    execution_strategies: Collection[int] | None


#: Cache transformation
CACHE_T = Transformation("cache", None, None)  # type: ignore


def name(function: Callable[..., Any]) -> str:
    """
    Retrieve a pretty name for the function
    :param function: function to get name from
    :return: pretty name
    """
    if isinstance(function, types.FunctionType):
        return function.__name__
    else:
        return str(function)


def map_t(func: Callable[[_T], _R]) -> Transformation[Iterable[_T], Iterable[_R]]:
    """
    Transformation for Sequence.map
    :param func: map function
    :return: transformation
    """
    return Transformation(
        f"map({name(func)})",
        partial(map, func),
        {ExecutionStrategies.PARALLEL},
    )


def select_t(func: Callable[[_T], _R]) -> Transformation[Iterable[_T], Iterable[_R]]:
    """
    Transformation for Sequence.select
    :param func: select function
    :return: transformation
    """
    return Transformation(
        f"select({name(func)})",
        partial(map, func),
        {ExecutionStrategies.PARALLEL},
    )


def starmap_t(func: Callable):
    """
    Transformation for Sequence.starmap and Sequence.smap
    :param func: starmap function
    :return: transformation
    """
    return Transformation(
        f"starmap({name(func)})",
        partial(starmap, func),
        {ExecutionStrategies.PARALLEL},
    )


def filter_t(func: Callable[[_T], Any]) -> Transformation[Iterable[_T], Iterable[_T]]:
    """
    Transformation for Sequence.filter
    :param func: filter function
    :return: transformation
    """
    return Transformation(
        f"filter({name(func)})",
        partial(filter, func),  # type: ignore[arg-type]
        {ExecutionStrategies.PARALLEL},
    )


def where_t(func: Callable[[_T], Any]) -> Transformation[Iterable[_T], Iterable[_T]]:
    """
    Transformation for Sequence.where
    :param func: where function
    :return: transformation
    """
    return Transformation(
        f"where({name(func)})",
        partial(filter, func),
        {ExecutionStrategies.PARALLEL},
    )


def filter_not_t(
    func: Callable[[_T], Any],
) -> Transformation[Iterable[_T], Iterable[_T]]:
    """
    Transformation for Sequence.filter_not
    :param func: filter_not function
    :return: transformation
    """
    return Transformation(
        f"filter_not({name(func)})",
        partial(filterfalse, func),
        {ExecutionStrategies.PARALLEL},
    )


@overload
def reversed_t() -> Transformation[Reversible[_T], Iterable[_T]]:
    ...  # type: ignore[misc]


@overload
def reversed_t() -> Transformation[SupportsLenAndGetItem[_T], Iterable[_T]]:
    ...  # type: ignore[misc]


def reversed_t():
    """
    Transformation for Sequence.reverse
    :return: transformation
    """
    return Transformation("reversed", reversed, [ExecutionStrategies.PRE_COMPUTE])


def slice_t(start: int, until: int) -> Transformation[_T, _T]:
    """
    Transformation for Sequence.slice
    :param start: start index
    :param until: until index (does not include element at until)
    :return: transformation
    """
    return Transformation(
        f"slice({start}, {until})",
        lambda sequence: islice(sequence, start, until),
        None,
    )


def distinct_t() -> Transformation[Iterable[_T], Generator[_T, None, None]]:
    """
    Transformation for Sequence.distinct
    :return: transformation
    """

    def distinct(sequence: Iterable[_T]) -> Generator[_T, None, None]:
        seen = set()
        for element in sequence:
            if element in seen:
                continue
            seen.add(element)
            yield element

    return Transformation("distinct", distinct, None)


def distinct_by_t(
    func: Callable[[_T], Hashable],
) -> Transformation[Iterable[_T], Iterable[_T]]:
    """
    Transformation for Sequence.distinct_by
    :param func: distinct_by function
    :return: transformation
    """

    def distinct_by(sequence: Iterable[_T]) -> Iterable[_T]:
        distinct_lookup: dict[Hashable, _T] = {}
        for element in sequence:
            key = func(element)
            if key not in distinct_lookup:
                distinct_lookup[key] = element
        return distinct_lookup.values()

    return Transformation(f"distinct_by({name(func)})", distinct_by, None)


def sorted_t(
    key: Callable[[_T], SupportsRichComparison] | None = None, reverse: bool = False
) -> Transformation[Iterable[_T], list[_T]]:
    """
    Transformation for Sequence.sorted
    :param key: key to sort by
    :param reverse: reverse or not
    :return: transformation
    """
    return Transformation(
        "sorted", lambda sequence: sorted(sequence, key=key, reverse=reverse), None
    )


def order_by_t(
    func: Callable[[_T], SupportsRichComparison],
) -> Transformation[Iterable[_T], list[_T]]:
    """
    Transformation for Sequence.order_by
    :param func: order_by function
    :return: transformation
    """
    return Transformation(
        f"order_by({name(func)})",
        lambda sequence: sorted(sequence, key=func),
        None,
    )


_SequenceT = TypeVar("_SequenceT", bound=typing.Sequence)


def drop_right_t(n: int) -> Transformation[_SequenceT, _SequenceT]:
    """
    Transformation for Sequence.drop_right
    :param n: number to drop from right
    :return: transformation
    """
    if n <= 0:
        end_index = None
    else:
        end_index = -n
    return Transformation(
        f"drop_right({n})",
        lambda sequence: sequence[:end_index],
        [ExecutionStrategies.PRE_COMPUTE],
    )


def drop_t(n: int) -> Transformation[Iterable[_T], Iterable[_T]]:
    """
    Transformation for Sequence.drop
    :param n: number to drop from left
    :return: transformation
    """
    return Transformation(
        f"drop({n})", lambda sequence: islice(sequence, n, None), None
    )


def drop_while_t(
    func: Callable[[_T], Any],
) -> Transformation[Iterable[_T], Iterable[_T]]:
    """
    Transformation for Sequence.drop_while
    :param func: drops while func is true
    :return: transformation
    """
    return Transformation(f"drop_while({name(func)})", partial(dropwhile, func), None)


def take_t(n: int) -> Transformation[Iterable[_T], Iterable[_T]]:
    """
    Transformation for Sequence.take
    :param n: number to take
    :return: transformation
    """
    return Transformation(f"take({n})", lambda sequence: islice(sequence, 0, n), None)


def take_while_t(
    func: Callable[[_T], Any],
) -> Transformation[Iterable[_T], Iterable[_T]]:
    """
    Transformation for Sequence.take_while
    :param func: takes while func is True
    :return: transformation
    """
    return Transformation(f"take_while({name(func)})", partial(takewhile, func), None)


def flat_map_impl(
    func: Callable[[_T], Iterable[_R]], sequence: Iterable[_T]
) -> Generator[_R, None, None]:
    """
    Implementation for flat_map_t
    :param func: function to map
    :param sequence: sequence to flat_map over
    :return: flat_map generator
    """
    for element in sequence:
        yield from func(element)


def flat_map_t(
    func: Callable[[_T], Iterable[_R]],
) -> Transformation[Iterable[_T], Generator[_R, None, None]]:
    """
    Transformation for Sequence.flat_map
    :param func: function to flat_map
    :return: transformation
    """
    return Transformation(
        f"flat_map({name(func)})",
        partial(flat_map_impl, func),
        {ExecutionStrategies.PARALLEL},
    )


def flatten_t() -> Transformation[Iterable[Iterable[_T]], Iterable[_T]]:
    """
    Transformation for Sequence.flatten
    :return: transformation
    """
    return Transformation(
        "flatten", partial(flat_map_impl, lambda x: x), {ExecutionStrategies.PARALLEL}
    )


def zip_t(
    zip_sequence: Iterable[_U],
) -> Transformation[Iterable[_T], Iterable[tuple[_T, _U]]]:
    """
    Transformation for Sequence.zip
    :param zip_sequence: sequence to zip with
    :return: transformation
    """
    return Transformation(
        "zip(<sequence>)", lambda sequence: zip(sequence, zip_sequence), None
    )


def zip_with_index_t(
    start: int,
) -> Transformation[Iterable[_T], Iterable[tuple[_T, int]]]:
    """
    Transformation for Sequence.zip_with_index
    :return: transformation
    """
    return Transformation(
        "zip_with_index", lambda sequence: zip(sequence, count(start=start)), None
    )


def enumerate_t(start: int) -> Transformation[Iterable[_T], Iterable[tuple[_T, int]]]:
    """
    Transformation for Sequence.enumerate
    :param start: start index for enumerate
    :return: transformation
    """
    return Transformation(
        "enumerate", lambda sequence: enumerate(sequence, start=start), None
    )


# TODO: Add support for multiple iterables
def cartesian_t(
    iterables: Iterable[Iterable[Any]], repeat: int
) -> Transformation[Iterable[_T], product[Any]]:
    """
    Transformation for Sequence.cartesian
    :param iterables: elements for cartesian product
    :param repeat: how many times to repeat iterables
    :return: transformation
    """
    return Transformation(
        "cartesian", lambda sequence: product(sequence, *iterables, repeat=repeat), None
    )


def init_t() -> Transformation[Sequence[_T], Sequence[_T]]:
    """
    Transformation for Sequence.init
    :return: transformation
    """
    return Transformation(
        "init", lambda sequence: sequence[:-1], {ExecutionStrategies.PRE_COMPUTE}
    )


def tail_t() -> Transformation[Iterable[_T], Iterable[_T]]:
    """
    Transformation for Sequence.tail
    :return: transformation
    """
    return Transformation("tail", lambda sequence: islice(sequence, 1, None), None)


def inits_t(
    wrap: Callable[[Iterable[_T]], _R],
) -> Transformation[typing.Sequence[_T], list[_R]]:
    """
    Transformation for Sequence.inits
    :param wrap: wrap children values with this
    :return: transformation
    """
    return Transformation(
        "inits",
        lambda sequence: [
            wrap(sequence[:i]) for i in reversed(range(len(sequence) + 1))
        ],
        {ExecutionStrategies.PRE_COMPUTE},
    )


def tails_t(
    wrap: Callable[[Iterable[_T]], _R],
) -> Transformation[typing.Sequence[_T], list[_R]]:
    """
    Transformation for Sequence.tails
    :param wrap: wrap children values with this
    :return: transformation
    """
    return Transformation(
        "tails",
        lambda sequence: [wrap(sequence[i:]) for i in range(len(sequence) + 1)],
        {ExecutionStrategies.PRE_COMPUTE},
    )


def union_t(other: Iterable[_U]) -> Transformation[Iterable[_T], set[_T | _U]]:
    """
    Transformation for Sequence.union
    :param other: sequence to union with
    :return: transformation
    """
    return Transformation("union", lambda sequence: set(sequence).union(other), None)


def intersection_t(other: Iterable[_U]) -> Transformation[Iterable[_T], set[_T]]:
    """
    Transformation for Sequence.intersection
    :param other: sequence to intersect with
    :return: transformation
    """
    return Transformation(
        "intersection", lambda sequence: set(sequence).intersection(other), None
    )


def difference_t(other: Iterable[_U]) -> Transformation[Iterable[_T], set[_T]]:
    """
    Transformation for Sequence.difference
    :param other: sequence to different with
    :return: transformation
    """
    return Transformation(
        "difference", lambda sequence: set(sequence).difference(other), None
    )


def symmetric_difference_t(
    other: Iterable[_U],
) -> Transformation[Iterable[_T], set[_T | _U]]:
    """
    Transformation for Sequence.symmetric_difference
    :param other: sequence to symmetric_difference with
    :return: transformation
    """
    return Transformation(
        "symmetric_difference",
        lambda sequence: set(sequence).symmetric_difference(other),
        None,
    )


def group_by_key_impl(
    sequence: Iterable[tuple[_K, _V]],
) -> Iterable[tuple[_K, list[_V]]]:
    """
    Implementation for group_by_key_t
    :param sequence: sequence to group
    :return: grouped sequence
    """
    result: dict[_K, list[_V]] = collections.defaultdict(list)
    for element in sequence:
        result[element[0]].append(element[1])
    return result.items()


def group_by_key_t() -> (
    Transformation[Iterable[tuple[_K, _V]], Iterable[tuple[_K, list[_V]]]]
):
    """
    Transformation for Sequence.group_by_key
    :return: transformation
    """
    return Transformation("group_by_key", group_by_key_impl, None)


# TODO: CHECKING TYPING AGAINS PIPELINE IMPL
def reduce_by_key_impl(
    func: Callable[[_V, _V], _V], sequence: Iterable[tuple[_K, _V]]
) -> Iterable[tuple[_K, _V]]:
    """
    Implementation for reduce_by_key_t
    :param func: reduce function
    :param sequence: sequence to reduce
    :return: reduced sequence
    """
    result: dict[_K, _V] = {}
    for key, value in sequence:
        if key in result:
            result[key] = func(result[key], value)
        else:
            result[key] = value
    return result.items()


def reduce_by_key_t(
    func: Callable[[_V, _V], _V],
) -> Transformation[Iterable[tuple[_K, _V]], Iterable[tuple[_K, _V]]]:
    """
    Transformation for Sequence.reduce_by_key
    :param func: reduce function
    :return: transformation
    """
    return Transformation(
        f"reduce_by_key({name(func)})", partial(reduce_by_key_impl, func), None
    )


def accumulate_impl(
    func: Callable[[_R, _T], _R] | None, sequence: Iterable[_T]
) -> Iterable[_R]:
    # pylint: disable=no-name-in-module
    """
    Implementation for accumulate
    :param sequence: sequence to accumulate
    :param func: accumulate function
    """
    from itertools import accumulate

    return accumulate(sequence, func)


def accumulate_t(
    func: Callable[[_R, _T], _R] | None,
) -> Transformation[Iterable[_T], Iterable[_R]]:
    """
    Transformation for Sequence.accumulate
    """
    return Transformation(
        f"accumulate({name(func)})", partial(accumulate_impl, func), None
    )


def count_by_key_impl(sequence: Iterable[tuple[_K, _V]]) -> Iterable[tuple[_K, int]]:
    """
    Implementation for count_by_key_t
    :param sequence: sequence of (key, value) pairs
    :return: counts by key
    """
    counter: dict[_K, int] = collections.Counter()
    for key, _ in sequence:
        counter[key] += 1
    return counter.items()


def count_by_key_t() -> (
    Transformation[Iterable[tuple[_K, _V]], Iterable[tuple[_K, int]]]
):
    """
    Transformation for Sequence.count_by_key
    :return: transformation
    """
    return Transformation("count_by_key", count_by_key_impl, None)


def count_by_value_impl(
    sequence: Iterable[_HashableT],
) -> Iterable[tuple[_HashableT, int]]:
    """
    Implementation for count_by_value_t
    :param sequence: sequence of values
    :return: counts by value
    """
    counter = collections.Counter()
    for e in sequence:
        counter[e] += 1
    return counter.items()


def count_by_value_t() -> (
    Transformation[Iterable[_HashableT], Iterable[tuple[_HashableT, int]]]
):
    """
    Transformation for Sequence.count_by_value
    :return: transformation
    """
    return Transformation("count_by_value", count_by_value_impl, None)


def group_by_impl(
    func: Callable[[_T], _HashableT], sequence: Iterable[_T]
) -> Iterable[tuple[_HashableT, list[_T]]]:
    """
    Implementation for group_by_t
    :param func: grouping function
    :param sequence: sequence to group
    :return: grouped sequence
    """
    result: dict[_HashableT, list[_T]] = collections.defaultdict(list)
    for element in sequence:
        result[func(element)].append(element)
    return result.items()


def group_by_t(
    func: Callable[[_T], _HashableT],
) -> Transformation[Iterable[_T], Iterable[tuple[_HashableT, list[_T]]]]:
    """
    Transformation for Sequence.group_by
    :param func: grouping function
    :return: transformation
    """
    return Transformation(f"group_by({name(func)})", partial(group_by_impl, func), None)


def grouped_impl(size: int, sequence: Iterable[_T]) -> Generator[list[_T], None, None]:
    """
    Implementation for grouped_t
    :param size: size of groups
    :param sequence: sequence to group
    :return: grouped sequence
    """
    iterator = iter(sequence)
    try:
        while True:
            batch = islice(iterator, size)
            yield list(chain((next(batch),), batch))
    except StopIteration:
        return


def grouped_t(
    size: int,
) -> Transformation[Iterable[_T], Generator[list[_T], None, None]]:
    """
    Transformation for Sequence.grouped
    :param size: size of groups
    :return: transformation
    """
    return Transformation(f"grouped({size})", partial(grouped_impl, size), None)


def sliding_impl(
    wrap: Callable[..., _R], size: int, step: int, sequence: typing.Sequence[_T]
) -> Generator[_R, None, None]:
    """
    Implementation for sliding_t
    :param wrap: wrap children values with this
    :param size: size of window
    :param step: step size
    :param sequence: sequence to create sliding windows from
    :return: sequence of sliding windows
    """
    i = 0
    n = len(sequence)
    while i + size <= n or (step != 1 and i < n):
        yield wrap(sequence[i : i + size])
        i += step


def sliding_t(
    wrap: Callable[..., _R], size: int, step: int
) -> Transformation[typing.Sequence[_T], Generator[_R, None, None]]:
    """
    Transformation for Sequence.sliding
    :param wrap: wrap children values with this
    :param size: size of window
    :param step: step size
    :return: transformation
    """
    return Transformation(
        f"sliding({size}, {step})",
        partial(sliding_impl, wrap, size, step),
        {ExecutionStrategies.PRE_COMPUTE},
    )


def partition_impl(
    wrap: Callable[[list[_T]], _R],
    predicate: Callable[[_T], Any],
    sequence: Iterable[_T],
) -> tuple[_R, _R]:
    truthy_partition = []
    falsy_partition = []
    for e in sequence:
        if predicate(e):
            truthy_partition.append(e)
        else:
            falsy_partition.append(e)

    return wrap((wrap(truthy_partition), wrap(falsy_partition)))


def partition_t(
    wrap: Callable[[list[_T]], _R], func: Callable[[_T], Any]
) -> Transformation[Iterable[_T], tuple[_R, _R]]:
    """
    Transformation for Sequence.partition
    :param wrap: wrap children values with this
    :param func: partition function
    :return: transformation
    """
    return Transformation(
        f"partition({name(func)})", partial(partition_impl, wrap, func), None
    )


def inner_join_impl(other, sequence):
    """
    Implementation for part of join_impl
    :param other: other sequence to join with
    :param sequence: first sequence to join with
    :return: joined sequence
    """
    seq_dict = {}
    for element in sequence:
        seq_dict[element[0]] = element[1]
    seq_kv = seq_dict
    other_kv = dict(other)
    keys = seq_kv.keys() if len(seq_kv) < len(other_kv) else other_kv.keys()
    result = {}
    for k in keys:
        if k in seq_kv and k in other_kv:
            result[k] = (seq_kv[k], other_kv[k])
    return result.items()


def join_impl(other, join_type, sequence):
    """
    Implementation for join_t
    :param other: other sequence to join with
    :param join_type: join type (inner, outer, left, right)
    :param sequence: first sequence to join with
    :return: joined sequence
    """
    if join_type == "inner":
        return inner_join_impl(other, sequence)
    seq_dict = {}
    for element in sequence:
        seq_dict[element[0]] = element[1]
    seq_kv = seq_dict
    other_kv = dict(other)
    if join_type == "left":
        keys = seq_kv.keys()
    elif join_type == "right":
        keys = other_kv.keys()
    elif join_type == "outer":
        keys = set(list(seq_kv.keys()) + list(other_kv.keys()))
    else:
        raise TypeError("Wrong type of join specified")
    result = {}
    for k in keys:
        result[k] = (seq_kv.get(k), other_kv.get(k))
    return result.items()


def join_t(other, join_type):
    """
    Transformation for Sequence.join, Sequence.inner_join, Sequence.outer_join, Sequence.right_join,
    and Sequence.left_join
    :param other: other sequence to join with
    :param join_type: join type from left, right, inner, and outer
    :return: transformation
    """
    return Transformation(
        f"{join_type}_join", partial(join_impl, other, join_type), None
    )


def peek_impl(
    func: Callable[[_T], Any], sequence: Iterable[_T]
) -> Generator[_T, None, None]:
    """
    Implementation for peek_t
    :param func: apply func
    :param sequence: sequence to peek
    :return: original sequence
    """
    for element in sequence:
        func(element)
        yield element


def peek_t(func: Callable[[_T], Any]) -> Transformation[_T, _T]:
    """
    Transformation for Sequence.peek
    :param func: peek function
    :return: transformation
    """
    return Transformation(f"peek({name(func)})", partial(peek_impl, func), None)
