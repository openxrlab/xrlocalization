import xrloc.utils.dataset as dataset

dataset_path = 'tests/data/query.txt'


def test_datset_io():
    res0 = dataset.read_ios_logger_query_dataset(dataset_path)
    assert len(res0) == 3
    assert len(res0[0]) == 2

    res1 = dataset.read_ios_logger_query_dataset(dataset_path, relative=False)
    assert len(res1) == 3
    assert len(res1[0]) == 2
