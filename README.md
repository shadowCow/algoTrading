# algoTrading

## Running the Pipeline
From the root of the project:
```python
$ python -m algoTrading.evaluation_pipeline <data_directory>
```
To use the included futures price data set:
```python
$ python -m algoTrading.evaluation_pipeline futures_price_data
```

### Running the Tests
To run all tests...

From the root of the project:
```python
$ python -m unittest
```

To run a specific test...
```python
$ python -m unittest tests.{name_of_test}
```

For example:
```python
$ python -m unittest tests.test_feature
```
