import pdb
from benchmark_datasets.data_loader import load_informbench_benchmark_data

data = load_informbench_benchmark_data(
    data_path="./benchmark_datasets/data",
    debug=True
)

pdb.set_trace()

print(data)