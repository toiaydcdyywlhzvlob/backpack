#pytest additional_benchmark_cifar10_sigmoid.py -v --benchmark-json=additional_benchmark_cifar10_sigmoid.json --benchmark-min-rounds=20
#pytest additional_benchmark_cifar10.py -v --benchmark-json=additional_benchmark_cifar10.json --benchmark-min-rounds=20
#pytest benchmark_cifar10.py -v --benchmark-json=benchmark_cifar10.json --benchmark-min-rounds=20
#pytest benchmark_cifar100.py -v --benchmark-json=benchmark_cifar100.json --benchmark-min-rounds=20


python translate_json_to_pk.py additional_benchmark_cifar10_sigmoid.json additional_benchmark_cifar10_sigmoid.pk
python translate_json_to_pk.py additional_benchmark_cifar10.json additional_benchmark_cifar10.pk
python translate_json_to_pk.py benchmark_cifar10.json benchmark_cifar10.pk
python translate_json_to_pk.py benchmark_cifar100.json benchmark_cifar100.pk