import sys
import json
import pickle

inp = sys.argv[1]
out = sys.argv[2]


def get_method_and_N_from_benchmark_name(name):
    name, params = name.split('[')
    params = params[:-1]
    params_split = params.split('-')
    if len(params_split) == 2:
        return params_split[1], int(params_split[0])
    else:
        return name, int(params_split[0])

if __name__ == "__main__":

    results = {}
    with open(inp, 'r') as inp_fh:
        data = json.load(inp_fh)

        for benchmark in data["benchmarks"]:
            method, N = get_method_and_N_from_benchmark_name(benchmark["name"])
            if method not in results:
                results[method] = {}
            results[method][N] = benchmark['stats']['data']

    with open(out, 'wb') as out_fh:
        pickle.dump(results, out_fh, protocol=pickle.HIGHEST_PROTOCOL)



