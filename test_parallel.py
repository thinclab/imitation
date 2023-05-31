from multiprocessing import Pool
  
  
def print_range(range):
    
    # print range
    print('From {} to {}:'.format(range[0], range[1]))
    return range
  
def run_parallel():
    
    # list of ranges
    list_ranges = [[0, 10], [10, 20], [20, 30]]
  
    # pool object with number of elements in the list
    pool = Pool(processes=len(list_ranges))
  
    # map the function to the list and pass 
    # function and list_ranges as arguments
    result = pool.map(print_range, list_ranges)
    print(list(result))

    # for result in pool.map(print_range, list_ranges):
    #     print(result)
  
# Driver code
if __name__ == '__main__':
    run_parallel()

# from concurrent.futures import ThreadPoolExecutor

# def call_venv_gibbs(list_inputs):
    
#     venv = list_inputs[0]
#     mean_Gs_s_g_t, cov_Gs_s_g_t, mean_Gs_a_g_t, cov_Gs_a_g_t = \
#         venv.env_method(method_name='gibbs_sampling_mean_cov',indices=[0],\
#         list_inputs=list_inputs[1:])[0]
    
#     return [mean_Gs_s_g_t, cov_Gs_s_g_t, mean_Gs_a_g_t, cov_Gs_a_g_t]

# def my_func(num):
#     return num + 1

# def worker(data):
#     with ThreadPoolExecutor(len(data)) as executor:
#         result = executor.map(my_func, data)
#         # print(result)
#         # result = executor.map(call_venv_gibbs, data)

#     return result


# if __name__ == "__main__":
#     arr = [1, 2, 3, 4, 5]
#     print(arr)
#     arr = worker(arr)
#     print(tuple(arr))    