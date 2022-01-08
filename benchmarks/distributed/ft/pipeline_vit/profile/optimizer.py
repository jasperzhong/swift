from torch.distributed.fault_tolerance import compute_logging_size


def merge_groups(threshold, bandwidth, checkpoint_interval, num_micro_batches, num_machines=16, workers_per_machine=8):
    # TODO: check how to read the file
    recovery_time = []
    for i in range(num_machines):
        with open(f"compute_time_{i}.txt", "r") as f:
            time = float(f.read())
            recovery_time.append(time)
    print("recovery time {}".format(recovery_time))
    activation_size = []
    
    activation_size = compute_logging_size(num_micro_batches, file="../profile.txt", num_machines=num_machines)
    
    activation_sum = sum(activation_size) * checkpoint_interval
    print(f"activation_sum {activation_sum}")
    group_size = len(recovery_time)
    assert group_size == num_machines, "initial group size must be equal to num of machines"
    group = [[i] for i in range(group_size)]
    while threshold < activation_sum:
        print("get in the loop")
        min_r_m = float('inf')
        merge_id = -1
        min_r_merge = 0
        min_m_merge = 0
        for i in range(group_size - 1):
            r_merge = recovery_time[i] + recovery_time[i + 1] + activation_size[i] / bandwidth
            delta_m = activation_size[i] * checkpoint_interval
            delta_r = r_merge * 2 * workers_per_machine / num_machines \
                        - recovery_time[i] * workers_per_machine / num_machines \
                        - recovery_time[i + 1] * workers_per_machine / num_machines

            r_m = delta_r / delta_m
            if r_m < min_r_m:
                min_r_m = r_m
                merge_id = i
                min_r_merge = r_merge
                min_m_merge = delta_m
        
        # udpate recovery time
        recovery_time[merge_id] = min_r_merge
        activation_sum -= min_m_merge
        a = group[merge_id]
        b = group[merge_id + 1]
        a.extend(b)
        group[merge_id] = a
        del group[merge_id + 1]
        del activation_size[merge_id]
        del recovery_time[merge_id + 1] 
        group_size -= 1
    
    print(f"the merged group is {group}")
    return group 

if __name__ == "__main__":
    threshold = 8e11
    bandwidth = 5e10
    checkpoint_interval = 100
    num_micro_batches = 128
    merge_groups(threshold, bandwidth, checkpoint_interval, num_micro_batches, num_machines=16, workers_per_machine=8)