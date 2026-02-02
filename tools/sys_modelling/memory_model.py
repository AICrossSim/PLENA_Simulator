class memory_model:
    def __init__(self, memory_capacity, memory_bandwidth, access_latency):
        self.memory_capacity = memory_capacity
        self.memory_bandwidth = memory_bandwidth
        self.access_latency = access_latency




class Three_Dim_SRAM(memory_model):
    def __init__(self, memory_capacity, memory_bandwidth, access_latency, layers):
        super().__init__(memory_capacity, memory_bandwidth, access_latency)
        self.layers = layers

    def get_effective_bandwidth(self):
        return self.memory_bandwidth * self.layers