import numpy as np
from abc import ABC, abstractmethod

class BaseSlice(ABC):
    def __init__(self, name, priority, min_prbs):
        self.name = name
        self.priority = priority
        self.min_prbs = min_prbs
        self.current_prbs = min_prbs
        self.traffic_demand = 0.0
        self.latency = 0.0
        self.throughput = 0.0
        self.packet_loss = 0.0
        self.qos_violations = 0
        self.total_capacity = 0.0
        # Physical Layer parameters
        self.tx_power_dbm = 46.0  # gNB Tx Powe
        self.noise_power_dbm = -174.0 + 10 * np.log10(180000 * 12)
        self.scs_khz = 30  # Subcarrier spacing
        
    def calculate_path_loss(self, distance_m):
        """3GPP"""
        d_km = max(0.01, distance_m / 1000.0)
        return 128.1 + 37.6 * np.log10(d_km)

    def calculate_sinr(self, distance_m):
        pl = self.calculate_path_loss(distance_m)
        rx_power = self.tx_power_dbm - pl
        # Simple SINR: Signal / (Noise + small Interference)
        interference_dbm = -90.0
        sinr_linear = 10**((rx_power - self.noise_power_dbm) / 10.0)
        return max(0.1, sinr_linear)

    @abstractmethod
    def calculate_metrics(self, ues_data):
        # UE distance and traffic
        pass

    def update_resource(self, prbs):
        self.current_prbs = max(self.min_prbs, int(prbs))

class eMBBSlice(BaseSlice):
    def __init__(self):
        super().__init__(name="eMBB", priority=1, min_prbs=20)
        self.target_throughput_per_ue = 20.0 # Mbps

    def calculate_metrics(self, ues_data):
        self.traffic_demand = sum([ue['demand'] for ue in ues_data])
        if not ues_data:
            self.throughput = 0
            self.latency = 0
            self.qos_violations = 0
            return

        # Spectral efficiency per UE
        total_capacity = 0
        prbs_per_ue = self.current_prbs / len(ues_data)
        avg_sinr = 0
        for ue in ues_data:
            sinr = self.calculate_sinr(ue['distance'])
            avg_sinr += sinr
            # Shannon capacity: BW * log2(1 + SINR)
            # BW per PRB = 180kHz (12 subcarriers * 15kHz) or 360kHz (30kHz)
            # For 30kHz SCS, 12 subcarriers = 360kHz
            ue_capacity = (prbs_per_ue * 0.360) * np.log2(1 + sinr)
            total_capacity += ue_capacity  
        self.avg_sinr = avg_sinr / len(ues_data)
        self.throughput = min(self.traffic_demand, total_capacity)
        load_factor = self.traffic_demand / (total_capacity + 1e-6)
        self.latency = 10.0 / (1.0 - min(0.99, load_factor))
        self.packet_loss = max(0, load_factor - 1.0) * 0.1
        self.qos_violations = 1 if self.throughput < self.traffic_demand * 0.7 else 0
        self.total_capacity = total_capacity

class URLLCSlice(BaseSlice):
    def __init__(self):
        super().__init__(name="URLLC", priority=3, min_prbs=10)
        self.target_latency = 5.0 # ms

    def calculate_metrics(self, ues_data):
        self.traffic_demand = sum([ue['demand'] for ue in ues_data])
        if not ues_data:
            self.throughput = 0
            self.latency = 0
            self.qos_violations = 0
            return
        total_capacity = 0
        prbs_per_ue = self.current_prbs / len(ues_data)
        avg_sinr = 0
        for ue in ues_data:
            sinr = self.calculate_sinr(ue['distance'])
            avg_sinr += sinr
            # Lower efficiency due to robust MCS in URLLC
            ue_capacity = (prbs_per_ue * 0.360) * np.log2(1 + sinr) * 0.6 
            total_capacity += ue_capacity
            
        self.avg_sinr = avg_sinr / len(ues_data)
        self.throughput = min(self.traffic_demand, total_capacity)
        load_factor = self.traffic_demand / (total_capacity + 1e-6)
        self.latency = 2.0 / (1.0 - min(0.95, load_factor))
        self.qos_violations = 1 if self.latency > self.target_latency else 0
        self.total_capacity = total_capacity

class mMTCSlice(BaseSlice):
    def __init__(self):
        super().__init__(name="mMTC", priority=2, min_prbs=5)

    def calculate_metrics(self, ues_data):
        self.traffic_demand = sum([ue['demand'] for ue in ues_data])
        if not ues_data:
            self.throughput = 0
            self.latency = 0
            self.qos_violations = 0
            return
        total_capacity = 0
        prbs_per_ue = self.current_prbs / len(ues_data)
        avg_sinr = 0
        for ue in ues_data:
            sinr = self.calculate_sinr(ue['distance'])
            avg_sinr += sinr
            ue_capacity = (prbs_per_ue * 0.360) * np.log2(1 + sinr)
            total_capacity += ue_capacity 
        self.avg_sinr = avg_sinr / len(ues_data)
        self.throughput = min(self.traffic_demand, total_capacity)
        self.latency = 50.0 + (self.traffic_demand / (total_capacity + 1e-6)) * 50
        self.packet_loss = max(0, (self.traffic_demand/total_capacity) - 0.9) * 0.2
        self.qos_violations = 1 if self.packet_loss > 0.05 else 0
        self.total_capacity = total_capacity
