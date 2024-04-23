import numpy as np
import pandapower as pp
import pandas as pd
import pickle
import matplotlib.pyplot as plt

class PowerGridSimulation:
    def __init__(self):
        self.hours_per_year = 87
        self.S_max_G = [1.0, 1.0]  # Max power capacity for static generators
        self.E_max_batt = [0.5, 0.5]  # Max energy capacity for batteries
        self.SoC_initial = [0.5, 0.5]  # Initial State of Charge for batteries
    
    def calculate_S_G(self, S_max_G, size=1):
        # Generate S_G using a uniform distribution between 0 and S_max_G
        return np.random.uniform(0, S_max_G, size=size)
    
    @staticmethod
    def compute_Q_G(P_G, S_G, sigma):
        # Correct to handle multiple values
        return sigma * np.sqrt(np.maximum(S_G**2 - P_G**2, 0))
    
    def simulate_penalty_loss_vs_deviation(self, simulation_data):
        # Prepare the simulation data
        P_G_array = np.array(simulation_data['P_G'].tolist())
        Q_G_array = np.array(simulation_data['Q_G'].tolist())
        S_G_squared = np.array(self.S_max_G)**2
        sigma_array = np.abs(S_G_squared - P_G_array**2 - Q_G_array**2)
        plt.figure(figsize=(10, 6))
        a_value = 1
        b_values = [1, 2, ]
        for b in b_values:
            penalty_losses = a_value * np.exp(-b / np.maximum(sigma_array, 1e-4))
            sigma_vals = np.insert(sigma_array.flatten(), 0, 0)
            penalty_losses_vals = np.insert(penalty_losses.flatten(), 0, a_value)
            plt.plot(sigma_vals, penalty_losses_vals, label=f'b = {b}')
        plt.xlabel('Sigma')
        plt.ylabel('Penalty Loss')
        plt.title('Penalty Loss vs. Sigma for Different b Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    def initialize_network(self):
        net = pp.create_empty_network()
        bus1 = pp.create_bus(net, vn_kv=20.0)
        bus2 = pp.create_bus(net, vn_kv=20.0)
        bus3 = pp.create_bus(net, vn_kv=20.0)
        pp.create_ext_grid(net, bus1)
        pp.create_line_from_parameters(net, bus1, bus2, length_km=1, r_ohm_per_km=0.05, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        pp.create_line_from_parameters(net, bus1, bus3, length_km=1, r_ohm_per_km=0.05, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        pp.create_line_from_parameters(net, bus2, bus3, length_km=1, r_ohm_per_km=0.05, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        return net, [bus1, bus2, bus3]

    def run_simulation(self, net, bus_ids, P_L, Q_L, E_max_batt, SoC_initial):
        bus1, bus2, bus3 = bus_ids
        P_G = self.calculate_S_G(np.array(self.S_max_G), size=2)
        sigma = np.random.choice([-1, 1], size=2)
        Q_G = self.compute_Q_G(P_G, P_G, sigma)
        pp.create_sgen(net, bus2, p_mw=P_G[0], q_mvar=Q_G[0])
        pp.create_sgen(net, bus3, p_mw=P_G[1], q_mvar=Q_G[1])
        P_batt2 = np.random.uniform(-E_max_batt[0], E_max_batt[0])
        P_batt3 = np.random.uniform(-E_max_batt[1], E_max_batt[1])
        pp.create_storage(net, bus2, p_mw=P_batt2, max_e_mwh=E_max_batt[0], soc_percent=SoC_initial[0] * 100)
        pp.create_storage(net, bus3, p_mw=P_batt3, max_e_mwh=E_max_batt[1], soc_percent=SoC_initial[1] * 100)
        pp.create_load(net, bus1, p_mw=P_L[0], q_mvar=Q_L[0])
        pp.create_load(net, bus2, p_mw=P_L[1], q_mvar=Q_L[1])
        pp.create_load(net, bus3, p_mw=P_L[2], q_mvar=Q_L[2])
        pp.runpp(net)
        P_line = net.res_line['p_from_mw'].tolist()
        Q_line = net.res_line['q_from_mvar'].tolist()
        V = net.res_bus['vm_pu'].tolist()
        P_G_target = np.array([0.8, 0.8])
        Q_G_target = np.array([0, 0])
        delta_P_G = P_G - P_G_target
        delta_Q_G = Q_G - Q_G_target
        penalty_loss_P = np.sum(delta_P_G**2)
        penalty_loss_Q = np.sum(delta_Q_G**2)
        total_penalty_loss = penalty_loss_P + penalty_loss_Q
        results = {
            "P_G": P_G.tolist(),
            "Q_G": Q_G.tolist(),
            "P_batt": [P_batt2, P_batt3],
            "P_L": P_L,
            "Q_L": Q_L,
            "r_line": [0.05, 0.05, 0.05],
            "x_line": [0.1, 0.1, 0.1],
            "P_line": P_line,
            "Q_line": Q_line,
            "V": V,
            "penalty_loss": total_penalty_loss
        }
        return results

    def simulate_one_year(self):
        data = []
        for _ in range(self.hours_per_year):
            net, bus_ids = self.initialize_network()
            P_L = np.random.uniform(0.1, 0.5, size=3)
            Q_L = np.random.uniform(0.1, 0.5, size=3)
            results = self.run_simulation(net, bus_ids, P_L, Q_L, self.E_max_batt, self.SoC_initial)
            data.append(results)
        with open('simulation_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        return pd.DataFrame(data)
