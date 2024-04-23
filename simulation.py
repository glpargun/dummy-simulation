import numpy as np
import pandapower as pp
import pandas as pd
import pickle
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.example_libraries import optimizers
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
    
    def simulate_penalty_loss_vs_deviation(self, trained_nn_params, nn_model, simulation_data):
        # Prepare the simulation data
        P_G_array = np.array(simulation_data['P_G'].tolist())
        Q_G_array = np.array(simulation_data['Q_G'].tolist())
        
        # No need to replicate S_max_G, directly use it as it matches P_G and Q_G's second dimension
        S_G_squared = np.array(self.S_max_G)**2

        # Calculate sigma for each generator
        sigma_array = np.abs(S_G_squared - P_G_array**2 - Q_G_array**2)

        # Set up the plot
        plt.figure(figsize=(10, 6))

        # Calculate and plot penalty loss for various values of b
        a_value = 1  # Assuming a constant multiplier a
        b_values = [0.1, 0.05, 0.01]

        for b in b_values:
            # Compute penalty losses for each non-zero sigma
            penalty_losses = a_value * np.exp(-b / np.maximum(sigma_array, 1e-4))

            # Prepend the initial sigma value and penalty loss value
            sigma_vals = np.insert(sigma_array.flatten(), 0, 0)
            penalty_losses_vals = np.insert(penalty_losses.flatten(), 0, a_value)

            plt.plot(sigma_vals, penalty_losses_vals, label=f'b = {b}')

        plt.xlabel('Sigma')
        plt.ylabel('Penalty Loss')
        plt.title('Penalty Loss vs. Sigma for Different b Values')
        plt.legend()
        plt.grid(True)
        plt.show()

    
    def plot_penalty_loss_vs_deviation(self, b_values, losses):
        plt.figure(figsize=(10, 6))
        plt.plot(b_values, losses, marker='o')
        plt.xlabel('b values')
        plt.ylabel('Average Penalty Loss')
        plt.title('Penalty Loss vs Deviation for different b')
        plt.grid(True)
        plt.show()

    def initialize_network(self):
        net = pp.create_empty_network()
        bus1 = pp.create_bus(net, vn_kv=20.0)
        bus2 = pp.create_bus(net, vn_kv=20.0)
        bus3 = pp.create_bus(net, vn_kv=20.0)
        pp.create_ext_grid(net, bus1)  # Adding an external grid to bus 1 as the slack bus
        pp.create_line_from_parameters(net, bus1, bus2, length_km=1, r_ohm_per_km=0.05, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        pp.create_line_from_parameters(net, bus1, bus3, length_km=1, r_ohm_per_km=0.05, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        pp.create_line_from_parameters(net, bus2, bus3, length_km=1, r_ohm_per_km=0.05, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
        return net, [bus1, bus2, bus3]

    def run_simulation(self, net, bus_ids, P_L, Q_L, E_max_batt, SoC_initial):
        bus1, bus2, bus3 = bus_ids

        # Adjusted to handle sizes correctly
        P_G = self.calculate_S_G(np.array(self.S_max_G), size=2)  # For two generators
        sigma = np.random.choice([-1, 1], size=2)
        Q_G = self.compute_Q_G(P_G, P_G, sigma)  # Passing P_G as S_G for simplicity here

        pp.create_sgen(net, bus2, p_mw=P_G[0], q_mvar=Q_G[0])
        pp.create_sgen(net, bus3, p_mw=P_G[1], q_mvar=Q_G[1])

        # Battery storage logic
        P_batt2 = np.random.uniform(-E_max_batt[0], E_max_batt[0])
        P_batt3 = np.random.uniform(-E_max_batt[1], E_max_batt[1])
        pp.create_storage(net, bus2, p_mw=P_batt2, max_e_mwh=E_max_batt[0], soc_percent=SoC_initial[0] * 100)
        pp.create_storage(net, bus3, p_mw=P_batt3, max_e_mwh=E_max_batt[1], soc_percent=SoC_initial[1] * 100)

        pp.create_load(net, bus1, p_mw=P_L[0], q_mvar=Q_L[0])
        pp.create_load(net, bus2, p_mw=P_L[1], q_mvar=Q_L[1])
        pp.create_load(net, bus3, p_mw=P_L[2], q_mvar=Q_L[2])

        pp.runpp(net)

        # Extracting results and calculating penalties
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

class NeuralNetworkModel:
    def __init__(self, layer_sizes, step_size=0.01):
        self.step_size = step_size
        self.params = self.init_network_params(layer_sizes)
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(self.step_size)
        self.opt_state = self.opt_init(self.params)
        self.loss_grad = jit(grad(self.loss))

    def penalty_loss(self, params, x, y, b):
        # Implement penalty loss calculation using JAX
        y_pred = self.predict(params, x)
        sigma = np.abs(y_pred - y)
        return np.exp(-b / sigma)

    def init_network_params(self, layer_sizes, rng_key=jax.random.PRNGKey(0)):
        keys = jax.random.split(rng_key, len(layer_sizes) - 1)
        params = []
        for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
            w = jax.random.normal(k, (m, n)) * jnp.sqrt(2.0 / m)
            b = jnp.zeros(n)
            params.append((w, b))
        return params

    def predict(self, params, x):
        activations = x
        for w, b in params[:-1]:  # All but the last layer
            activations = jnp.tanh(jnp.dot(activations, w) + b)
        final_w, final_b = params[-1]
        return jnp.dot(activations, final_w) + final_b

    def loss(self, params, x, y):
        pred = self.predict(params, x)
        return jnp.mean((pred - y) ** 2)

    def update(self, epoch, opt_state, x, y):
        params = self.get_params(opt_state)
        gradients = self.loss_grad(params, x, y)
        return self.opt_update(epoch, gradients, opt_state)

    def train_network(self, x, y, epochs=1000):
        for epoch in range(epochs):
            self.opt_state = self.update(epoch, self.opt_state, x, y)
            if epoch % 10 == 0:
                params = self.get_params(self.opt_state)
                print(f"Epoch {epoch}: loss {self.loss(params, x, y)}")
        return self.get_params(self.opt_state)

# Function to run the entire simulation and training process
def run_simulation_and_training():
    # Instantiate the simulation and run it
    power_grid_sim = PowerGridSimulation()
    simulation_data = power_grid_sim.simulate_one_year()

    P_G_array = np.array(simulation_data['P_G'].tolist())
    Q_G_array = np.array(simulation_data['Q_G'].tolist())
    P_L_array = np.array(simulation_data['P_L'].tolist())
    Q_L_array = np.array(simulation_data['Q_L'].tolist())
    r_line_array = np.array(simulation_data['r_line'].tolist())
    x_line_array = np.array(simulation_data['x_line'].tolist())

    x_data = np.hstack([P_G_array, Q_G_array, P_L_array, Q_L_array, r_line_array, x_line_array])
    print(f"x_data shape: {x_data.shape} \n")

    P_line_array = np.array(simulation_data['P_line'].tolist())
    Q_line_array = np.array(simulation_data['Q_line'].tolist())
    V_array = np.array(simulation_data['V'].tolist())

    y_data = np.hstack([P_line_array, Q_line_array, V_array])
    print(f"y_data shape: {y_data.shape}")

    nn_model = NeuralNetworkModel([16, 10, 9], 0.001)
    trained_params = nn_model.train_network(jnp.array(x_data, dtype=jnp.float32), jnp.array(y_data, dtype=jnp.float32))
    return trained_params, simulation_data

if __name__ == "__main__":
    sim = PowerGridSimulation()
    trained_nn_params, simulation_data = run_simulation_and_training()
    nn_model = NeuralNetworkModel([16, 10, 9], 0.001)
    sim.simulate_penalty_loss_vs_deviation(trained_nn_params, nn_model, simulation_data)