from simulation import PowerGridSimulation
from neural_network import NeuralNetworkModel
import numpy as np
import jax.numpy as jnp

def run_simulation_and_training():
    power_grid_sim = PowerGridSimulation()
    simulation_data = power_grid_sim.simulate_one_year()
    nn_model = NeuralNetworkModel([16, 10, 9], 0.001)

    P_G_array = np.array(simulation_data['P_G'].tolist())
    Q_G_array = np.array(simulation_data['Q_G'].tolist())
    P_L_array = np.array(simulation_data['P_L'].tolist())
    Q_L_array = np.array(simulation_data['Q_L'].tolist())
    r_line_array = np.array(simulation_data['r_line'].tolist())
    x_line_array = np.array(simulation_data['x_line'].tolist())

    x_data = np.hstack([P_G_array, Q_G_array, P_L_array, Q_L_array, r_line_array, x_line_array])
    print(f"x_data shape: {x_data.shape}")

    P_line_array = np.array(simulation_data['P_line'].tolist())
    Q_line_array = np.array(simulation_data['Q_line'].tolist())
    V_array = np.array(simulation_data['V'].tolist())

    y_data = np.hstack([P_line_array, Q_line_array, V_array])
    print(f"y_data shape: {y_data.shape}")

    trained_params = nn_model.train_network(jnp.array(x_data, dtype=jnp.float32), jnp.array(y_data, dtype=jnp.float32))
    return trained_params, simulation_data

if __name__ == "__main__":
    trained_nn_params, simulation_data = run_simulation_and_training()
    nn_model = NeuralNetworkModel([16, 10, 9], 0.001)
    sim = PowerGridSimulation()
    sim.simulate_penalty_loss_vs_deviation(simulation_data)
