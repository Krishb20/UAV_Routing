import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GridWorldEnvironment:
    def __init__(self, grid_size, num_service_nodes, num_charging_stations):
        # Initialize the environment
        self.grid_size = grid_size
        self.num_service_nodes = num_service_nodes
        self.num_charging_stations = num_charging_stations
        self.grid = np.zeros(grid_size)  # Create an empty grid
        self.depot = (0, 0)  # Define the depot location
        # self.service_nodes_count = num_service_nodes

        # Initialize the charging stations randomly
        self.charging_stations = []
        while len(self.charging_stations) < num_charging_stations:
          node = (np.random.randint(grid_size[0]), np.random.randint(grid_size[1]))
          # Check if the node is distinct from charging stations
          if node != self.depot and node not in self.charging_stations:
              self.charging_stations.append(node)

        self.service_nodes = []

        while len(self.service_nodes) < num_service_nodes:
            node = (np.random.randint(grid_size[0]), np.random.randint(grid_size[1]))
            # Check if the node is distinct from charging stations
            if node != self.depot and node not in self.charging_stations and node not in self.service_nodes:
                self.service_nodes.append(node)

        self.copy_service_nodes = []

        # Define other environment parameters (battery, rewards, etc.)
        self.battery = 100  # Initial battery level
        self.reward = 0  # Initialize the reward
        self.uav_position = self.depot  # Initialize UAV position at the depot

    def reset(self):
        # Reset the environment to the initial state
        self.grid = np.zeros(self.grid_size)

        # Define other environment parameters (battery, rewards, etc.)
        self.battery = 100  # Initial battery level
        self.reward = 0  # Initialize the reward
        self.uav_position = self.depot  # Initialize UAV position at the depot

        for node in self.service_nodes:
            self.copy_service_nodes.append(node)

        return self.uav_position  # Return the depot as the initial state

    def move_uav(self, action):
        # Implement the UAV's movement based on the chosen action
        # Action: 0 - Move Up, 1 - Move Down, 2 - Move Left, 3 - Move Right

        if action == 0:  # Move Up
            new_position = (self.uav_position[0] - 1, self.uav_position[1])
        elif action == 1:  # Move Down
            new_position = (self.uav_position[0] + 1, self.uav_position[1])
        elif action == 2:  # Move Left
            new_position = (self.uav_position[0], self.uav_position[1] - 1)
        elif action == 3:  # Move Right
            new_position = (self.uav_position[0], self.uav_position[1] + 1)

        # Check if the new position is within the grid boundaries
        if 0 <= new_position[0] < self.grid_size[0] and 0 <= new_position[1] < self.grid_size[1]:
            # Update the UAV's position
            self.uav_position = new_position
            self.reward -= 5
            return True  # Movement successful

        self.reward -= 10 #If UAV is going out of boundary
        return False  # Movement failed (out of bounds)

    def visit_service_node(self, node_position):
        # Implement the logic when the UAV visits a service node
        # node_position: (x, y) coordinates of the visited service node

        # Check if the visited node is a valid service node
        if node_position in self.copy_service_nodes:
            # Update the reward (positive reward for visiting a service node)
            self.reward += 25  # You can customize the reward value

            # Mark the node as visited
            self.copy_service_nodes.remove(node_position)

            # Check if all service nodes are visited
            if not self.copy_service_nodes:
                self.reward += 100  # Additional reward for visiting all nodes

            return True  # Visiting successful

        return False  # Visiting failed (invalid node)

    def visit_charging_station(self, station_position):
        # Implement the logic when the UAV visits a charging station
        # station_position: (x, y) coordinates of the visited charging station

        # Check if the visited station is a valid charging station
        if station_position in self.charging_stations:
            # Recharge the battery
            self.battery = 90  # Fully charge the battery

            # Update the reward (positive reward for visiting a charging station)
            self.reward += 25  # You can customize the reward value

            return True  # Visiting successful

        return False  # Visiting failed (invalid station)

    def is_all_nodes_visited(self):
        # Check if all service nodes are visited
        return len(self.copy_service_nodes) == 0

    def step(self, action):
        # Take a step in the environment based on the chosen action
        # Action: 0 - Move Up, 1 - Move Down, 2 - Move Left, 3 - Move Right

        # Define constants for battery consumption
        MOVE_BATTERY_COST = 10
        VISIT_NODE_BATTERY_COST = 20

        # Check if the action is valid and perform the corresponding action
        if 0 <= action < 4:
            # Check if all service nodes are visited
            if self.is_all_nodes_visited():
                self.reward += 100  # Additional reward for completing the task
                return self.uav_position, self.reward, True  # Set done to True

            # Perform the UAV's movement
            if self.move_uav(action):
                # Consume battery for movement
                self.battery -= MOVE_BATTERY_COST

            # Check if the UAV is running out of battery
            if self.battery <= 0:
                # Find the nearest charging station and go there
                nearest_station = min(self.charging_stations, key=lambda s: abs(s[0] - self.uav_position[0]) + abs(s[1] - self.uav_position[1]))
                self.visit_charging_station(nearest_station)

            # Check if the UAV is not running out of battery but still visiting charging station or visiting already visited service node then impose penalty
            if (self.uav_position in self.charging_stations and self.battery > 0) or (self.uav_position in self.service_nodes and self.uav_position not in self.copy_service_nodes):
                self.reward -= 10

            # Check if the UAV is at a service node
            if self.uav_position in self.copy_service_nodes:
                # Perform the service node visit action
                if self.visit_service_node(self.uav_position):
                    # Consume additional battery for visiting a service node
                    self.battery -= VISIT_NODE_BATTERY_COST

            # Return the new state, reward, and done flag
            return self.uav_position, self.reward, False  # Assuming the task is not done yet

        return None  # Invalid action, return None

    def get_flattened_state(self):
        # Flatten the grid and concatenate other state variables
        flattened_grid = self.grid.flatten().tolist()
        flattened_state = flattened_grid + [self.battery]
        return flattened_state