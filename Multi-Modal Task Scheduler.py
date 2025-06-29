import numpy as np
import random
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import threading
import time

class MultiModalScheduler:
    def __init__(self):
        self.parallel_scheduler = ParallelScheduler()
        self.batch_scheduler = BatchScheduler()
        self.dqn_scheduler = DQNScheduler(state_size=10, action_size=5)
        self.performance_metrics = {}
        
    def schedule_tasks(self, mds_set):
        # Preprocessing & Validation
        validated_set = self.preprocess_and_validate(mds_set)
        
        # Scheduler Mode Selection
        mode = self.select_scheduler_mode(validated_set)
        
        if mode == "P-MDS":
            scheduled_tasks = self.parallel_scheduler.schedule(validated_set)
        elif mode == "B-MDS":
            scheduled_tasks = self.batch_scheduler.schedule(validated_set)
        elif mode == "DQN":
            scheduled_tasks = self.dqn_scheduler.schedule(validated_set)
        
        # Task Distribution Optimization
        optimized_tasks = self.optimize_task_distribution(scheduled_tasks)
        
        # Route Path Selection
        route_paths = self.select_route_paths(optimized_tasks)
        
        # Generate Final Strategy
        strategy = self.generate_final_strategy(route_paths)
        
        # Feedback Integration
        self.integrate_feedback(strategy)
        
        return strategy
    
    def preprocess_and_validate(self, mds_set):
        """Preprocess and validate incoming MDS set"""
        validated_set = []
        for task in mds_set:
            if self.validate_task(task):
                processed_task = self.preprocess_task(task)
                validated_set.append(processed_task)
        return validated_set
    
    def validate_task(self, task):
        """Validate individual task parameters"""
        required_fields = ['id', 'priority', 'resource_requirements', 'deadline']
        return all(field in task for field in required_fields)
    
    def preprocess_task(self, task):
        """Normalize and preprocess task parameters"""
        task['normalized_priority'] = task['priority'] / 10.0
        task['urgency_score'] = self.calculate_urgency(task)
        return task
    
    def calculate_urgency(self, task):
        """Calculate urgency score based on deadline and current time"""
        current_time = time.time()
        time_to_deadline = task['deadline'] - current_time
        return max(0, 1.0 - (time_to_deadline / 3600))  # Normalize to 1 hour
    
    def select_scheduler_mode(self, validated_set):
        """Select appropriate scheduler mode based on task characteristics"""
        total_tasks = len(validated_set)
        avg_urgency = np.mean([task['urgency_score'] for task in validated_set])
        resource_intensity = np.mean([sum(task['resource_requirements'].values()) 
                                    for task in validated_set])
        
        # Decision logic based on task characteristics
        if avg_urgency > 0.8 and resource_intensity < 0.5:
            return "P-MDS"  # High urgency, low resource intensity
        elif total_tasks > 50 and avg_urgency < 0.3:
            return "B-MDS"  # Large batch, low urgency
        else:
            return "DQN"    # Complex optimization required
    
    def optimize_task_distribution(self, scheduled_tasks):
        """Optimize task distribution across available resources"""
        optimized_tasks = []
        
        # Load balancing algorithm
        resource_loads = {}
        for task in scheduled_tasks:
            best_resource = self.find_best_resource(task, resource_loads)
            task['assigned_resource'] = best_resource
            
            # Update resource load
            if best_resource not in resource_loads:
                resource_loads[best_resource] = 0
            resource_loads[best_resource] += task['resource_requirements'].get('cpu', 0)
            
            optimized_tasks.append(task)
        
        return optimized_tasks
    
    def find_best_resource(self, task, resource_loads):
        """Find the best resource for task assignment"""
        available_resources = ['resource_1', 'resource_2', 'resource_3', 'resource_4']
        min_load = float('inf')
        best_resource = available_resources[0]
        
        for resource in available_resources:
            current_load = resource_loads.get(resource, 0)
            if current_load < min_load:
                min_load = current_load
                best_resource = resource
        
        return best_resource
    
    def select_route_paths(self, optimized_tasks):
        """Select optimal routing paths for task execution"""
        route_paths = {}
        
        for task in optimized_tasks:
            # Graph-based path selection algorithm
            source = task.get('source_node', 'default_source')
            destination = task['assigned_resource']
            
            # Simplified shortest path selection
            path = self.calculate_shortest_path(source, destination)
            route_paths[task['id']] = path
        
        return route_paths
    
    def calculate_shortest_path(self, source, destination):
        """Calculate shortest path between source and destination"""
        # Simplified path calculation (in real implementation, use Dijkstra's algorithm)
        return [source, f"intermediate_{source}_{destination}", destination]
    
    def generate_final_strategy(self, route_paths):
        """Generate final clustered and balanced scheduling strategy"""
        strategy = {
            'execution_plan': [],
            'resource_allocation': {},
            'timing_constraints': {},
            'routing_information': route_paths
        }
        
        # Group tasks by resource clusters
        resource_clusters = {}
        for task_id, path in route_paths.items():
            destination = path[-1]
            if destination not in resource_clusters:
                resource_clusters[destination] = []
            resource_clusters[destination].append(task_id)
        
        strategy['resource_clusters'] = resource_clusters
        return strategy
    
    def integrate_feedback(self, strategy):
        """Integrate feedback for continuous improvement"""
        # Store performance metrics
        self.performance_metrics[time.time()] = {
            'strategy_id': id(strategy),
            'resource_utilization': self.calculate_resource_utilization(strategy),
            'completion_rate': 0.0  # Will be updated after execution
        }

class DQNScheduler:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.epochs = 100
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        """Build deep neural network model for Q-learning"""
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update target model with current model weights"""
        self.target_model.set_weights(self.model.get_weights())
        
    def schedule(self, tasks):
        """Main scheduling method using DQN algorithm"""
        scheduled_tasks = []
        
        for epoch in range(self.epochs):
            state = self.get_state(tasks)
            total_reward = 0
            
            for task in tasks:
                # Select action based on epsilon-greedy strategy
                if np.random.random() <= self.epsilon:
                    action = random.randrange(self.action_size)
                else:
                    q_values = self.model.predict(state.reshape(1, -1), verbose=0)
                    action = np.argmax(q_values[0])
                
                # Execute action and get performance parameters
                reward, next_state = self.execute_action(task, action)
                total_reward += reward
                
                # Store experience in replay buffer
                self.memory.append((state, action, reward, next_state, False))
                
                # Train the model if enough experiences are collected
                if len(self.memory) > 32:
                    self.replay(32)
                    
                state = next_state
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Update target model periodically
            if epoch % 10 == 0:
                self.update_target_model()
        
        # Generate final scheduled tasks based on learned policy
        scheduled_tasks = self.generate_optimized_schedule(tasks)
        return scheduled_tasks
    
    def get_state(self, tasks=None):
        """Get current state representation"""
        if tasks is None:
            # Default state
            return np.random.random(self.state_size)
        
        # Extract state features from tasks
        state_features = []
        
        # Feature 1-3: Resource utilization metrics
        cpu_util = np.mean([task.get('resource_requirements', {}).get('cpu', 0) for task in tasks[:3]])
        memory_util = np.mean([task.get('resource_requirements', {}).get('memory', 0) for task in tasks[:3]])
        network_util = np.mean([task.get('resource_requirements', {}).get('network', 0) for task in tasks[:3]])
        
        state_features.extend([cpu_util, memory_util, network_util])
        
        # Feature 4-6: Priority and urgency metrics
        avg_priority = np.mean([task.get('normalized_priority', 0.5) for task in tasks])
        max_urgency = max([task.get('urgency_score', 0) for task in tasks] + [0])
        pending_tasks = len(tasks)
        
        state_features.extend([avg_priority, max_urgency, pending_tasks / 100.0])
        
        # Feature 7-10: System performance metrics
        current_time = time.time() % 3600 / 3600.0  # Normalized hour
        system_load = random.random()  # Placeholder for actual system load
        network_latency = random.random()  # Placeholder for network latency
        queue_length = len(tasks) / 50.0  # Normalized queue length
        
        state_features.extend([current_time, system_load, network_latency, queue_length])
        
        return np.array(state_features[:self.state_size])
    
    def execute_action(self, task, action):
        """Execute scheduling action and return reward and next state"""
        # Action mapping
        actions = ['high_priority', 'medium_priority', 'low_priority', 'batch_process', 'defer']
        selected_action = actions[action % len(actions)]
        
        # Calculate reward based on action effectiveness
        reward = self.calculate_reward(task, selected_action)
        
        # Get next state
        next_state = self.get_state()
        
        return reward, next_state
    
    def calculate_reward(self, task, action):
        """Calculate reward for the given action"""
        base_reward = 0
        
        # Reward based on priority alignment
        if action == 'high_priority' and task.get('urgency_score', 0) > 0.7:
            base_reward += 10
        elif action == 'medium_priority' and 0.3 <= task.get('urgency_score', 0) <= 0.7:
            base_reward += 8
        elif action == 'low_priority' and task.get('urgency_score', 0) < 0.3:
            base_reward += 6
        elif action == 'batch_process':
            base_reward += 5
        elif action == 'defer':
            base_reward += 2
        else:
            base_reward -= 5  # Penalty for mismatched priority
        
        # Additional rewards for resource efficiency
        resource_efficiency = 1.0 - sum(task.get('resource_requirements', {}).values()) / 3.0
        base_reward += resource_efficiency * 5
        
        return base_reward
    
    def replay(self, batch_size):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])
        
        # Predict Q-values for current states
        current_q_values = self.model.predict(states, verbose=0)
        
        # Predict Q-values for next states using target model
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values
        for i in range(batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + 0.95 * np.max(next_q_values[i])
        
        # Train the model
        self.model.fit(states, current_q_values, epochs=1, verbose=0)
    
    def generate_optimized_schedule(self, tasks):
        """Generate final optimized schedule using learned policy"""
        scheduled_tasks = []
        
        for task in tasks:
            state = self.get_state([task])
            q_values = self.model.predict(state.reshape(1, -1), verbose=0)
            best_action = np.argmax(q_values[0])
            
            # Apply the best action to the task
            actions = ['high_priority', 'medium_priority', 'low_priority', 'batch_process', 'defer']
            task['scheduled_action'] = actions[best_action % len(actions)]
            task['scheduling_score'] = np.max(q_values[0])
            
            scheduled_tasks.append(task)
        
        # Sort by scheduling score for execution order
        scheduled_tasks.sort(key=lambda x: x['scheduling_score'], reverse=True)
        
        return scheduled_tasks

# Supporting scheduler classes
class ParallelScheduler:
    def schedule(self, tasks):
        """Parallel scheduling implementation"""
        # Sort by urgency for parallel execution
        tasks.sort(key=lambda x: x.get('urgency_score', 0), reverse=True)
        
        # Assign parallel execution slots
        for i, task in enumerate(tasks):
            task['execution_slot'] = i % 4  # Assume 4 parallel slots
            task['scheduling_method'] = 'parallel'
        
        return tasks

class BatchScheduler:
    def schedule(self, tasks):
        """Batch scheduling implementation"""
        # Group tasks by resource requirements
        batches = {}
        
        for task in tasks:
            resource_key = str(sorted(task.get('resource_requirements', {}).items()))
            if resource_key not in batches:
                batches[resource_key] = []
            batches[resource_key].append(task)
        
        # Schedule batches
        scheduled_tasks = []
        batch_id = 0
        for resource_key, batch_tasks in batches.items():
            for task in batch_tasks:
                task['batch_id'] = batch_id
                task['scheduling_method'] = 'batch'
                scheduled_tasks.append(task)
            batch_id += 1
        
        return scheduled_tasks

# Example usage
if __name__ == "__main__":
    # Initialize scheduler
    scheduler = MultiModalScheduler()
    
    # Sample MDS set
    sample_tasks = [
        {
            'id': 'task_1',
            'priority': 8,
            'resource_requirements': {'cpu': 0.6, 'memory': 0.4, 'network': 0.2},
            'deadline': time.time() + 1800,  # 30 minutes from now
            'source_node': 'node_A'
        },
        {
            'id': 'task_2',
            'priority': 5,
            'resource_requirements': {'cpu': 0.3, 'memory': 0.7, 'network': 0.1},
            'deadline': time.time() + 3600,  # 1 hour from now
            'source_node': 'node_B'
        }
    ]
    
    # Schedule tasks
    result = scheduler.schedule_tasks(sample_tasks)
    print("Scheduling completed:", result)
