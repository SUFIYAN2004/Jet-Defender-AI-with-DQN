import pygame
import random
import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ==========================================
# 1. THE RL TRAINING ENVIRONMENT
# ==========================================
class JetDefenderTrainEnv:
    def __init__(self):
        pygame.init()
        self.WIDTH, self.HEIGHT = 800, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Jet Defender: AI Training Simulator")
        self.clock = pygame.time.Clock()
        
        self.BLACK = (0, 0, 0)
        self.CYAN = (0, 255, 255)
        self.ORANGE = (255, 165, 0)
        self.RED = (255, 0, 0)
        
        self.GROUND_HEIGHT = 60
        self.protector_width, self.protector_height = 80, 80
        self.jet_width, self.jet_height = 100, 100
        
        self.jet_bullet_speed = 18
        self.MISSILE_COOLDOWN = 20

    def reset(self):
        self.protector_x = self.WIDTH // 2 - self.protector_width // 2
        self.protector_y = self.HEIGHT - self.GROUND_HEIGHT - self.protector_height
        self.base_health = 5
        self.score = 0
        
        self.jet = {
            "x": random.randint(-300, -100),
            "y": random.randint(30, 250),
            "speed": random.uniform(3.5, 5.5),
            "hp": 10
        }
        
        self.player_bullets = []
        self.jet_bullets = []
        self.shoot_cooldown = 0
        
        return self._get_state()

    def _get_state(self):
        closest_bullet_x, closest_bullet_y = -1.0, -1.0
        if self.jet_bullets:
            closest_bullet = min(self.jet_bullets, key=lambda b: math.hypot(b["x"] - self.protector_x, b["y"] - self.protector_y))
            closest_bullet_x = closest_bullet["x"] / self.WIDTH
            closest_bullet_y = closest_bullet["y"] / self.HEIGHT

        state = np.array([
            self.protector_x / self.WIDTH,           
            self.jet["x"] / self.WIDTH,              
            self.jet["y"] / self.HEIGHT,             
            closest_bullet_x,                        
            closest_bullet_y,                        
            1.0 if self.shoot_cooldown > 0 else 0.0  
        ], dtype=np.float32)
        
        return state

    def step(self, action):
        reward = 0.1 
        done = False
        protector_speed = 7

        # HINTS FOR THE AI
        if abs((self.protector_x + self.protector_width/2) - (self.jet["x"] + self.jet_width/2)) < 80:
            reward += 0.2 
            
        for b in self.jet_bullets:
            if abs(b["x"] - (self.protector_x + self.protector_width/2)) < self.protector_width/2 and b["y"] < self.protector_y:
                reward -= 0.5 

        # ACTIONS
        if action == 1 and self.protector_x > 0:
            self.protector_x -= protector_speed
        elif action == 2 and self.protector_x < self.WIDTH - self.protector_width:
            self.protector_x += protector_speed
            
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
            
        if action == 3 and self.shoot_cooldown <= 0:
            self.player_bullets.append({
                "x": self.protector_x + self.protector_width // 2 - 5,
                "y": self.protector_y,
                "vx": 0, "vy": -8, "damage": 3, "width": 10, "height": 10
            })
            self.shoot_cooldown = self.MISSILE_COOLDOWN

        # UPDATE JET
        self.jet["x"] += self.jet["speed"]
        if self.jet["x"] > self.WIDTH:
            self.jet["x"] = random.randint(-300, -100)
            self.jet["y"] = random.randint(30, 250)
            self.jet["hp"] = 10
            
        if random.randint(1, 60) == 1: 
            j_bx = self.jet["x"] + self.jet_width // 2
            j_by = self.jet["y"] + self.jet_height - 20
            dx = (self.protector_x + self.protector_width / 2) - j_bx
            dy = (self.protector_y + self.protector_height / 2) - j_by
            dist = math.hypot(dx, dy)
            if dist > 0:
                j_vx = (dx / dist) * self.jet_bullet_speed
                j_vy = (dy / dist) * self.jet_bullet_speed
            else:
                j_vx, j_vy = 0, self.jet_bullet_speed
            self.jet_bullets.append({"x": j_bx, "y": j_by, "vx": j_vx, "vy": j_vy, "width": 6, "height": 20})

        # UPDATE MISSILES (HOMING)
        for bullet in self.player_bullets[:]:
            dx = (self.jet["x"] + self.jet_width / 2) - bullet["x"]
            dy = (self.jet["y"] + self.jet_height / 2) - bullet["y"]
            dist = math.hypot(dx, dy)
            if dist > 0:
                missile_speed = 12 
                bullet["vx"] = (dx / dist) * missile_speed
                bullet["vy"] = (dy / dist) * missile_speed

            bullet["x"] += bullet["vx"]
            bullet["y"] += bullet["vy"]
            if bullet["y"] < -50 or bullet["y"] > self.HEIGHT or bullet["x"] < -50 or bullet["x"] > self.WIDTH:
                self.player_bullets.remove(bullet)
                
        for bullet in self.jet_bullets[:]:
            bullet["x"] += bullet["vx"]
            bullet["y"] += bullet["vy"]
            if bullet["y"] > self.HEIGHT - self.GROUND_HEIGHT or bullet["x"] < 0 or bullet["x"] > self.WIDTH:
                self.jet_bullets.remove(bullet)

        # INTERCEPTIONS
        for p_bullet in self.player_bullets[:]:
            p_rect = pygame.Rect(p_bullet["x"], p_bullet["y"], p_bullet["width"], p_bullet["height"])
            bullet_destroyed = False
            for j_bullet in self.jet_bullets[:]:
                if p_rect.colliderect(pygame.Rect(j_bullet["x"], j_bullet["y"], j_bullet["width"], j_bullet["height"])):
                    self.jet_bullets.remove(j_bullet)
                    bullet_destroyed = True
                    reward += 5 
                    break 
            if bullet_destroyed:
                self.player_bullets.remove(p_bullet)

        # COLLISIONS
        protector_rect = pygame.Rect(self.protector_x + 10, self.protector_y + 10, self.protector_width - 20, self.protector_height - 20)
        jet_rect = pygame.Rect(self.jet["x"] + 10, self.jet["y"] + 10, self.jet_width - 20, self.jet_height - 20)
        
        for bullet in self.player_bullets[:]:
            if jet_rect.colliderect(pygame.Rect(bullet["x"], bullet["y"], bullet["width"], bullet["height"])):
                self.player_bullets.remove(bullet)
                self.jet["hp"] -= bullet["damage"]
                reward += 15 # Good hit!
                
                if self.jet["hp"] <= 0:
                    self.score += 10
                    reward += 100 
                    self.jet["x"] = random.randint(-300, -100)
                    self.jet["y"] = random.randint(30, 250)
                    self.jet["hp"] = 10 
                    self.jet["speed"] += 0.2
                    
        for bullet in self.jet_bullets[:]:
            if protector_rect.colliderect(pygame.Rect(bullet["x"], bullet["y"], bullet["width"], bullet["height"])):
                self.jet_bullets.remove(bullet)
                self.base_health -= 1 
                reward -= 50 
                if self.base_health <= 0:
                    done = True
                    reward -= 200 

        return self._get_state(), reward, done

    def render(self):
        pygame.event.pump()
        self.screen.fill((20, 20, 40)) 
        pygame.draw.rect(self.screen, (50, 50, 50), (0, self.HEIGHT - self.GROUND_HEIGHT, self.WIDTH, self.GROUND_HEIGHT))

        pygame.draw.rect(self.screen, (50, 200, 50), (self.protector_x, self.protector_y, self.protector_width, self.protector_height))
        pygame.draw.rect(self.screen, (200, 50, 50), (self.jet["x"], self.jet["y"], self.jet_width, self.jet_height))
        
        for bullet in self.player_bullets:
            pygame.draw.rect(self.screen, self.ORANGE, (bullet["x"], bullet["y"], bullet["width"], bullet["height"]))
        for bullet in self.jet_bullets:
            pygame.draw.rect(self.screen, self.RED, (bullet["x"], bullet["y"], bullet["width"], bullet["height"]))

        pygame.display.flip()
        self.clock.tick(60)

# ==========================================
# 2. THE NEURAL NETWORK (BRAIN)
# ==========================================
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self):
        self.state_size = 6  
        self.action_size = 4 # 0:Idle, 1:Left, 2:Right, 3:Shoot
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998 
        self.batch_size = 128
        self.learning_rate = 0.0005
        
        self.policy_net = DQN(self.state_size, self.action_size)
        self.target_net = DQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.update_target_network()
        self.target_update_freq = 100
        self.step_count = 0

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([x[0] for x in batch]))
        actions = torch.LongTensor([x[1] for x in batch]).unsqueeze(1)
        rewards = torch.FloatTensor([x[2] for x in batch]).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([x[3] for x in batch]))
        dones = torch.FloatTensor([x[4] for x in batch]).unsqueeze(1)

        current_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ==========================================
# 3. TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    env = JetDefenderTrainEnv()
    agent = Agent()
    episodes = 2000
    
    best_reward = -float('inf')
    
    print("🚀 Starting AI Training Protocol...")
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            # Render every 25th episode to monitor progress
            if e % 25 == 0:  
                env.render()

        if total_reward > best_reward:
            best_reward = total_reward
            print(f"⭐ NEW BEST! Ep {e+1:4d} | Reward: {total_reward:6.1f} | Epsilon: {agent.epsilon:.3f} | Score: {env.score}")
            # Save the brain!
            torch.save(agent.policy_net.state_dict(), "best_missile_brain.pth")
        elif e % 20 == 0: 
            print(f"Episode {e+1:4d}/{episodes} | Reward: {total_reward:6.1f} | Epsilon: {agent.epsilon:.3f} | Score: {env.score}")
    
    print("🎉 Training complete!")
