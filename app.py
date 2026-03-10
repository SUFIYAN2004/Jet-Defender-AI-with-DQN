import pygame
import random
import math
import sys
import os
import numpy as np
import torch
import torch.nn as nn

# --- THE PYINSTALLER RESOURCE FINDER ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
# ---------------------------------------

# ==========================================
# 1. THE GAME ENVIRONMENT (Windowed 800x600)
# ==========================================
class JetDefenderEnv:
    def __init__(self):
        pygame.init()
        self.WIDTH, self.HEIGHT = 800, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Jet Defender AI")
        self.clock = pygame.time.Clock()
        
        # Extended Colors
        self.SKY_BLUE = (135, 206, 235)
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.CYAN = (0, 255, 255)
        self.ORANGE = (255, 165, 0)
        self.RED = (255, 50, 50)
        self.DARK_RED = (150, 0, 0) 
        self.YELLOW = (255, 255, 100) 
        self.GREEN = (50, 200, 50) 
        self.BLUE = (50, 100, 255)
        self.GROUND_COLOR = (30, 30, 35)
        
        self.GROUND_HEIGHT = 60
        self.protector_width, self.protector_height = 80, 80
        self.jet_width, self.jet_height = 100, 100
        
        self.font = pygame.font.SysFont("impact", 28)
        self.title_font = pygame.font.SysFont("impact", 72)
        self.hud_font = pygame.font.SysFont("consolas", 20)
        
        try:
            raw_p = pygame.image.load(resource_path("base.png")).convert_alpha()
            raw_j = pygame.image.load(resource_path("plane.png")).convert_alpha()
            raw_p.set_colorkey(raw_p.get_at((0, 0)))
            raw_j.set_colorkey(raw_j.get_at((0, 0)))
            self.p_img = pygame.transform.scale(raw_p, (self.protector_width, self.protector_height))
            self.j_img = pygame.transform.scale(raw_j, (self.jet_width, self.jet_height))
        except:
            self.p_img = pygame.Surface((self.protector_width, self.protector_height))
            self.p_img.fill(self.GREEN)
            self.j_img = pygame.Surface((self.jet_width, self.jet_height))
            self.j_img.fill(self.RED)

        self.jet_bullet_speed = 18
        self.MISSILE_COOLDOWN = 20
        self.shake = 0
        self.high_score = 0
        
        # 1. Generate Realistic Gradient Sky
        self.bg_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        for y in range(self.HEIGHT):
            r = int(10 + (y / self.HEIGHT) * 30)
            g = int(10 + (y / self.HEIGHT) * 40)
            b = int(40 + (y / self.HEIGHT) * 80)
            pygame.draw.line(self.bg_surface, (r, g, b), (0, y), (self.WIDTH, y))

        # 2. Generate Stars
        self.stars = [{"x": random.randint(0, self.WIDTH), "y": random.randint(0, self.HEIGHT - self.GROUND_HEIGHT), "speed": random.uniform(0.1, 0.3), "size": random.randint(1, 3)} for _ in range(40)]

        # 3. Generate Fluffy Clouds
        self.clouds = []
        for _ in range(8):
            subs = [(random.randint(-30, 30), random.randint(-15, 15), random.randint(20, 50)) for _ in range(5)]
            self.clouds.append({"x": random.randint(0, self.WIDTH), "y": random.randint(50, 350), "speed": random.uniform(0.5, 1.5), "subs": subs})

    def create_jets(self, num_jets):
        return [{"x": random.randint(-600, -100) - (i * 250), 
                 "y": random.randint(30, 250), 
                 "speed": random.uniform(3.5, 5.5), 
                 "hp": 10} for i in range(num_jets)]

    def reset(self):
        self.protector_x = self.WIDTH // 2 - self.protector_width // 2
        self.protector_y = self.HEIGHT - self.GROUND_HEIGHT - self.protector_height
        self.base_health = 10 
        self.score = 0
        self.jets = self.create_jets(2)
        self.player_bullets = []
        self.jet_bullets = []
        self.explosions = []
        self.shoot_cooldown = 0
        self.shake = 0
        return self._get_state()

    def _get_state(self):
        closest_bullet_x, closest_bullet_y = -1.0, -1.0
        if self.jet_bullets:
            closest_bullet = min(self.jet_bullets, key=lambda b: math.hypot(b["x"] - self.protector_x, b["y"] - self.protector_y))
            closest_bullet_x = closest_bullet["x"] / self.WIDTH
            closest_bullet_y = closest_bullet["y"] / self.HEIGHT

        closest_jet = min(self.jets, key=lambda j: math.hypot(j["x"] - self.protector_x, j["y"] - self.protector_y))

        state = np.array([
            self.protector_x / self.WIDTH,           
            closest_jet["x"] / self.WIDTH,     
            closest_jet["y"] / self.HEIGHT,             
            closest_bullet_x,                        
            closest_bullet_y,                        
            1.0 if self.shoot_cooldown > 0 else 0.0  
        ], dtype=np.float32)
        
        return state

    def step(self, action):
        done = False
        protector_speed = 7

        if action == 1 and self.protector_x > 0:
            self.protector_x -= protector_speed
        elif action == 2 and self.protector_x < self.WIDTH - self.protector_width:
            self.protector_x += protector_speed
            
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
            
        if action == 3 and self.shoot_cooldown <= 0:
            self.player_bullets.append({
                "type": "missile",
                "x": self.protector_x + self.protector_width // 2 - 5,
                "y": self.protector_y,
                "vx": 0, "vy": -8, "damage": 3, "width": 10, "height": 10,
                "trail": [] 
            })
            self.shoot_cooldown = self.MISSILE_COOLDOWN

        for star in self.stars:
            star["x"] -= star["speed"]
            if star["x"] < 0: star["x"] = self.WIDTH
            
        for cloud in self.clouds:
            cloud["x"] -= cloud["speed"]
            if cloud["x"] < -150: cloud["x"], cloud["y"] = self.WIDTH + 100, random.randint(50, 350)

        for jet in self.jets:
            jet["x"] += jet["speed"]
            if jet["x"] > self.WIDTH:
                jet["x"] = random.randint(-300, -100)
                jet["y"] = random.randint(30, 250)
                jet["hp"] = 10
                
            if random.randint(1, 60) == 1: 
                j_bx = jet["x"] + self.jet_width // 2
                j_by = jet["y"] + self.jet_height - 20
                dx = (self.protector_x + self.protector_width / 2) - j_bx
                dy = (self.protector_y + self.protector_height / 2) - j_by
                dist = math.hypot(dx, dy)
                if dist > 0:
                    j_vx = (dx / dist) * self.jet_bullet_speed
                    j_vy = (dy / dist) * self.jet_bullet_speed
                else:
                    j_vx, j_vy = 0, self.jet_bullet_speed
                
                self.jet_bullets.append({
                    "x": j_bx, "y": j_by, "vx": j_vx, "vy": j_vy, "width": 6, "height": 20,
                    "trail": [] 
                })

        for bullet in self.player_bullets[:]:
            bullet["trail"].append((bullet["x"] + bullet["width"]//2, bullet["y"] + bullet["height"]//2))
            if len(bullet["trail"]) > 10: 
                bullet["trail"].pop(0)

            if self.jets:
                target_jet = min(self.jets, key=lambda j: math.hypot((j["x"] + self.jet_width/2) - bullet["x"], (j["y"] + self.jet_height/2) - bullet["y"]))
                dx = (target_jet["x"] + self.jet_width / 2) - bullet["x"]
                dy = (target_jet["y"] + self.jet_height / 2) - bullet["y"]
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
            bullet["trail"].append((bullet["x"] + bullet["width"]//2, bullet["y"] + bullet["height"]//2))
            if len(bullet["trail"]) > 6: 
                bullet["trail"].pop(0)

            bullet["x"] += bullet["vx"]
            bullet["y"] += bullet["vy"]
            
            if bullet["x"] < 0 or bullet["x"] > self.WIDTH:
                self.jet_bullets.remove(bullet)
            elif bullet["y"] > self.HEIGHT - self.GROUND_HEIGHT:
                self.explosions.append({"x": bullet["x"], "y": self.HEIGHT - self.GROUND_HEIGHT, "radius": 8, "timer": 15, "color": self.ORANGE})
                self.jet_bullets.remove(bullet)

        for p_bullet in self.player_bullets[:]:
            p_rect = pygame.Rect(p_bullet["x"], p_bullet["y"], p_bullet["width"], p_bullet["height"])
            bullet_destroyed = False
            for j_bullet in self.jet_bullets[:]:
                if p_rect.colliderect(pygame.Rect(j_bullet["x"], j_bullet["y"], j_bullet["width"], j_bullet["height"])):
                    self.jet_bullets.remove(j_bullet)
                    bullet_destroyed = True
                    self.explosions.append({"x": p_bullet["x"], "y": p_bullet["y"], "radius": 15, "timer": 10, "color": self.YELLOW})
                    break 
            if bullet_destroyed:
                self.player_bullets.remove(p_bullet)

        for exp in self.explosions[:]:
            exp["radius"] += 2 
            exp["timer"] -= 1  
            if exp["timer"] <= 0:
                self.explosions.remove(exp)

        protector_rect = pygame.Rect(self.protector_x + 10, self.protector_y + 10, self.protector_width - 20, self.protector_height - 20)
        
        for bullet in self.player_bullets[:]:
            b_rect = pygame.Rect(bullet["x"], bullet["y"], bullet["width"], bullet["height"])
            for jet in self.jets:
                jet_rect = pygame.Rect(jet["x"] + 10, jet["y"] + 10, self.jet_width - 20, self.jet_height - 20)
                
                if jet_rect.colliderect(b_rect):
                    if bullet in self.player_bullets:
                        self.player_bullets.remove(bullet)
                    jet["hp"] -= bullet["damage"]
                    
                    if jet["hp"] <= 0:
                        self.score += 10
                        self.explosions.append({"x": jet["x"] + self.jet_width//2, "y": jet["y"] + self.jet_height//2, "radius": 30, "timer": 20, "color": self.ORANGE})
                        jet["x"] = random.randint(-400, -100)
                        jet["y"] = random.randint(30, 250)
                        jet["hp"] = 10 
                        jet["speed"] += 0.5
                    break
                    
        for bullet in self.jet_bullets[:]:
            if protector_rect.colliderect(pygame.Rect(bullet["x"], bullet["y"], bullet["width"], bullet["height"])):
                self.explosions.append({"x": bullet["x"], "y": bullet["y"], "radius": 25, "timer": 15, "color": self.RED})
                self.jet_bullets.remove(bullet)
                self.base_health -= 1 
                self.shake = 15 # Screen shake
                if self.base_health <= 0:
                    done = True
                    if self.score > self.high_score:
                        self.high_score = self.score

        return self._get_state(), done

    def render(self, mode="AI"):
        sx, sy = 0, 0
        if self.shake > 0:
            sx, sy = random.randint(-8, 8), random.randint(-8, 8)
            self.shake -= 1

        self.screen.fill(self.BLACK) 
        self.screen.blit(self.bg_surface, (sx, sy))
        
        for star in self.stars:
            pygame.draw.circle(self.screen, self.WHITE, (int(star["x"] + sx), int(star["y"] + sy)), star["size"])

        for cloud in self.clouds:
            cloud_surf = pygame.Surface((150, 100), pygame.SRCALPHA)
            for cx, cy, cr in cloud["subs"]:
                pygame.draw.circle(cloud_surf, (255, 255, 255, 60), (75 + cx, 50 + cy), cr)
            self.screen.blit(cloud_surf, (int(cloud["x"]) + sx, int(cloud["y"]) + sy))

        pygame.draw.rect(self.screen, self.GROUND_COLOR, (0 + sx, self.HEIGHT - self.GROUND_HEIGHT + sy, self.WIDTH, self.GROUND_HEIGHT))

        self.screen.blit(self.p_img, (self.protector_x + sx, self.protector_y + sy))
        
        for jet in self.jets:
            self.screen.blit(self.j_img, (jet["x"] + sx, jet["y"] + sy))
            pygame.draw.rect(self.screen, self.RED, (jet["x"] + sx + 10, jet["y"] + sy - 15, 80, 5))
            pygame.draw.rect(self.screen, self.GREEN, (jet["x"] + sx + 10, jet["y"] + sy - 15, 80 * (jet["hp"]/10), 5))
        
        for bullet in self.player_bullets:
            for i, (tx, ty) in enumerate(bullet["trail"]):
                radius = int(4 * (i / len(bullet["trail"]))) + 1
                pygame.draw.circle(self.screen, self.YELLOW, (int(tx) + sx, int(ty) + sy), radius)
            pygame.draw.rect(self.screen, self.ORANGE, (bullet["x"] + sx, bullet["y"] + sy, bullet["width"], bullet["height"]))
            
        for bullet in self.jet_bullets:
            for i, (tx, ty) in enumerate(bullet["trail"]):
                radius = int(3 * (i / len(bullet["trail"]))) + 1
                pygame.draw.circle(self.screen, self.DARK_RED, (int(tx) + sx, int(ty) + sy), radius)
            pygame.draw.rect(self.screen, self.RED, (bullet["x"] + sx, bullet["y"] + sy, bullet["width"], bullet["height"]))

        for exp in self.explosions:
            pygame.draw.circle(self.screen, exp["color"], (int(exp["x"]) + sx, int(exp["y"]) + sy), int(exp["radius"]))
            if exp["color"] == self.ORANGE or exp["color"] == self.RED:
                pygame.draw.circle(self.screen, self.YELLOW, (int(exp["x"]) + sx, int(exp["y"]) + sy), int(exp["radius"] * 0.5))
                pygame.draw.circle(self.screen, self.WHITE, (int(exp["x"]) + sx, int(exp["y"]) + sy), int(exp["radius"] * 0.2))

        score_text = self.font.render(f"SCORE: {self.score}", True, self.WHITE)
        health_text = self.font.render(f"HEALTH: {'|' * self.base_health}", True, self.GREEN if self.base_health > 3 else self.RED)
        
        driver_color = self.CYAN if mode == "AI" else self.ORANGE
        driver_text = self.font.render(f"BASE: {mode}", True, driver_color)
        toggle_text = self.hud_font.render("[PRESS 'TAB' TO SWITCH]", True, self.WHITE)
        
        self.screen.blit(score_text, (20, 20))
        self.screen.blit(health_text, (20, 60))
        self.screen.blit(driver_text, (self.WIDTH - 150, 20))
        self.screen.blit(toggle_text, (self.WIDTH//2 - toggle_text.get_width()//2, 20))
        
        pygame.display.flip()
        self.clock.tick(60)

# ==========================================
# 2. THE NEURAL NETWORK
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

# ==========================================
# 3. STATE MACHINE UI & PLAY LOOP
# ==========================================
if __name__ == "__main__":
    env = JetDefenderEnv()
    brain = DQN(6, 4) 
    
    ai_available = False
    try:
        brain.load_state_dict(torch.load(resource_path("best_missile_brain.pth"), weights_only=True))
        ai_available = True
    except FileNotFoundError:
        pass

    game_state = "MENU"
    active_mode = "AI" 
    state = None
    
    btn_w, btn_h = 260, 60
    btn_human_rect = pygame.Rect(env.WIDTH//2 - btn_w//2, env.HEIGHT//2 - 20, btn_w, btn_h)
    btn_ai_rect = pygame.Rect(env.WIDTH//2 - btn_w//2, env.HEIGHT//2 + 60, btn_w, btn_h)

    while True:
        if game_state == "MENU":
            env.screen.blit(env.bg_surface, (0, 0))
            for star in env.stars:
                star["x"] -= star["speed"] * 0.5
                if star["x"] < 0: star["x"] = env.WIDTH
                pygame.draw.circle(env.screen, env.WHITE, (int(star["x"]), int(star["y"])), star["size"])
                
            for cloud in env.clouds:
                cloud["x"] -= cloud["speed"] * 0.3
                if cloud["x"] < -150: cloud["x"], cloud["y"] = env.WIDTH + 100, random.randint(50, 350)
                cloud_surf = pygame.Surface((150, 100), pygame.SRCALPHA)
                for cx, cy, cr in cloud["subs"]:
                    pygame.draw.circle(cloud_surf, (255, 255, 255, 60), (75 + cx, 50 + cy), cr)
                env.screen.blit(cloud_surf, (int(cloud["x"]), int(cloud["y"])))
            
            title = env.title_font.render("JET DEFENDER", True, env.WHITE)
            env.screen.blit(title, (env.WIDTH//2 - title.get_width()//2, 120))
            
            sub = env.hud_font.render("REALISTIC COMBAT SIMULATOR", True, env.CYAN)
            env.screen.blit(sub, (env.WIDTH//2 - sub.get_width()//2, 200))
            
            if env.high_score > 0:
                hs_text = env.font.render(f"Session High Score: {env.high_score}", True, env.YELLOW)
                env.screen.blit(hs_text, (env.WIDTH//2 - hs_text.get_width()//2, 250))

            pygame.draw.rect(env.screen, env.BLUE, btn_human_rect, border_radius=10)
            pygame.draw.rect(env.screen, env.BLACK, btn_human_rect, 3, border_radius=10)
            h_txt = env.font.render("PLAY AS HUMAN", True, env.WHITE)
            env.screen.blit(h_txt, (btn_human_rect.centerx - h_txt.get_width()//2, btn_human_rect.centery - h_txt.get_height()//2))

            ai_color = env.GREEN if ai_available else (100, 100, 100)
            pygame.draw.rect(env.screen, ai_color, btn_ai_rect, border_radius=10)
            pygame.draw.rect(env.screen, env.BLACK, btn_ai_rect, 3, border_radius=10)
            ai_txt = env.font.render("WATCH AI PLAY", True, env.BLACK if ai_available else env.WHITE)
            env.screen.blit(ai_txt, (btn_ai_rect.centerx - ai_txt.get_width()//2, btn_ai_rect.centery - ai_txt.get_height()//2))
            
            pygame.display.flip()
            env.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_human_rect.collidepoint(event.pos):
                        active_mode = "HUMAN"
                        state = env.reset()
                        game_state = "PLAYING"
                    elif btn_ai_rect.collidepoint(event.pos) and ai_available:
                        active_mode = "AI"
                        state = env.reset()
                        game_state = "PLAYING"

        elif game_state == "PLAYING":
            action = 0 
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_TAB and ai_available:
                        active_mode = "AI" if active_mode == "HUMAN" else "HUMAN"

            if active_mode == "AI" and ai_available:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = brain(state_tensor)
                action = torch.argmax(q_values).item()
                if env.shoot_cooldown <= 0: action = 3 
                
            elif active_mode == "HUMAN":
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]: action = 1
                elif keys[pygame.K_RIGHT]: action = 2
                if keys[pygame.K_SPACE] and env.shoot_cooldown <= 0: action = 3

            state, done = env.step(action)
            env.render(mode=active_mode)
            
            if done:
                game_state = "GAME_OVER"

        elif game_state == "GAME_OVER":
            overlay = pygame.Surface((env.WIDTH, env.HEIGHT))
            overlay.set_alpha(150)
            overlay.fill(env.BLACK)
            env.screen.blit(overlay, (0, 0))

            go_text = env.title_font.render("MISSION FAILED", True, env.RED)
            env.screen.blit(go_text, (env.WIDTH//2 - go_text.get_width()//2, 120))
            
            score_text = env.font.render(f"Final Score: {env.score}", True, env.WHITE)
            env.screen.blit(score_text, (env.WIDTH//2 - score_text.get_width()//2, 200))
            
            pygame.draw.rect(env.screen, env.ORANGE, btn_human_rect, border_radius=10)
            pygame.draw.rect(env.screen, env.WHITE, btn_human_rect, 3, border_radius=10)
            btn_txt = env.font.render("MAIN MENU", True, env.BLACK)
            env.screen.blit(btn_txt, (btn_human_rect.centerx - btn_txt.get_width()//2, btn_human_rect.centery - btn_txt.get_height()//2))
            
            pygame.display.flip()
            env.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_human_rect.collidepoint(event.pos):
                        game_state = "MENU"
