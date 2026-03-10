```markdown
# 🚀 Jet Defender AI with DQN

A high-speed arcade game built in Python where a custom **Deep Q-Network (DQN)** Artificial Intelligence learns to defend a ground base from incoming fighter jets using homing missiles. 

The AI was trained entirely from scratch using PyTorch to calculate spatial geometry, dodge incoming high-speed lasers, and optimize its weapon cooldowns.

## 🎮 Features
* **Custom Deep RL Agent:** A DQN neural network trained over thousands of episodes to master the game mechanics.
* **Seamless Mode Switching:** Press `TAB` mid-game to instantly swap between playing the game yourself and watching the AI take over.
* **Smart Homing Missiles:** Physics-based projectile tracking that calculates the closest enemy jet mid-flight.
* **Advanced VFX Engine:** Features screen shake, multi-layered explosions, fading missile trails, and a scrolling parallax night sky.
* **True Fullscreen Support:** Dynamic stretching to fit any monitor resolution without breaking the underlying AI coordinate math.

## 🧠 How the AI Works
The Reinforcement Learning agent evaluates the game state at 60 Frames Per Second to make instantaneous decisions.
* **State Space (Inputs):** The AI "sees" 6 normalized values: Base X, Closest Jet X, Closest Jet Y, Closest Incoming Laser X, Closest Incoming Laser Y, and Weapon Cooldown Status.
* **Neural Network Architecture:** Two hidden dense layers (128 neurons each) with ReLU activation and Dropout for noise resilience.
* **Action Space (Outputs):** The network predicts the optimal Q-Value for 4 distinct actions: `[0: Idle, 1: Move Left, 2: Move Right, 3: Fire Missile]`.

## 🛠️ How to Run Locally

**1. Clone the repository:**
```
git clone https://github.com/SUFIYAN2004/Jet-Defender-AI-with-DQN.git
cd Jet-Defender-AI-with-DQN

```

**2. Install dependencies:**
Make sure you have Python installed, then run:

```
pip install -r requirements.txt

```

**3. Run the game:**

```bash
python game.py

```

## 🕹️ Controls (Human Mode)

* **Left / Right Arrows:** Move the base
* **Spacebar:** Fire Homing Missiles
* **TAB:** Switch instantly between Human Pilot and AI Pilot
* **ESC:** Return to Main Menu / Quit Game
