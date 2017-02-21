import gym
from PIL import Image

env = gym.make("Pong-v0")
s = env.reset()
for i in range(10):
    env.render()
    s, r, d, _ = env.step(1)
    image = Image.fromarray(s)
    image.save('state.png')
