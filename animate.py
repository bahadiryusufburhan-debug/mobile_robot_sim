import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

from environment import Environment
from robot import DifferentialRobot
from path_planner import AStarPlanner
from controller import PurePursuitController

def plot_environment(ax, env):
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    for ox, oy, ow, oh in env.obstacles:
        rect = Rectangle((ox - ow/2, oy - oh/2), ow, oh, color='gray', alpha=0.8)
        ax.add_patch(rect)
    ax.plot(env.start[0], env.start[1], 'go', markersize=8, label='Start')
    ax.plot(env.goal[0], env.goal[1], 'ro', markersize=8, label='Goal')

def main():
    print("Animasyon için simülasyon koşturuluyor...")
    env = Environment()
    robot = DifferentialRobot(x=env.start[0], y=env.start[1], theta=np.pi/4)
    planner = AStarPlanner(env, resolution=0.2, robot_rad=0.3)
    path = planner.plan(env.start, env.goal)
    
    controller = PurePursuitController(lookahead=1.0, max_v=1.0, max_omega=1.5, goal_tol=0.3)
    
    history_true = []
    dt = 0.1
    t = 0.0
    
    while t < 60.0:
        state = robot.get_state()
        history_true.append(state)
        
        if controller.reached_goal(state, env.goal):
            break
            
        v_cmd, omega_cmd = controller.compute(state, path)
        robot.update(v_cmd, omega_cmd, dt)
        t += dt

    H_true = np.array(history_true)
    
    print("Animasyon oluşturuluyor...")
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_environment(ax, env)
    ax.plot(path[:, 0], path[:, 1], 'b--', alpha=0.5, label='Planlanan Yol')
    
    robot_marker, = ax.plot([], [], 'ro', markersize=10, label='Robot', zorder=5)
    trail, = ax.plot([], [], 'g-', linewidth=2, label='Gerçek Yol', zorder=4)
    ax.legend(loc='upper left')
    
    def init():
        robot_marker.set_data([], [])
        trail.set_data([], [])
        return robot_marker, trail

    def update(frame):
        # frame is an index
        state = H_true[frame]
        robot_marker.set_data([state[0]], [state[1]])
        trail.set_data(H_true[:frame+1, 0], H_true[:frame+1, 1])
        return robot_marker, trail

    # Create animation, taking every 2nd frame to speed up gif generation
    frames_to_render = range(0, len(H_true), 2)
    ani = animation.FuncAnimation(fig, update, frames=frames_to_render, 
                                  init_func=init, blit=True)
    
    # Save as GIF
    ani.save('simulation.gif', writer='pillow', fps=10)
    print("Animasyon 'simulation.gif' olarak kaydedildi.")

if __name__ == '__main__':
    main()
