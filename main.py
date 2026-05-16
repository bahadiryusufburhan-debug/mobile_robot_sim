import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

# Import our modules
from environment import Environment
from robot import DifferentialRobot
from sensors import LiDAR, IMU, WheelEncoder
from kalman_filter import ExtendedKalmanFilter
from localization import DeadReckoning, compute_errors
from path_planner import AStarPlanner
from controller import PurePursuitController


def plot_environment(ax, env):
    """Utility to plot the environment with obstacles, start, and goal."""
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    
    # Plot obstacles
    for ox, oy, ow, oh in env.obstacles:
        rect = Rectangle((ox - ow/2, oy - oh/2), ow, oh, color='gray', alpha=0.8)
        ax.add_patch(rect)
    
    # Plot start and goal
    ax.plot(env.start[0], env.start[1], 'go', markersize=8, label='Start')
    ax.plot(env.goal[0], env.goal[1], 'ro', markersize=8, label='Goal')


def main():
    print("Simülasyon başlatılıyor...")
    
    # 1. Initialize Components
    env = Environment()
    robot = DifferentialRobot(x=env.start[0], y=env.start[1], theta=np.pi/4)
    
    lidar = LiDAR(max_range=8.0, num_beams=120)
    imu = IMU()
    encoder = WheelEncoder()
    
    ekf = ExtendedKalmanFilter(init_state=robot.get_state())
    dr = DeadReckoning(init_state=robot.get_state())
    
    print("Yol planlanıyor (A*)...")
    planner = AStarPlanner(env, resolution=0.2, robot_rad=0.3)
    path = planner.plan(env.start, env.goal)
    
    if path is None:
        print("Hedefe yol bulunamadı!")
        return
    print("Yol başarıyla planlandı.")

    controller = PurePursuitController(lookahead=1.0, max_v=1.0, max_omega=1.5, goal_tol=0.3)
    
    # Data recording lists
    history_true = []
    history_ekf = []
    history_dr = []
    history_t = []
    
    # Save one LiDAR scan for visualization (e.g., halfway through)
    sample_scan_data = None

    dt = 0.1
    t = 0.0
    max_time = 60.0
    
    print("Robot harekete geçiyor...")
    
    # 2. Simulation Loop
    while t < max_time:
        true_state = robot.get_state()
        history_true.append(true_state)
        history_ekf.append(ekf.state.copy())
        history_dr.append(dr.state.copy())
        history_t.append(t)
        
        if controller.reached_goal(true_state, env.goal):
            print(f"Hedefe başarıyla ulaşıldı! (Süre: {t:.1f} saniye)")
            break
            
        # Measurements
        v_m, omega_enc = encoder.measure(robot)
        omega_imu, accel = imu.measure(robot)
        ranges = lidar.scan(true_state, env)
        
        # Save a sample scan when robot is roughly in the middle
        if sample_scan_data is None and t > 5.0 and len(ranges) > 0:
            pts = lidar.scan_to_points(true_state, ranges)
            sample_scan_data = {'state': true_state, 'ranges': ranges, 'pts': pts}

        # Localization Update
        # Dead Reckoning
        dr.update(v_m, omega_enc, dt)
        
        # EKF Prediction (using encoders)
        ekf.predict(v_m, omega_enc, dt)
        
        # EKF Update (IMU)
        ekf.update_imu(omega_imu, dt)
        
        # EKF Update (LiDAR)
        # We simulate a position fix from LiDAR scan matching / landmark detection
        pos_meas = lidar.noisy_position(true_state, noise_std=0.2)
        ekf.update_lidar(pos_meas)
        
        # Control (using EKF state)
        v_cmd, omega_cmd = controller.compute(ekf.state, path)
        
        # Robot Kinematics Update
        robot.update(v_cmd, omega_cmd, dt)
        t += dt

    # Convert history to arrays
    H_true = np.array(history_true)
    H_ekf = np.array(history_ekf)
    H_dr = np.array(history_dr)
    H_t = np.array(history_t)
    
    # 3. Calculate Errors
    err_ekf, rmse_ekf, mae_ekf = compute_errors(H_true[:, :2], H_ekf[:, :2])
    err_dr, rmse_dr, mae_dr = compute_errors(H_true[:, :2], H_dr[:, :2])
    
    print("\n--- Hata Analizi (RMSE / MAE) ---")
    print(f"Dead Reckoning : {rmse_dr:.3f} m / {mae_dr:.3f} m")
    print(f"EKF (Füzyon)   : {rmse_ekf:.3f} m / {mae_ekf:.3f} m")

    # 4. Visualization (6 Required Plots)
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Mobil Robot Simülasyon Sonuçları', fontsize=18, fontweight='bold')
    
    # 6.1 Ortam Haritası (Environment Map)
    ax1 = fig.add_subplot(231)
    plot_environment(ax1, env)
    ax1.set_title('6.1 Ortam Haritası')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 6.2 Robot Yol Planı (Robot Path)
    ax2 = fig.add_subplot(232)
    plot_environment(ax2, env)
    ax2.plot(path[:, 0], path[:, 1], 'b--', linewidth=2, label='Planlanan Yol (A*)')
    ax2.plot(H_true[:, 0], H_true[:, 1], 'g-', linewidth=2, label='Gerçek Yol')
    ax2.set_title('6.2 Robot Yol Planı')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 6.3 Sensör Görselleştirmesi (Sensor Vis - LiDAR)
    ax3 = fig.add_subplot(233)
    if sample_scan_data:
        plot_environment(ax3, env)
        sx, sy, sth = sample_scan_data['state']
        pts = sample_scan_data['pts']
        ax3.plot(sx, sy, 'm^', markersize=10, label='Robot (Anlık)')
        if len(pts) > 0:
            ax3.scatter(pts[:, 0], pts[:, 1], c='r', s=10, label='LiDAR Noktaları')
        
        # Plot some rays
        for r, a in zip(sample_scan_data['ranges'][::10], lidar.angles[::10]):
            if r < lidar.max_range - 0.1:
                ax3.plot([sx, sx + r*np.cos(sth+a)], [sy, sy + r*np.sin(sth+a)], 'y-', alpha=0.3)
    ax3.set_title('6.3 Sensör Görselleştirmesi (LiDAR)')
    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.legend()
    ax3.set_xlim(sx-5, sx+5) if sample_scan_data else None
    ax3.set_ylim(sy-5, sy+5) if sample_scan_data else None
    ax3.grid(True, linestyle=':', alpha=0.6)

    # 6.4 Lokalizasyon Sonuçları (Localization Results)
    ax4 = fig.add_subplot(234)
    ax4.plot(H_true[:, 0], H_true[:, 1], 'g-', linewidth=2, label='Gerçek Yol')
    ax4.plot(H_dr[:, 0], H_dr[:, 1], 'r--', linewidth=1.5, label='Tahmin: Dead Reckoning')
    ax4.plot(H_ekf[:, 0], H_ekf[:, 1], 'b-.', linewidth=2, label='Tahmin: EKF (Füzyon)')
    ax4.set_title('6.4 Lokalizasyon Sonuçları (Yörünge)')
    ax4.set_xlabel('X [m]')
    ax4.set_ylabel('Y [m]')
    ax4.legend()
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.axis('equal')

    # 6.5 Hata Analizi (Error Analysis)
    ax5 = fig.add_subplot(235)
    ax5.plot(H_t, err_dr, 'r-', alpha=0.7, label=f'Dead Reckoning (RMSE: {rmse_dr:.2f}m)')
    ax5.plot(H_t, err_ekf, 'b-', linewidth=2, label=f'EKF (RMSE: {rmse_ekf:.2f}m)')
    ax5.set_title('6.5 Konum Hatası Analizi')
    ax5.set_xlabel('Zaman [s]')
    ax5.set_ylabel('Konum Hatası [m]')
    ax5.legend()
    ax5.grid(True, linestyle=':', alpha=0.6)

    # 6.6 Zaman Serisi Grafiği (Time Series)
    ax6 = fig.add_subplot(236)
    ax6.plot(H_t, H_true[:, 0], 'g-', label='Gerçek X')
    ax6.plot(H_t, H_ekf[:, 0], 'g--', alpha=0.7, label='EKF X')
    
    ax6.plot(H_t, H_true[:, 1], 'b-', label='Gerçek Y')
    ax6.plot(H_t, H_ekf[:, 1], 'b--', alpha=0.7, label='EKF Y')
    
    ax6.plot(H_t, H_true[:, 2], 'r-', label='Gerçek Theta')
    ax6.plot(H_t, H_ekf[:, 2], 'r--', alpha=0.7, label='EKF Theta')
    
    ax6.set_title('6.6 Zaman Serisi: Durum Tahmini vs Gerçek')
    ax6.set_xlabel('Zaman [s]')
    ax6.set_ylabel('Değer [m veya rad]')
    ax6.legend(fontsize=8, loc='upper left')
    ax6.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the plot
    plt.savefig('simulation_results.png', dpi=300)
    print("Grafikler oluşturuldu ve 'simulation_results.png' olarak kaydedildi.")
    
    # Optional: Display the plot
    # plt.show()


if __name__ == '__main__':
    main()
