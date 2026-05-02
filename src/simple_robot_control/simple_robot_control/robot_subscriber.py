#!/usr/bin/env python3
"""
Robot WASD keyboard controller.
Direct manual control via keyboard input.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import sys
import termios
import tty
import select
import signal


# ─────────────────────────────────────────────
#  Main Node
# ─────────────────────────────────────────────

class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')

        # ── Speed configuration ──────────────────────────────────────────
        self.max_linear_speed: float  = 0.5
        self.max_angular_speed: float = 1.0

        # Manual mode current speeds
        self.current_linear_speed  = 0.3
        self.current_angular_speed = 0.5

        # ── State ────────────────────────────────────────────────────────
        self.running: bool = True

        # ── ROS interfaces ───────────────────────────────────────────────
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        signal.signal(signal.SIGINT, self.sigint_handler)
        self.print_startup_info()

    # ─────────────────────────────────────────
    #  Signal handling
    # ─────────────────────────────────────────

    def sigint_handler(self, signum, frame):
        self.get_logger().info('Ctrl+C detected — shutting down')
        self.running = False

    # ─────────────────────────────────────────
    #  Velocity helpers
    # ─────────────────────────────────────────

    def send_velocity(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x  = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(msg)

    def stop_robot(self):
        self.send_velocity(0.0, 0.0)

    # ─────────────────────────────────────────
    #  Startup info
    # ─────────────────────────────────────────

    def print_startup_info(self):
        self.get_logger().info('=' * 55)
        self.get_logger().info('WASD Keyboard Control')
        self.get_logger().info('W/S   : Forward / Backward')
        self.get_logger().info('A/D   : Rotate Left / Right')
        self.get_logger().info('↑/↓   : Increase / Decrease speed')
        self.get_logger().info('Space : Stop')
        self.get_logger().info('Q / Ctrl+C : Quit')
        self.get_logger().info('=' * 55)

    # ─────────────────────────────────────────
    #  Main loop
    # ─────────────────────────────────────────

    def run(self):
        self._run_manual()

    # ── Manual (WASD) loop ───────────────────────────────────────────────

    def _run_manual(self):
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())

        try:
            self.get_logger().info('Manual WASD control started.')

            while rclpy.ok() and self.running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)

                    if key == '\x03':   # Ctrl+C raw
                        break

                    key_lower = key.lower()

                    if key_lower == 'q':
                        break

                    elif key_lower == 'w':
                        self.send_velocity(self.current_linear_speed, 0.0)
                        self.get_logger().info(f'Forward  {self.current_linear_speed:.1f} m/s')

                    elif key_lower == 's':
                        self.send_velocity(-self.current_linear_speed, 0.0)
                        self.get_logger().info(f'Backward {self.current_linear_speed:.1f} m/s')

                    elif key_lower == 'a':
                        self.send_velocity(0.0, self.current_angular_speed)
                        self.get_logger().info(f'Left     {self.current_angular_speed:.1f} rad/s')

                    elif key_lower == 'd':
                        self.send_velocity(0.0, -self.current_angular_speed)
                        self.get_logger().info(f'Right    {self.current_angular_speed:.1f} rad/s')

                    elif key == ' ':
                        self.stop_robot()
                        self.get_logger().info('Stop')

                    elif key == '\x1b':  # Escape → arrow key sequence
                        seq = sys.stdin.read(2)
                        if seq == '[A':   # ↑ increase speed
                            self.current_linear_speed  = min(self.current_linear_speed  + 0.1, self.max_linear_speed)
                            self.current_angular_speed = min(self.current_angular_speed + 0.1, self.max_angular_speed)
                            self.get_logger().info(
                                f'Speed ↑  lin={self.current_linear_speed:.1f}  ang={self.current_angular_speed:.1f}'
                            )
                        elif seq == '[B':  # ↓ decrease speed
                            self.current_linear_speed  = max(self.current_linear_speed  - 0.1, 0.1)
                            self.current_angular_speed = max(self.current_angular_speed - 0.1, 0.1)
                            self.get_logger().info(
                                f'Speed ↓  lin={self.current_linear_speed:.1f}  ang={self.current_angular_speed:.1f}'
                            )

                rclpy.spin_once(self, timeout_sec=0.01)

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            self.stop_robot()
            self.get_logger().info('Manual control stopped.')
            self.destroy_node()
            rclpy.shutdown()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    node.run()


if __name__ == '__main__':
    main()