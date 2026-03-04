#!/usr/bin/env python3
"""
Simple keyboard control for the robot using WASD and arrow keys with IMU monitoring.
W/A/S/D: Forward/Left/Backward/Right
Up/Down Arrow: Increase/Decrease speed
Space: Stop
I: Toggle IMU display
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
import sys
import select
import termios
import tty
import math


class RobotKeyboardControl(Node):
    """Keyboard control node for the robot with IMU monitoring."""
    
    def __init__(self):
        super().__init__('robot_keyboard_control')
        
        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber for IMU data
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        
        # Control parameters
        self.max_linear_speed = 0.5  # m/s
        self.max_angular_speed = 1.0  # rad/s
        self.current_linear_speed = 0.3
        self.current_angular_speed = 0.5
        
        # IMU data storage
        self.imu_data = {
            'orientation': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
            'angular_velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'linear_acceleration': {'x': 0.0, 'y': 0.0, 'z': 0.0}
        }
        self.show_imu = False
        self.imu_received = False
        
        self.get_logger().info('Robot Keyboard Control with IMU Started!')
        self.get_logger().info('=' * 50)
        self.get_logger().info('Controls:')
        self.get_logger().info('  W: Forward')
        self.get_logger().info('  S: Backward')
        self.get_logger().info('  A: Turn Left')
        self.get_logger().info('  D: Turn Right')
        self.get_logger().info('  Up Arrow: Increase Speed')
        self.get_logger().info('  Down Arrow: Decrease Speed')
        self.get_logger().info('  Space: Stop')
        self.get_logger().info('  I: Toggle IMU Display')
        self.get_logger().info('  Q: Quit')
        self.get_logger().info('=' * 50)
        
        # Start keyboard input loop
        self.run_keyboard_control()
    
    def quaternion_to_euler(self, x, y, z, w):
        """
        Convert quaternion to euler angles (roll, pitch, yaw).
        Returns angles in degrees.
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Convert to degrees
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
    
    def imu_callback(self, msg):
        """Callback for IMU data."""
        self.imu_received = True
        
        # Extract orientation (quaternion)
        roll, pitch, yaw = self.quaternion_to_euler(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        )
        
        self.imu_data['orientation'] = {
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw
        }
        
        # Extract angular velocity
        self.imu_data['angular_velocity'] = {
            'x': msg.angular_velocity.x,
            'y': msg.angular_velocity.y,
            'z': msg.angular_velocity.z
        }
        
        # Extract linear acceleration
        self.imu_data['linear_acceleration'] = {
            'x': msg.linear_acceleration.x,
            'y': msg.linear_acceleration.y,
            'z': msg.linear_acceleration.z
        }
    
    def print_imu_data(self):
        """Print formatted IMU data."""
        if not self.imu_received:
            self.get_logger().info('IMU: Waiting for data...')
            return
        
        self.get_logger().info('=' * 50)
        self.get_logger().info('IMU Data:')
        self.get_logger().info(f"  Orientation (deg):")
        self.get_logger().info(f"    Roll:  {self.imu_data['orientation']['roll']:>7.2f}°")
        self.get_logger().info(f"    Pitch: {self.imu_data['orientation']['pitch']:>7.2f}°")
        self.get_logger().info(f"    Yaw:   {self.imu_data['orientation']['yaw']:>7.2f}°")
        self.get_logger().info(f"  Angular Velocity (rad/s):")
        self.get_logger().info(f"    X: {self.imu_data['angular_velocity']['x']:>7.3f}")
        self.get_logger().info(f"    Y: {self.imu_data['angular_velocity']['y']:>7.3f}")
        self.get_logger().info(f"    Z: {self.imu_data['angular_velocity']['z']:>7.3f}")
        self.get_logger().info(f"  Linear Acceleration (m/s²):")
        self.get_logger().info(f"    X: {self.imu_data['linear_acceleration']['x']:>7.3f}")
        self.get_logger().info(f"    Y: {self.imu_data['linear_acceleration']['y']:>7.3f}")
        self.get_logger().info(f"    Z: {self.imu_data['linear_acceleration']['z']:>7.3f}")
        self.get_logger().info('=' * 50)
    
    def get_key(self):
        """Get a single key from keyboard input."""
        try:
            # Check if there's input available
            if select.select([sys.stdin], [], [], 0)[0]:
                # Get the key
                old_settings = termios.tcgetattr(sys.stdin)
                try:
                    tty.setraw(sys.stdin.fileno())
                    key = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                return key
        except:
            pass
        return None
    
    def send_velocity(self, linear_x, angular_z):
        """Send velocity command to robot."""
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd)
        
        speed_pct = (self.current_linear_speed / self.max_linear_speed) * 100
        
        # Build status message
        status = f'Linear: {linear_x:>5.2f} m/s | Angular: {angular_z:>5.2f} rad/s | Speed: {speed_pct:>3.0f}%'
        
        # Add IMU yaw if available
        if self.imu_received:
            yaw = self.imu_data['orientation']['yaw']
            status += f' | Yaw: {yaw:>6.1f}°'
        
        self.get_logger().info(status)
    
    def run_keyboard_control(self):
        """Main loop for keyboard control."""
        try:
            # Save terminal settings
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            
            self.get_logger().info('Ready for input...')
            
            while rclpy.ok():
                # Get key input
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    
                    if key == 'q':
                        self.get_logger().info('Exiting...')
                        break
                    
                    # Toggle IMU display
                    elif key == 'i':
                        self.show_imu = not self.show_imu
                        if self.show_imu:
                            self.get_logger().info('IMU Display: ON')
                            self.print_imu_data()
                        else:
                            self.get_logger().info('IMU Display: OFF')
                    
                    # Speed control with arrow keys
                    elif key == '\x1b':  # Escape sequence for arrow keys
                        next_key = sys.stdin.read(2)
                        if next_key == '[A':  # Up arrow
                            self.current_linear_speed = min(
                                self.current_linear_speed + 0.1, 
                                self.max_linear_speed
                            )
                            self.current_angular_speed = min(
                                self.current_angular_speed + 0.1, 
                                self.max_angular_speed
                            )
                            self.get_logger().info(
                                f'Speed increased to {(self.current_linear_speed/self.max_linear_speed)*100:.0f}%'
                            )
                        elif next_key == '[B':  # Down arrow
                            self.current_linear_speed = max(
                                self.current_linear_speed - 0.1, 
                                0.1
                            )
                            self.current_angular_speed = max(
                                self.current_angular_speed - 0.1, 
                                0.1
                            )
                            self.get_logger().info(
                                f'Speed decreased to {(self.current_linear_speed/self.max_linear_speed)*100:.0f}%'
                            )
                    
                    # Movement control with WASD
                    elif key == 'w':
                        self.send_velocity(self.current_linear_speed, 0.0)
                        print("forward")
                    elif key == 's':
                        self.send_velocity(-self.current_linear_speed, 0.0)
                        print("backward")
                    elif key == 'a':
                        self.send_velocity(0.0, self.current_angular_speed)
                        print("left")
                    elif key == 'd':
                        self.send_velocity(0.0, -self.current_angular_speed)
                        print("right")
                    elif key == 'z':
                        self.send_velocity(0.0, 0.0)
                        print("stop")
                
                # Periodically print IMU data if enabled
                if self.show_imu:
                    # Print every ~1 second (10 iterations of 0.1s timeout)
                    if hasattr(self, '_imu_print_counter'):
                        self._imu_print_counter += 1
                        if self._imu_print_counter >= 10:
                            self.print_imu_data()
                            self._imu_print_counter = 0
                    else:
                        self._imu_print_counter = 0
                
                rclpy.spin_once(self, timeout_sec=0.01)
        
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            # Stop the robot
            self.send_velocity(0.0, 0.0)


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    node = RobotKeyboardControl()
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()