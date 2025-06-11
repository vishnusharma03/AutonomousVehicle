#imports
import carla
import pygame
import queue
import random
import time
import cv2
import socket
import argparse
import numpy as np
import multiprocessing
from generate_visuals import visualize  


class Vehicle():
    def __init__(self, world, sync, fps=20):
        self.world = world
        self.blueprint = world.get_blueprint_library()
        self.ego_vehicle = None
        self.fps = fps
        self.sync = True if sync is None else sync
        settings = self.world.get_settings()
        self.original_settings = settings

        self.sensors = {}
        self.sensor_data = {}
        self.sensor_queues = {}

    # def initialize(self):
    #     pass

    # def get_vehicle(self):
    #     try:
    #         spawn_points = self.world.get_map().get_spawn_points()
    #         if not spawn_points:
    #             raise RuntimeError("No spawn points available in the map.")  # Added check (Changed)
    #         ego_bp = self.blueprint.find('vehicle.dodge.charger')

    #         ego_bp.set_attribute('role_name', 'hero')

    #         # self.ego_vehicle = self.world.spawn_actor(ego_bp, random.choice(spawn_points))
    #         self.ego_vehicle = self.world.try_spawn_actor(
    #             ego_bp, random.choice(spawn_points))
    #         if self.ego_vehicle is None:
    #             raise RuntimeError("Vehicle spawn failed")
    #     except:
    #         raise RuntimeError("Failed to spawn vehicle. Try again.")
    #     return self.ego_vehicle
    def get_vehicle(self):
        """Spawn the ego vehicle safely with try_spawn_actor."""
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available in the map.")  
        bp = self.blueprint.find('vehicle.dodge.charger')
        bp.set_attribute('role_name', 'hero')
        spawn_point = random.choice(spawn_points)  
        self.ego_vehicle = self.world.try_spawn_actor(bp, spawn_point) 
        if self.ego_vehicle is None:
            raise RuntimeError("Vehicle spawn failed")
        return self.ego_vehicle

    def Camera(self, transform):
        if transform is None:
            transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        try:
            camera_blueprint = self.blueprint.find('sensor.camera.rgb')
            camera_blueprint.set_attribute('image_size_x', '800')
            camera_blueprint.set_attribute('image_size_y', '600')
            camera_blueprint.set_attribute('fov', '110')
            # Set the time in seconds between sensor captures
            if self.sync:
                camera_blueprint.set_attribute('sensor_tick', str(1.0 / self.fps))  
            self.sensor_queues['rgb_camera'] = queue.Queue()
            rgb_camera = self.world.spawn_actor(camera_blueprint, transform, attach_to=self.ego_vehicle)
            rgb_camera.listen(lambda image: self._process_rgb_image(image))
            self.sensors['rgb_camera'] = rgb_camera
            return rgb_camera
        except Exception as e:
            import traceback
            print("[ERROR] camera Initialization failed:")
            traceback.print_exc()
            raise RuntimeError(f"Failed camera Initialization: {e}")

    def Lidar(self, position): # position = carla.Transform(carla.Location(x=0, z=2.4))
        if position is None:
            position = carla.Transform(carla.Location(x=0, z=2.0))
        try:
            lidar_bp = self.blueprint.find('sensor.lidar.ray_cast')
            # lidar_bp.set_attribite('')
            self.sensor_queues['lidar'] = queue.Queue()
            lidar_bp.set_attribute('channels', '64')
            lidar_bp.set_attribute('range', '100.0')
            lidar_bp.set_attribute('points_per_second', '1000000')
            lidar_bp.set_attribute('rotation_frequency', '20.0')  # Match your FPS!
                        
            # FOV settings
            lidar_bp.set_attribute('upper_fov', '10.0')
            lidar_bp.set_attribute('lower_fov', '-30.0')
            lidar_bp.set_attribute('noise_stddev', '0.1')
            if self.sync:
                lidar_bp.set_attribute('sensor_tick', str(1.0 / self.fps))
            lidar = self.world.spawn_actor(lidar_bp, position, attach_to=self.ego_vehicle)
            lidar.listen(self._process_lidar_data)
            self.sensors['lidar'] = lidar
            return lidar
        except Exception as e:
            import traceback
            print("[ERROR] Lidar Initialization failed:")
            traceback.print_exc()
            raise RuntimeError(f"Failed Lidar Initialization: {e}")

    def GNSS(self, position):
        if position is None:
            position = carla.Transform(carla.Location(x=0, z=0)) 
        try:
            gnss_bp = self.blueprint.find('sensor.other.gnss')
            self.sensor_queues['gnss'] = queue.Queue()
            gnss = self.world.spawn_actor(gnss_bp, position, attach_to=self.ego_vehicle)
            gnss.listen(lambda gnss_data: self._process_gnss_data(gnss_data))
        except:
            raise RuntimeError("Failed GNSS Initialization")
        self.sensors['gnss'] = gnss
        return gnss

    def IMU(self, position):
        if position is None:
            position = carla.Transform(carla.Location(x=0, z=0)) 
        imu_bp = self.blueprint.find('sensor.other.imu')
        if self.sync:
            imu_bp.set_attribute('sensor_tick', str(1.0 / self.fps)) 
        try:
            self.sensor_queues['imu'] = queue.Queue()
            imu = self.world.spawn_actor(imu_bp, position, attach_to=self.ego_vehicle)
            imu.listen(self._process_imu_data)
        
            self.sensors['imu'] = imu
            return imu
        except Exception as e:
            import traceback
            print("[ERROR] IMU Initialization failed:")
            traceback.print_exc()
            raise RuntimeError(f"Failed IMU Initialization: {e}")
        

    def Radar(self, position):
        if position is None:
            position = carla.Transform(carla.Location(x=2.5, z=1.0))
        try:
            radar_bp = self.blueprint.find('sensor.other.radar')
            self.sensor_queues['radar'] = queue.Queue()
            radar = self.world.spawn_actor(radar_bp, position, attach_to=self.ego_vehicle)
            radar.listen(lambda radar_data: self._process_radar_data(radar_data))
            self.sensors['radar'] = radar
        except:
            raise RuntimeError("Failed Radar Initialization")

        return radar

    def collision_sensor(self):
        """Setup a collision sensor"""
        try:
            collision_bp = self.blueprint.find('sensor.other.collision')
            
            # Create a queue for sensor data
            self.sensor_queues['collision'] = queue.Queue()
            
            # Spawn the collision sensor and attach it to the vehicle
            collision = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
            
            # Register callback function for the collision sensor
            collision.listen(lambda collision_data: self._process_collision_data(collision_data))
            
            self.sensors['collision'] = collision
        except:
            raise RuntimeError("Failed COllosion Sensor Initialization")
        return collision

    def _process_rgb_image(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        self.sensor_data['rgb_camera'] = arr  # Ensures data is stored (Changed)
        self.sensor_queues['rgb_camera'].put(arr)  # Ensures queueing (Changed)

    def _process_lidar_data(self, data):
        pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        self.sensor_data['lidar'] = pts  # Added storage (Changed)
        self.sensor_queues['lidar'].put(pts)  # Added queueing (Changed)

    def _process_gnss_data(self, gnss_data):
        """Process GNSS (GPS) data"""
        data = {
            'latitude': gnss_data.latitude,
            'longitude': gnss_data.longitude,
            'altitude': gnss_data.altitude
        }
        self.sensor_data['gnss'] = data
        self.sensor_queues['gnss'].put(data)

    def _process_imu_data(self, imu):
        d = {'accelerometer': imu.accelerometer,
             'gyroscope': imu.gyroscope, 'compass': imu.compass}
        self.sensor_data['imu'] = d  # Ensures data is stored (Changed)
        self.sensor_queues['imu'].put(d)  # Ensures queueing (Changed)

    def _process_radar_data(self, radar_data):
        """Process Radar data"""
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self.sensor_data['radar'] = points
        self.sensor_queues['radar'].put(points)

    def _process_collision_data(self, collision_data):
        """Process collision data"""
        data = {
            'actor': collision_data.other_actor,
            'impulse': collision_data.normal_impulse
        }
        self.sensor_data['collision'] = data
        self.sensor_queues['collision'].put(data)
    
    def tick(self):
        """Tick the world to synchronize sensors"""
        self.world.tick()
    
    def reset(self):
        print("Destroying all actors...")
        
        # Destroy sensors first
        for sensor in self.sensors.values():
            if sensor is not None and sensor.is_alive:
                sensor.destroy()
        
        # Destroy vehicle - Fixed: was self.vehicle, now self.ego_vehicle
        if self.ego_vehicle is not None and self.ego_vehicle.is_alive:
            self.ego_vehicle.destroy()
        
        # Restore original settings
        self.world.apply_settings(self.original_settings)

    def apply_control(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        """Apply control to the vehicle"""
        if self.ego_vehicle is None:
            return
        
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steer
        control.brake = brake
        control.hand_brake = hand_brake
        control.reverse = reverse
        
        self.ego_vehicle.apply_control(control)

    def get_latest_sensor_data(self, sensor_name):
        if sensor_name in self.sensor_queues:
            try:
                return self.sensor_queues[sensor_name].get_nowait()
            except queue.Empty:
                return self.sensor_data.get(sensor_name, None)
        return self.sensor_data.get(sensor_name, None)

# def parse_args():
#     parser = argparse.ArgumentParser(description="CARLA + Sensor Viz + Recording")
#     parser.add_argument('--host',    default='212.2.246.117')
#     parser.add_argument('-p', '--port',    type=int, default=2000)
#     parser.add_argument('--tm-port',       type=int, default=8001)
#     parser.add_argument('--sync',    action='store_true', default=True,
#                         help='Run in synchronous mode')
#     parser.add_argument('--sensors', default=None,
#                         help='Comma-separated list: rgb_camera,lidar,radar,gnss,imu,collision. Default=all')
#     parser.add_argument('--visualize', action='store_true', default=True,
#                         help='Show real-time Pygame visualization')
#     parser.add_argument('--record', action='store_true', default=True,
#                         help='Record the Pygame window to MP4')
#     parser.add_argument('--out', default='output.mp4',
#                         help='Output path for recording (if --record)')
#     parser.add_argument('--fps',   type=int, default=20,
#                         help='Target FPS for sync loop & recording')
#     parser.add_argument('--width',  type=int, default=1200,
#                         help='Window width')
#     parser.add_argument('--height', type=int, default=800,
#                         help='Window height')
#     return parser.parse_args()

def connect_client(host, port, timeout=30.0):
    """Connect to CARLA server if not already connected."""
    try:
        client = carla.Client(host, port)
        return client
    except (socket.timeout, RuntimeError):
        raise RuntimeError(f"Unable to connect to CARLA at {host}:{port}")

def get_or_create_tm(client, tm_port):
    """Reconnect to existing Traffic Manager on same port or create new one."""
    try:
        # TrafficManager is singleton per port, connecting multiple times is allowed
        tm = client.get_trafficmanager(tm_port)
        return tm
    except Exception as e:
        raise RuntimeError(f"Failed to get Traffic Manager on port {tm_port}: {e}")
    


# --- Utility functions ---


def cleanup_all_actors(client):
    """Destroy all vehicles and sensors in the world."""
    world = client.get_world()
    all_actors = world.get_actors()
    targets = list(all_actors.filter('vehicle.*')) + list(all_actors.filter('sensor.*'))
    if targets:
        cmds = [carla.command.DestroyActor(actor) for actor in targets]
        client.apply_batch(cmds)


# --- Core simulation logic ---
def run_episode(args):
    """Run a single CARLA episode with proper cleanup in finally."""
    client = None
    tm = None
    sync = True

    try:
        # Connect and setup
        client = connect_client(args.host, args.port)
        cleanup_all_actors(client)
        tm = get_or_create_tm(client, args.tm_port)

        # Configure synchronous mode if requested
        world = client.get_world()
        if args.sync:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = min(1.0 / args.fps, 0.1)
            world.apply_settings(settings)
            tm.set_synchronous_mode(True)

        # Spawn ego vehicle
        # blueprint_lib = world.get_blueprint_library()
        # spawn_pts = world.get_map().get_spawn_points()
        # ego_bp = blueprint_lib.find('vehicle.dodge.charger')
        # ego_bp.set_attribute('role_name', 'hero')
        # vehicle = world.try_spawn_actor(ego_bp, random.choice(spawn_pts))
        # if vehicle is None:
        #     raise RuntimeError("Failed to spawn ego vehicle")
        # vehicle.set_autopilot(True, tm.get_port())
        vm = Vehicle(world, sync)
        vm.get_vehicle()
        if vm is None:
            raise RuntimeError("Failed to spawn ego vehicle")
        vm.ego_vehicle.set_autopilot(True, tm.get_port())
        print("phase 2 Good")
        # pick sensors
        selected = ['rgb_camera','lidar'] # ,'radar','gnss',,'collision', 'imu'


        # init sensors
        def inits(vm):
            return {
                'rgb_camera': lambda: vm.Camera(None),
                'lidar':      lambda: vm.Lidar(None),
                'radar':      lambda: vm.Radar(None),
                'gnss':       lambda: vm.GNSS(None),
                'imu':        lambda: vm.IMU(None),
                'collision':  lambda: vm.collision_sensor()
            }
        init_funcs = inits(vm)
        print("phase 3")
        for s in selected:
            if s in init_funcs:
                print("phase 3a")
                init_funcs[s]()
            else:
                print(f"[WARN] Unknown sensor '{s}'")
        print("phase 4")


        # Simulation loop
        client.start_recorder("recording.log")
        max_ticks = args.fps * 5
        ticks = 0
        while ticks < max_ticks:
            print(ticks)
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()
            ticks += 1

        return 0

    finally:
        # Always attempt cleanup, even after Python exceptions
        if client:
            try:
                # Destroy actors
                client.stop_recorder()
                cleanup_all_actors(client)
                # Restore world settings if sync
                if args.sync:
                    world = client.get_world()
                    settings = world.get_settings()
                    settings.synchronous_mode = False
                    settings.fixed_delta_seconds = 0.0
                    world.apply_settings(settings)
            except Exception as cleanup_err:
                print(f"Cleanup error: {cleanup_err}")


# --- Supervisor wrapper ---
def supervise(args):
    """Supervise run_episode, restart if it crashes at C++ level."""
    while True:
        proc = multiprocessing.Process(target=run_episode, args=(args,))
        proc.start()
        proc.join()
        exit_code = proc.exitcode
        if exit_code == 0:
            print("Episode finished successfully.")
            break
        else:
            print(f"Episode crashed (exit code={exit_code}), performing global cleanup and restarting...")
            try:
                client = connect_client(args.host, args.port)
                cleanup_all_actors(client)
            except Exception as e:
                print(f"Global cleanup failed: {e}")
            time.sleep(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='212.2.246.117')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--tm_port', type=int, default=8001)
    parser.add_argument('--sync', action='store_true')
    parser.add_argument('--fps', type=int, default=20)
    return parser.parse_args()


if __name__ == '__main__':
    try:
        args = parse_args()
        supervise(args)
    except KeyboardInterrupt:
        print("Program interrupted by user. Exiting gracefully.")

    
# def main():
#     args = parse_args()
#     # --- connect & setup CARLA ---
#     client = connect_client(args.host, args.port)
#     tm = get_or_create_tm(client, args.tm_port)
#     time.sleep(5)
#     if args.sync:
#         tm.set_synchronous_mode(True)

#     print("phase 1 Good")
#     world = client.get_world()
#     # world = client.load_world('Town10HD_Opt') # Load Only Once
#     print("phase 1.1 Good")
#     if args.sync:
#         settings = world.get_settings()
#         settings.synchronous_mode = True
#         fps = args.fps
#         if fps > 10:
#             delta = 1.0 / fps
#         else:
#             delta = 0.1  # max allowed for stability
#         settings.fixed_delta_seconds = delta
#         world.apply_settings(settings)
#     print("phase 1.2 Good")
#     # --- spawn vehicle & sensors ---
#     sync=True
#     vm = Vehicle(world)
#     vm.get_vehicle()
#     vm.ego_vehicle.set_autopilot(True, tm.get_port())
#     print("phase 2 Good")
#     # pick sensors
#     ALL = ['rgb_camera','lidar'] # ,'radar','gnss','imu','collision'
#     selected = args.sensors.split(',') if args.sensors else ALL

#     # init sensors
#     def inits(vm):
#         return {
#             'rgb_camera': lambda: vm.Camera(None, sync),
#             'lidar':      lambda: vm.Lidar(None),
#             'radar':      lambda: vm.Radar(None),
#             'gnss':       lambda: vm.GNSS(None),
#             'imu':        lambda: vm.IMU(None),
#             'collision':  lambda: vm.collision_sensor()
#         }
#     init_funcs = inits(vm)
#     for s in selected:
#         if s in init_funcs:
#             init_funcs[s]()
#         else:
#             print(f"[WARN] Unknown sensor '{s}'")
#     print("phase 3 Good")
#     # --- optional pygame + recorder init ---
#     if args.visualize or args.record:
#         pygame.init()
#         window = pygame.display.set_mode((args.width, args.height))
#         pygame.display.set_caption("CARLA + Sensors")
#         clock = pygame.time.Clock()

#         # setup OpenCV VideoWriter if needed
#         if args.record:
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             writer = cv2.VideoWriter(args.out, fourcc, args.fps,
#                                      (args.width, args.height), True)
#     print("phase 4 Good")
#     # start CARLA recorder (world state)
#     client.start_recorder("recording.log")

#     # --- main loop: 30s @ fps ---
#     max_ticks = args.fps * 30
#     ticks = 0
#     try:
#         while ticks < max_ticks:
#             # advance sim
#             if args.sync:
#                 world.tick()
#             else:
#                 world.wait_for_tick()

#             # fetch data
#             rgb = None
#             if 'rgb_camera' in vm.sensor_queues:
#                 try:
#                     rgb = vm.sensor_queues['rgb_camera'].get(timeout=0.1)
#                 except queue.Empty:
#                     print("No RGB data available yet")

#             others = {s: vm.get_latest_sensor_data(s) for s in selected if s!='rgb_camera'}
#             print("phase 5.1 Good")
#             if args.visualize or args.record:
#                 # clear
#                 window.fill((0,0,0))
#                 # left panel: camera
#                 if rgb is not None:
#                     cam_surf = pygame.surfarray.make_surface(rgb.swapaxes(0,1))
#                     cam_surf = pygame.transform.scale(cam_surf, (args.width//2, args.height))
#                     window.blit(cam_surf, (0,0))
#                 else:
#                     # placeholder text
#                     font = pygame.font.Font(None, 36)
#                     window.blit(font.render("Waiting for camera...",True,(255,255,255)),
#                                 (10,10))
#                 print("phase 5.2 Good")
#                 # right panel: grid of other sensors
#                 n = len(others)
#                 if n>0:
#                     cols = int(np.ceil(np.sqrt(n)))
#                     rows = int(np.ceil(n/cols))
#                     cw = (args.width//2)//cols
#                     ch = args.height//rows
#                     for idx,(name,data) in enumerate(others.items()):
#                         x0 = args.width//2 + (idx%cols)*cw
#                         y0 = (idx//cols)*ch
#                         if data is None:
#                             txt = pygame.font.Font(None,24).render(f"{name}: waiting",True,(200,200,200))
#                             window.blit(txt,(x0+5,y0+5))
#                         elif name=='lidar':
#                             pts = data[:,:2]
#                             pts -= pts.min(0)
#                             pts /= pts.ptp(0)
#                             pts *= [cw-1,ch-1]
#                             for px,py in pts.astype(int):
#                                 window.set_at((x0+px,y0+py),(255,255,255))
#                         else:
#                             # generic text dump
#                             lines = []
#                             if hasattr(data,'raw_data'):
#                                 lines = [f"bytes: {len(data.raw_data)}"]
#                             else:
#                                 for k,v in data.__dict__.items():
#                                     lines.append(f"{k}:{v:.2f}")
#                             for i,line in enumerate(lines[:ch//18]):
#                                 window.blit(pygame.font.Font(None,18).render(line,True,(255,255,255)),
#                                             (x0+5,y0+5+i*18))
#                 print("phase 5.3 Good")

#                 # flip + capture
#                 pygame.display.flip()
#                 if args.record:
#                     # grab frame, convert to BGR for cv2
#                     raw = pygame.surfarray.array3d(window)
#                     frame = cv2.cvtColor(np.rot90(raw,3), cv2.COLOR_RGB2BGR)
#                     writer.write(frame)

#                 # input handling
#                 for ev in pygame.event.get():
#                     if ev.type==pygame.QUIT:
#                         ticks = max_ticks
#                     elif ev.type==pygame.KEYDOWN and ev.key==pygame.K_ESCAPE:
#                         ticks = max_ticks

#                 clock.tick(args.fps)

#             ticks += 1

#         print("phase 6 Good")
#     except Exception as e:
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc()
              

#     finally:
#         # 1) Stop CARLA’s internal recorder (world state)
#         client.stop_recorder()

#         # 2) Gather all actors (sensors + vehicle) and destroy them in one batch
#         actors_to_destroy = list(vm.sensors.values())
#         if vm.ego_vehicle is not None:
#             actors_to_destroy.append(vm.ego_vehicle)

#         if actors_to_destroy:
#             destroy_commands = [
#                 carla.command.DestroyActor(actor) for actor in actors_to_destroy
#             ]
#             client.apply_batch(destroy_commands)

#         # 3) Restore original world settings (exits synchronous mode)
#         vm.reset()  

#         # 4) Shut down the Traffic Manager cleanly
#         tm.shut_down()

#         # 5) Quit pygame if you initialized it
#         if (args.visualize or args.record) and 'pygame' in globals():
#             pygame.quit()

#         # 6) Release OpenCV VideoWriter if recording
#         if args.record:
#             writer.release()

#         print("Cleanup complete. Logs -> recording.log; video ->", args.out)


# if __name__ == '__main__':
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("Program interrupted by user. Exiting gracefully.")



# def parse_args():
#     parser = argparse.ArgumentParser(description="Spawn vehicle, sensors, and optionally visualize")
#     parser.add_argument('--host', default='212.2.246.117', help='CARLA host')
#     parser.add_argument('-p', '--port', type=int, default=2000, help='CARLA port')
#     parser.add_argument('--tm-port', type=int, default=8000, help='Traffic manager port')
#     parser.add_argument('--sync', action='store_true', help='Use synchronous mode')
#     parser.add_argument('--width', type=int, default=1200, help='Window width for visualization')
#     parser.add_argument('--height', type=int, default=800, help='Window height for visualization')
#     parser.add_argument('--sensors', default=None,
#                         help='Comma-separated list of sensors to initialize. Default=all')
#     parser.add_argument('--visualize', action='store_true', help='Launch integrated pygame visualization')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     pygame.init()
    
#     # Display settings
#     display_width = 800
#     display_height = 600
#     display = pygame.display.set_mode((display_width, display_height))
#     pygame.display.set_caption('CARLA Simulation')
#     sync = False
    
#     # Clock for controlling frame rate
#     clock = pygame.time.Clock()
#     try:
#         client = carla.Client('212.2.246.117', 2000)
#         tm = client.get_trafficmanager(8001)
#         time.sleep(5)
#         client.set_timeout(10.0)
#         if args.sync:
#             tm.set_synchronous_mode(True)
#         world = client.get_world()
#         # print(client.get_available_maps())
#         world = client.load_world('Town10HD_Opt')
#         # print("Client version:", client.get_client_version())
#         # print("Server version:", client.get_server_version())
#         vehicle_manager = Vehicle(world, sync)
#         vehicle_manager.get_vehicle()
#         vehicle_manager.Camera(None)
#         # vehicle_manager.Radar(None)
#         # vehicle_manager.GNSS(None)
#         # vehicle_manager.IMU(None)
#         # vehicle_manager.Lidar(None)
#         # vehicle_manager.Camera()
#         # vehicle_manager.Camera()

#         # vehicle_manager.ego_vehicle.set_autopilot(True)
#         if vehicle_manager.ego_vehicle:
#             vehicle_manager.ego_vehicle.set_autopilot(True, tm.get_port())
#         else:
#             raise RuntimeError("Failed to spawn vehicle")
        
#         client.start_recorder("recording.log")
#         print("Running simulation for 30 seconds...")
#         running = 0
    
        
#         while running < 600:  # 30 seconds at 20 FPS
#             # Handle pygame events
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     running = False
#                 elif event.type == pygame.KEYDOWN:
#                     if event.key == pygame.K_ESCAPE:
#                         running = False
            
#             # Tick the simulation
#             if sync:    
#                 vehicle_manager.tick()
#             else:
#                 world.wait_for_tick() 
#             # Get RGB camera data
#             rgb_data = vehicle_manager.get_latest_sensor_data('rgb_camera')
#             if rgb_data is not None:
#                 # Convert the image data to pygame surface
#                 # Assuming rgb_data is a numpy array with shape (height, width, 3)
#                 if len(rgb_data.shape) == 3:
#                     # Resize if necessary to fit display
#                     if rgb_data.shape[:2] != (display_height, display_width):
#                         rgb_data = cv2.resize(rgb_data, (display_height, display_width))
                    
#                     # Convert to pygame surface
#                     surface = pygame.surfarray.make_surface(rgb_data.swapaxes(0, 1))
#                     display.blit(surface, (0, 0))
            
#             # Update display
#             pygame.display.flip()
#             clock.tick(20)  # 20 FPS
#             running += 1
        
#     except Exception as e:
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc()
        
    
#     finally:
#         # Clean up
#         if 'vehicle_manager' in locals():
#             vehicle_manager.reset()
#         pygame.quit()
#         client.stop_recorder()
#         tm.shut_down()


# if __name__ == "__main__":
#     main()  