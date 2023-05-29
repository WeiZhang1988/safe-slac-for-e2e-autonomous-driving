import numpy as np
import sys
import pygame
import random
import time
import gym
from gym.spaces import Dict, Discrete, Box, Tuple
import carla
from slac.carla_rl_env.bird_eye_view import BirdEyeView, PIXELS_PER_METER, \
PIXELS_AHEAD_VEHICLE
from slac.carla_rl_env.global_route_planner import GlobalRoutePlanner
from slac.carla_rl_env.controller import VehiclePIDController

class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time
    
    def time(self):
        return self.timer()

class DisplayManager:
    def __init__(self, grid_size, window_size, display_sensor):
        pygame.init()
        pygame.font.init()
        try:
            if display_sensor:
                self.display = pygame.display.set_mode(window_size, \
                pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.SHOWN)
            else:
                self.display = pygame.display.set_mode(window_size, \
                pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.HIDDEN)
            self.display.fill(pygame.Color(0,0,0))
        except Exception:
            print("display is not correctly created in init")
        
        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []
        self.bev = None # bev is short for bird eye view
        
    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]
        
    def get_display_size(self):
        return [int(self.window_size[0]/self.grid_size[1]), \
        int(self.window_size[1]/self.grid_size[0])]
        
    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), \
        int(gridPos[0] * dis_size[1])]
    
    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)
    
    def get_sensor_list(self):
        return self.sensor_list
        
    def add_birdeyeview(self, bev):
        self.bev = bev
        
    def render(self):
        for s in self.sensor_list:
            if s.surface is not None:
                self.display.blit(s.surface, \
                self.get_display_offset(s.display_pos))
        self.display.blit(self.bev.surface, \
        self.get_display_offset(self.bev.display_pos))
        self.display.blit(self.bev.surface_global, \
        self.get_display_offset(self.bev.display_pos_global))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.flip()
            
    def clear(self):
        self.sensor_list = []
        self.bev = None

class SensorManager:
    def __init__(self, world, sensor_type, transform, attached, \
    sensor_options, display_size, display_pos):
        # resources from outside
        self.world = world
        self.sensor_type = sensor_type
        self.transform = transform
        self.attached = attached
        self.sensor_options = sensor_options
        self.display_size = display_size
        self.display_pos = display_pos
        # self created resource
        self.surface = None
        self.measure_data = None
        self.sensor = self.init_sensor(sensor_type, transform, \
        attached, sensor_options, display_size)
        
        self.timer = CustomTimer()
        self.time_processing = 0.0
        self.tics_processing = 0
        
    def init_sensor(self, sensor_type, transform, \
        attached, sensor_options, display_size):
        if sensor_type == 'RGBCamera':
            camera_bp = \
            self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', \
            str(display_size[0]))
            camera_bp.set_attribute('image_size_y', \
            str(display_size[1]))
            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])
            camera = self.world.spawn_actor(camera_bp, transform, \
            attach_to=attached)
            img_size = (self.display_size[0], self.display_size[1], 3)
            self.measure_data = np.zeros((img_size), dtype=np.uint8)
            camera.listen(self.save_rgb_image)
            return camera
            
        elif sensor_type == 'LiDAR':
            lidar_bp = \
            self.world.get_blueprint_library().\
            find('sensor.lidar.ray_cast')
            #lidar_bp.set_attribute('range', '100')
            lidar_bp.set_attribute('dropoff_general_rate', \
            lidar_bp.get_attribute('dropoff_general_rate').\
            recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit', \
            lidar_bp.get_attribute('dropoff_intensity_limit').\
            recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity', \
            lidar_bp.get_attribute('dropoff_zero_intensity').\
            recommended_values[0])
            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])
            lidar = self.world.spawn_actor(lidar_bp, transform, \
            attach_to=attached)
            img_size = (self.display_size[0], self.display_size[1], 3)
            self.measure_data = np.zeros((img_size), dtype=np.uint8)
            lidar.listen(self.save_lidar_image)
            return lidar
            
        elif sensor_type == 'Radar':
            radar_bp = \
            self.world.get_blueprint_library().\
            find('sensor.other.radar')
            for key in sensor_options:
                radar_bp.set_attribute(key, sensor_options[key])
            radar = self.world.spawn_actor(radar_bp, transform, \
            attach_to=attached)
            img_size = (self.display_size[0], self.display_size[1], 3)
            self.measure_data = np.zeros((img_size), dtype=np.uint8)
            radar.listen(self.save_radar_image)
            return radar
            
        elif sensor_type == 'GNSS':
            gnss_bp = \
            self.world.get_blueprint_library().\
            find('sensor.other.gnss')
            for key in sensor_options:
                gnss_bp.set_attribute(key, sensor_options[key])
            gnss = self.world.spawn_actor(gnss_bp, transform, \
            attach_to=attached)
            gnss.listen(self.save_gnss_msg)
            return gnss
            
        elif sensor_type == 'IMU':
            imu_bp = \
            self.world.get_blueprint_library().\
            find('sensor.other.imu')
            for key in sensor_options:
                imu_bp.set_attribute(key, sensor_options[key])
            imu = self.world.spawn_actor(imu_bp, transform, \
            attach_to=attached)
            imu.listen(self.save_imu_msg)
            return imu
            
        elif sensor_type == 'Collision':
            collision_bp = \
            self.world.get_blueprint_library().\
            find('sensor.other.collision')
            for key in sensor_options:
                collision_bp.set_attribute(key, sensor_options[key])
            collision = self.world.spawn_actor(collision_bp, \
            transform, attach_to=attached)
            collision.listen(self.save_collision_msg)
            return collision
            
        elif sensor_type == 'Lane_invasion':
            lane_invasion_bp = \
            self.world.get_blueprint_library().\
            find('sensor.other.lane_invasion')
            for key in sensor_options:
                lane_invasion_bp.set_attribute(key, sensor_options[key])
            lane_invasion = self.world.spawn_actor(lane_invasion_bp, \
            transform, attach_to=attached)
            lane_invasion.listen(self.save_lane_invasion_msg)
            return lane_invasion
            
        else:
            return None
            
    def get_sensor(self):
        return self.sensor
        
    def destroy_sensor(self):
        if self.sensor.is_alive:
            self.sensor.destroy()
        
        del self.sensor
        del self.surface
        del self.measure_data
        del self.timer
        del self.time_processing
        del self.tics_processing
        
    def save_rgb_image(self, image):
        t_start = self.timer.time()
        
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.measure_data = array.swapaxes(0,1)
        self.surface = pygame.surfarray.make_surface(self.measure_data)
        
        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
        
    def save_lidar_image(self, image):
        t_start = self.timer.time()
        
        lidar_range = 2.0 * float(self.sensor_options['range'])
        
        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points_height = np.array(points[:, 2],dtype=np.float32)
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(self.display_size) / lidar_range
        lidar_data += (0.5 * self.display_size[0], \
        0.5 * self.display_size[1])
        lidar_data = np.fabs(lidar_data)
        lidar_data = lidar_data.astype(np.int32)
        lidar_img_size = (self.display_size[0], self.display_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
        
        #lidar_img[tuple(lidar_data.T)] = (0,255,0) 
        points_height = (points_height - (-1.4)) / (-1.0-(-1.4))
        points_height = np.expand_dims(points_height,axis=1)
        points_height = np.clip(points_height,0.0,1.0)
        
        height_data = points_height*255.0
        height_data = np.append(height_data,(1.0 - points_height)*255.0,axis=1)
        height_data = np.append(height_data,np.ones_like(points_height)*0.0,axis=1)
            
        lidar_img[tuple(lidar_data.T)] = height_data.astype(np.uint8) 

        lidar_img = np.rot90(lidar_img,3)
        self.measure_data = lidar_img
        self.surface = pygame.surfarray.make_surface(lidar_img)
        
        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
        
    def save_radar_image(self, radar_data):
        t_start = self.timer.time()
        
        radar_range = 1.0 * float(self.sensor_options['range'])
        radar_scale = min(self.display_size) / radar_range
        radar_offset = min(self.display_size) / 2.0
        
        self.surface = pygame.Surface(self.display_size).convert()
        self.surface.fill(pygame.Color(0,0,0))
        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            alt = detect.altitude
            azi = detect.azimuth
            dpt = detect.depth
            x = dpt * np.cos(alt) * np.cos(azi)
            y = dpt * np.cos(alt) * np.sin(azi)
            z = dpt * np.sin(alt)
            
            center_point = \
            pygame.math.Vector2(x * radar_scale, \
            y * radar_scale + radar_offset)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            velocity_limit = 20.0
            norm_velocity = detect.velocity / velocity_limit
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            pygame.draw.circle(self.surface,pygame.Color(r,g,b),\
            center_point,5)

        self.measure_data = pygame.surfarray.array3d(self.surface)
        
        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
        
    def save_gnss_msg(self, gnss_msg):
        lat = gnss_msg.latitude
        lon = gnss_msg.longitude
        alt = gnss_msg.altitude
        self.measure_data = np.array([lat, lon, alt])
        
    def save_imu_msg(self, imu_msg):
        acc = np.array([imu_msg.accelerometer.x, \
        imu_msg.accelerometer.y, \
        imu_msg.accelerometer.z])
        gyro = np.array([imu_msg.gyroscope.x, \
        imu_msg.gyroscope.y, \
        imu_msg.gyroscope.z])
        cmpa = np.array([imu_msg.compass])
        self.measure_data = (acc, gyro, cmpa)
        
    def save_collision_msg(self, collision_msg):
        self.measure_data = True
        
    def save_lane_invasion_msg(self, lane_invasion_msg):
        list_type = \
        list(set(x.type for x in \
        lane_invasion_msg.crossed_lane_markings))
        self.measure_data = str(list_type[-1])

class TargetPosition(object):
    def __init__(self, transform):
        # resource from outside
        self.transform = transform
        # self created resource
        self.box = None
        self.measure_data = None
    
        self.set_transform(transform)
    
    def set_transform(self, transform):
        self.transform = transform
        self.box = carla.BoundingBox(transform.location, \
        carla.Vector3D(1,1,1))
        self.measure_data = np.array([
        self.transform.location.x, 
        self.transform.location.y, 
        self.transform.location.z])
        
    def destroy_target_pos(self):
        del self.box
        del self.measure_data

class CarlaRlEnv(gym.Env):
    def __init__(self, params):
        # resource from outside
        # parse parameters
        self.carla_port = params['carla_port']
        self.map_name = params['map_name']
        self.window_resolution = params['window_resolution']
        self.grid_size = params['grid_size']
        self.sync = params['sync']
        self.no_render = params['no_render']
        self.display_sensor = params['display_sensor']
        self.ego_filter = params['ego_filter']
        self.num_vehicles = params['num_vehicles']
        self.num_pedestrians = params['num_pedestrians']
        self.enable_route_planner = params['enable_route_planner']
        self.sensors_to_amount = params['sensors_to_amount']
        
        # self created resource
        # connet to server
        self.client = carla.Client('localhost',self.carla_port) #
        self.client.set_timeout(10.0)
        # get world and map
        self.world = self.client.get_world() #
        self.world = self.client.load_world(self.map_name)
        self.spectator = self.world.get_spectator() #
        self.map = self.world.get_map() #
        self.spawn_points = self.map.get_spawn_points() #
        self.original_settings = self.world.get_settings() #
        if self.no_render:
            settings = self.world.get_settings()
            settings.no_rendering_mode = True
            self.world.apply_settings(settings)
        if self.sync:
            traffic_manager = self.client.get_trafficmanager(8000)
            settings = self.world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            self.world.apply_settings(settings)
        
        self.ego_vehicle = None
        self.display_manager = DisplayManager(self.grid_size, \
        self.window_resolution, self.display_sensor)
        self.display_size = self.display_manager.get_display_size()
        self.vehicle_list = []
        self.sensor_list = []
        self.bev = None
        self.pedestrian_list = []
        self.pedestrian_controller_list = []
        
        self.target_pos = None #TargetPosition(carla.Transform())
        
        self.route_planner_global = GlobalRoutePlanner(self.map,0.5)
        self.waypoints = None
        self.waypoints_horizon = None
        self.last_waypoints_len = None
        
        self.current_step = 0
        self.reward = 0.0
        self.done = False
        self.cost = 0.0
        self.total_step = 0
        
        self.left_camera = None
        self.front_camera = None
        self.right_camera = None
        self.rear_camera = None
        self.lidar = None
        self.radar = None
        self.gnss = None
        self.imu = None
        self.collision = None
        self.lane_invasion = None
        
        # acc and brake percentage, steering percentage, and reverse flag
        self.action_space = Tuple((Box(np.array([0.0, 0.0, -1.0]), 1.0, \
        shape=(3,), dtype=np.float32), Discrete(2)))
        # 4 cameras, 1 lidar, and 1 GNSS
        self.observation_space = Dict({
            'front_camera': Box(0, 255, shape=(self.display_size[0], \
            self.display_size[1], 3), dtype=np.uint8),
            'left_camera': Box(0, 255, shape=(self.display_size[0], \
            self.display_size[1], 3), dtype=np.uint8),
            'right_camera': Box(0, 255, shape=(self.display_size[0], \
            self.display_size[1], 3), dtype=np.uint8),
            'rear_camera': Box(0, 255, shape=(self.display_size[0], \
            self.display_size[1], 3), dtype=np.uint8),
            'lidar_image': Box(0, 255, shape=(self.display_size[0], \
            self.display_size[1], 3), dtype=np.uint8),
            'radar_image': Box(0, 255, shape=(self.display_size[0], \
            self.display_size[1], 3), dtype=np.uint8),
            'gnss': Box(-np.inf, np.inf, shape=(3,), dtype=np.float32), 
            'imu': Tuple((\
            Box(-np.inf, np.inf, shape=(3,), dtype=np.float32), \
            Box(-np.inf, np.inf, shape=(3,), dtype=np.float32), \
            Box(-np.inf, np.inf, shape=(1,), dtype=np.float32), \
            )),
            'bev': Box(0, 255, shape=(self.display_size[0], \
            self.display_size[1], 3), dtype=np.uint8),
            'trgt_pos' : Box(-np.inf, np.inf, shape=(3,), \
            dtype=np.float32),
            'wp_hrz' : Box(-np.inf, np.inf, shape=(40,2), \
            dtype=np.float32)
        })
    
    def step(self,action):
        self.current_step += 1
        self.total_step += 1
    
        acc = action[0][0]
        brk = action[0][1]
        trn = action[0][2]
        rvs = action[1][0]
        
        #print("acc",acc)
        #print("brk",brk)
        #print("trn",trn)
        #print("rvs",rvs)
        
        act = carla.VehicleControl(throttle=float(acc), \
        steer=float(trn), brake=float(brk), reverse=bool(rvs))
        
        self.ego_vehicle.apply_control(act)
        
        self.world.tick()
        reward, done, cost = self.deal_with_reward_and_done()
        
        self.bev.update_bird_eye_view()
        
        transform = self.ego_vehicle.get_transform()
        transform.location.z += 10
        transform.rotation.pitch -= 90
        #self.spectator.set_transform(transform)
        
        img_size = (180,180)
        observation = {
        'left_camera' : self.left_camera.measure_data if self.left_camera is not None else np.zeros(img_size),
        'front_camera': self.front_camera.measure_data if self.front_camera is not None else np.zeros(img_size),
        'right_camera': self.right_camera.measure_data if self.right_camera is not None else np.zeros(img_size),
        'rear_camera' : self.rear_camera.measure_data if self.rear_camera is not None else np.zeros(img_size),
        'lidar_image' : self.lidar.measure_data if self.lidar is not None else np.zeros(img_size),
        'radar_image' : self.radar.measure_data if self.radar is not None else np.zeros(img_size),
        'gnss': self.gnss.measure_data if self.gnss is not None else np.zeros(3),
        'imu': self.imu.measure_data if self.imu is not None else (np.zeros(3),np.zeros(3),np.zeros(1)),
        'bev': self.bev.measure_data,
        'trgt_pos' : self.target_pos.measure_data,
        'wp_hrz' : self.waypoints_horizon,
        }

        info = {'cost':cost}
        
        return observation,reward,done,info
    
    def reset(self):
        self.current_step = 0
    
        self.remove_all_actors()
        self.world.tick()
        self.create_all_actors()
        self.world.tick()

        self.reward = 0.0
        self.done = False
        self.cost = 0.0
        _,_ = self.update_waypoints_and_horizon()
        img_size = (180,180)
        observation = {
        'left_camera' : self.left_camera.measure_data if self.left_camera is not None else np.zeros(img_size),
        'front_camera': self.front_camera.measure_data if self.front_camera is not None else np.zeros(img_size),
        'right_camera': self.right_camera.measure_data if self.right_camera is not None else np.zeros(img_size),
        'rear_camera' : self.rear_camera.measure_data if self.rear_camera is not None else np.zeros(img_size),
        'lidar_image' : self.lidar.measure_data if self.lidar is not None else np.zeros(img_size),
        'radar_image' : self.radar.measure_data if self.radar is not None else np.zeros(img_size),
        'gnss': self.gnss.measure_data if self.gnss is not None else np.zeros(3),
        'imu': self.imu.measure_data if self.imu is not None else (np.zeros(3),np.zeros(3),np.zeros(1)),
        'bev': self.bev.measure_data,
        'trgt_pos' : self.target_pos.measure_data,
        'wp_hrz' : self.waypoints_horizon,
        }
        
        return observation
        
    def update_waypoints_and_horizon(self):
        current_location = self.ego_vehicle.get_transform().location
        num_checked_waypoints = min(10,len(self.waypoints))
        dis = 10000.0
        idx = 1000
        for n_c_wp in range(num_checked_waypoints):
            if current_location.distance(self.waypoints[n_c_wp][0].transform.location) \
            < dis:
                dis = current_location.distance(self.waypoints[n_c_wp][0].transform.location)
                if len(self.waypoints) > 10:
                    wp = self.waypoints[9]
                else:
                    wp = self.waypoints[-1]
                idx = n_c_wp
        for _ in range(idx):
            self.waypoints.pop(0)
        
        waypoints_horizon = []
        if len(self.waypoints)>40:
            wpsh = self.waypoints[:40]
        else:
            wpsh = self.waypoints + [self.waypoints[-1]]*(40-len(self.waypoints))
        for wph in wpsh:
            waypoints_horizon.append(np.array([wph[0].transform.location.x,wph[0].transform.location.y]))
        self.waypoints_horizon = np.array(waypoints_horizon)

        return current_location, wp
        
    def deal_with_reward_and_done(self):
        self.reward = 0.0
        self.cost = 0.0
        current_location, wp = self.update_waypoints_and_horizon()
        
        def cal_lat_error_2D(waypoint,location):
            vec_2D = np.array([
            location.x-waypoint[0].transform.location.x,
            location.y-waypoint[0].transform.location.y])
            lv_2D = np.linalg.norm(np.array(vec_2D))
            omega_2D = np.array([
            np.cos(waypoint[0].transform.rotation.yaw/180.0*np.pi),
            np.sin(waypoint[0].transform.rotation.yaw/180.0*np.pi)])
            cross = np.cross(omega_2D, vec_2D/lv_2D)
            return - lv_2D * cross, omega_2D
        
        # time elapes    
        time_reward = -1.0
        if self.current_step > 1000:
            self.done = True
            
        # collision
        if self.collision.measure_data:
            collision_cost = 1.0
            self.done = True
            self.collision.measure_data = None
        else:
            collision_cost = 0.0
            
        # lane invasion
        if self.lane_invasion.measure_data:
            if self.lane_invasion.measure_data == 'Broken' or \
            self.lane_invasion.measure_data == 'BrokenSolid' or \
            self.lane_invasion.measure_data == 'BrokenBroken':
                lane_invasion_cost = 0.0
            else:
                self.done = True
                lane_invasion_cost = 1.0
            self.lane_invasion.measure_data = None
        else:
            lane_invasion_cost = 0.0
            
        current_wp = self.map.get_waypoint(current_location)
        if current_location.distance(current_wp.transform.location) > (current_wp.lane_width / 2.0 + 0.2):
            self.done = True
            lane_invasion_cost = 1.0
   
        # traffic light
        #if self.ego_vehicle.is_at_traffic_light():
        #    self.done = True
        #    cross_red_light_reward = -1.0
        #else:
        #    cross_red_light_reward = 0.0
        
        # speed limit
        current_velocity = self.ego_vehicle.get_velocity()
        current_speed = np.sqrt(current_velocity.x**2 + \
        current_velocity.y**2 + \
        current_velocity.z**2)    # unit m/s
        current_speed_limit = 20.0    # unit m/s
        road_speed_limit = self.ego_vehicle.get_speed_limit()
        if road_speed_limit is not None:
            current_speed_limit = \
            min(current_speed_limit,road_speed_limit / 3.6)
        
        distance = \
        current_location.distance(self.target_pos.transform.location)
        if distance < 1.0:
            arriving_reward = 1.0
            self.done = True
        else:
            arriving_reward = 0.0
        
        lat_err, omg = cal_lat_error_2D(wp,current_location)

        if abs(lat_err) > 1.2 * 4.0 :
            off_way_reward = - 10.0 * abs(lat_err)
            #self.done = True
        else:
            off_way_reward = - 0.0 * abs(lat_err)
        
        v_long = np.dot(np.array([current_velocity.x, \
        current_velocity.y]),omg)
        
        if v_long > current_speed_limit:
            speed_reward = -5.0 * (current_speed_limit - v_long)**2.0
        else:
            speed_reward = v_long
        
        #print("------------------------------------------")
        #print("current_velocity.x",current_velocity.x)
        #print("current_speed_limit",current_speed_limit)
        #print("v_long",v_long)
        #print("speed_reward",speed_reward)
        #print("------------------------------------------")
        
        steer_reward = -self.ego_vehicle.get_control().steer**2
        lat_acc_reward = - abs(self.ego_vehicle.get_control().steer) * v_long**2
        
        # waypoints len reward
        waypoints_len_reward = self.last_waypoints_len - len(self.waypoints)
        self.last_waypoints_len = len(self.waypoints)
        
        self.reward = 0.1 * time_reward + \
        200.0 * arriving_reward + \
        2.0 * off_way_reward + \
        0.1 * speed_reward + \
        3.0 * steer_reward + \
        0.5 * lat_acc_reward + \
        3.0 * waypoints_len_reward
        
        self.cost = 200.0 * collision_cost + 10.0 * lane_invasion_cost
        
        
        return self.reward, self.done, self.cost
            
    def create_all_actors(self):
        self.target_pos = TargetPosition(carla.Transform\
        (carla.Location(-114.23,53.82,0.6),carla.Rotation(0.0,90.0,0.0)))
        #self.target_pos.set_transform(random.choice(self.spawn_points))

        # create ego vehicle
        ego_vehicle_bp = \
        random.choice([bp for bp in self.world.get_blueprint_library().\
        filter(self.ego_filter) \
        if int(bp.get_attribute('number_of_wheels'))==4])
        ego_vehicle_bp.set_attribute('role_name','hero')
        
        self.ego_vehicle = None
        #while self.ego_vehicle is None:
        #    self.ego_vehicle = \
        #    self.world.try_spawn_actor(ego_vehicle_bp, \
        #    random.choice(self.spawn_points))
        
        while self.ego_vehicle is None:
            self.ego_vehicle = \
            self.world.try_spawn_actor(ego_vehicle_bp, \
            carla.Transform(carla.Location(-18.386,130.21,0.546),carla.Rotation(0.0,-180.0,0.0)))
            #time.sleep(0.1)
        self.vehicle_list.append(self.ego_vehicle)
        self.world.tick()

        self.waypoints = \
        self.route_planner_global.trace_route( \
        self.ego_vehicle.get_location(), \
        self.target_pos.transform.location)
        self.last_waypoints_len = len(self.waypoints)
        if len(self.waypoints) ==0 :
            print("planned waypoints length is zero")
        
        bbe_x = self.ego_vehicle.bounding_box.extent.x
        bbe_y = self.ego_vehicle.bounding_box.extent.y
        bbe_z = self.ego_vehicle.bounding_box.extent.z
        
        if 'left_rgb' in self.sensors_to_amount:
            self.left_camera = SensorManager(self.world, 'RGBCamera', \
            carla.Transform(carla.Location(x=0, z=bbe_z+1.4), \
            carla.Rotation(yaw=-90)), self.ego_vehicle, {}, \
            self.display_size, [0, 0])
            self.sensor_list.append(self.left_camera)
            self.display_manager.add_sensor(self.left_camera)

        if 'front_rgb' in self.sensors_to_amount:
            self.front_camera = SensorManager(self.world, 'RGBCamera', \
            carla.Transform(carla.Location(x=0, z=bbe_z+1.4), \
            carla.Rotation(yaw=+00)), self.ego_vehicle, {}, \
            self.display_size, [0, 1])
            self.sensor_list.append(self.front_camera)
            self.display_manager.add_sensor(self.front_camera)
        
        if 'right_rgb' in self.sensors_to_amount:
            self.right_camera = SensorManager(self.world, 'RGBCamera', \
            carla.Transform(carla.Location(x=0, z=bbe_z+1.4), \
            carla.Rotation(yaw=+90)), self.ego_vehicle, {}, \
            self.display_size, [0, 2])
            self.sensor_list.append(self.right_camera)
            self.display_manager.add_sensor(self.right_camera)
        
        if 'rear_rgb' in self.sensors_to_amount:
            self.rear_camera = SensorManager(self.world, 'RGBCamera', \
            carla.Transform(carla.Location(x=0, z=bbe_z+1.4), \
            carla.Rotation(yaw=180)), self.ego_vehicle, {}, \
            self.display_size, [1, 1])
            self.sensor_list.append(self.rear_camera)
            self.display_manager.add_sensor(self.rear_camera)
        
        if 'top_rgb' in self.sensors_to_amount:
            self.top_camera = SensorManager(self.world, 'RGBCamera', \
            carla.Transform(carla.Location(x=0, z=20), \
            carla.Rotation(pitch=-90)), self.ego_vehicle, {}, \
            self.display_size, [2, 1])
            self.sensor_list.append(self.top_camera)
            self.display_manager.add_sensor(self.top_camera)
        
        if 'lidar' in self.sensors_to_amount:
            self.lidar = SensorManager(self.world, 'LiDAR', \
            carla.Transform(carla.Location(x=0, z=bbe_z+1.4)), \
            self.ego_vehicle, \
            {'channels' : '64', 'range' : '25.0',  \
            'points_per_second': '250000', 'rotation_frequency': '20'}, \
            self.display_size, [1, 0])
            self.sensor_list.append(self.lidar)
            self.display_manager.add_sensor(self.lidar)
        
        if 'radar' in self.sensors_to_amount:
            bound_x = 0.5 + bbe_x
            bound_y = 0.5 + bbe_y
            bound_z = 0.5 + bbe_z
            self.radar = SensorManager(self.world, 'Radar', \
            carla.Transform(\
            carla.Location(x=bound_x + 0.05, z=bound_z+0.05), \
            carla.Rotation(pitch=5)), \
            self.ego_vehicle, \
            {'horizontal_fov' : '60', 'vertical_fov' : '30', \
            'range' : '20.0'}, \
            self.display_size, [1, 2])
            self.sensor_list.append(self.radar)
            self.display_manager.add_sensor(self.radar)
        
        if 'gnss' in self.sensors_to_amount:
            self.gnss = SensorManager(self.world, 'GNSS', \
            carla.Transform(), self.ego_vehicle, {}, None, None)
            self.sensor_list.append(self.gnss)
        
        if 'imu' in self.sensors_to_amount:
            self.imu = SensorManager(self.world, 'IMU', \
            carla.Transform(), self.ego_vehicle, {}, None, None)
            self.sensor_list.append(self.imu)
        
        self.bev = BirdEyeView(self.world, \
        PIXELS_PER_METER, PIXELS_AHEAD_VEHICLE, \
        self.display_size, [2, 0], [2, 2], \
        self.ego_vehicle, self.target_pos.transform, \
        self.waypoints)
        self.display_manager.add_birdeyeview(self.bev)
        
        self.collision = SensorManager(self.world, 'Collision', \
        carla.Transform(), self.ego_vehicle, {}, None, None)
        self.sensor_list.append(self.collision)
        
        self.lane_invasion = SensorManager(self.world, 'Lane_invasion', \
        carla.Transform(), self.ego_vehicle, {}, None, None)
        self.sensor_list.append(self.lane_invasion)

        transform = self.ego_vehicle.get_transform()
        transform.location.z += 50#10
        transform.rotation.pitch -= 90
        self.spectator.set_transform(transform)
        
        # create other vehicles
        vehicle_bps = self.world.get_blueprint_library().\
        filter('vehicle.*')
        vehicle_bps_4wheel = []
        vehicle_bps_4wheel = \
        vehicle_bps_4wheel + [x for x in vehicle_bps if int(x.get_attribute('number_of_wheels')) == 4]
        for vehicle_bp in vehicle_bps_4wheel:
            vehicle_bp.set_attribute('role_name', 'autopilot')
        for _ in range(self.num_vehicles):
            vehicle_tmp_ref = None
            while vehicle_tmp_ref is None:
                vehicle_tmp_ref = \
                self.world.try_spawn_actor(random.choice(vehicle_bps_4wheel), \
                random.choice(self.spawn_points))
            vehicle_tmp_ref.set_autopilot()
            self.vehicle_list.append(vehicle_tmp_ref)
        
        # create pedestrians
        pedestrian_bps = self.world.get_blueprint_library().\
        filter('walker.*')
        for pedestrian_bp in pedestrian_bps:
            if pedestrian_bp.has_attribute('is_invincible'):
                pedestrian_bp.set_attribute('is_invincible','false')
        for _ in range(self.num_pedestrians): 
            pedestrian_tmp_ref = None
            while pedestrian_tmp_ref is None:
                pedestrian_spawn_transform = carla.Transform()  
                loc = self.world.get_random_location_from_navigation()  
                if (loc != None):
                    pedestrian_spawn_transform.location = loc 
                pedestrian_tmp_ref = \
                self.world.try_spawn_actor(\
                random.choice(pedestrian_bps), \
                pedestrian_spawn_transform)
            pedestrian_controller_bp = \
            self.world.get_blueprint_library().\
            find('controller.ai.walker')
            pedestrian_controller_actor = \
            self.world.spawn_actor(pedestrian_controller_bp, \
            carla.Transform(), pedestrian_tmp_ref)
            pedestrian_controller_actor.start()
            pedestrian_controller_actor.go_to_location( \
            self.world.get_random_location_from_navigation())
            pedestrian_controller_actor.set_max_speed(1.0 + \
            random.random())
            self.pedestrian_list.append(pedestrian_tmp_ref)
            self.pedestrian_controller_list.append( \
            pedestrian_controller_actor)
            
    
    def remove_all_actors(self):
        if self.waypoints is not None:
            del self.waypoints
        if self.target_pos is not None:
            del self.target_pos
        for s in self.sensor_list:
            s.destroy_sensor()
            del s
        self.sensor_list = []
        for v in self.vehicle_list:
            if v.is_alive:
                v.destroy()
            del v
        if self.ego_vehicle is not None:
            del self.ego_vehicle
        self.vehicle_list = []
        for c in self.pedestrian_controller_list:
            if c.is_alive:
                c.stop()
                c.destroy()
            del c
        self.pedestrian_controller_list = []
        for p in self.pedestrian_list:
            if p.is_alive:
                p.destroy()
            del p
        self.pedestrian_list = []
        time.sleep(0.1)
        if self.bev is not None:
            self.bev.destroy()
            del self.bev

        self.display_manager.clear()
    
    def display(self):
        self.display_manager.render()
        
    def pid_sample(self):
        act=[0.0,0.0]
        pid = VehiclePIDController(self.ego_vehicle, \
        {'K_P': 0.5, 'K_I': 0.2, 'K_D': 0.01, 'dt': 0.1},\
        {'K_P': 0.15, 'K_I': 0.07, 'K_D': 0.05, 'dt': 0.1})
        if len(self.waypoints)>20:
            control = pid.run_step(20.0,self.waypoints[19][0])
        else:
            control = pid.run_step(20.0,self.waypoints[-1][0])
        control.steer += np.random.normal(0.0,0.05)
        if control.brake>0:
            act[0] = -control.brake
        else:
            act[0] = control.throttle
        act[1] = control.steer
        return act
    
