# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import glob
import os
import sys
import numpy as np
import carla
from carla import TrafficLightState as tls

import argparse
import logging
import datetime
import weakref
import math
import random
import hashlib

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

# ==============================================================================
# -- Constants -----------------------------------------------------------------
# ==============================================================================

# Colors

# We will use the color palette used in Tango Desktop Project (Each color is indexed depending on brightness level)
# See: https://en.wikipedia.org/wiki/Tango_Desktop_Project

COLOR_BUTTER_0 = pygame.Color(252, 233, 79)
COLOR_BUTTER_1 = pygame.Color(237, 212, 0)
COLOR_BUTTER_2 = pygame.Color(196, 160, 0)

COLOR_ORANGE_0 = pygame.Color(252, 175, 62)
COLOR_ORANGE_1 = pygame.Color(245, 121, 0)
COLOR_ORANGE_2 = pygame.Color(209, 92, 0)

COLOR_CHOCOLATE_0 = pygame.Color(233, 185, 110)
COLOR_CHOCOLATE_1 = pygame.Color(193, 125, 17)
COLOR_CHOCOLATE_2 = pygame.Color(143, 89, 2)

COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
COLOR_CHAMELEON_1 = pygame.Color(115, 210, 22)
COLOR_CHAMELEON_2 = pygame.Color(78, 154, 6)

COLOR_SKY_BLUE_0 = pygame.Color(114, 159, 207)
COLOR_SKY_BLUE_1 = pygame.Color(52, 101, 164)
COLOR_SKY_BLUE_2 = pygame.Color(32, 74, 135)

COLOR_PLUM_0 = pygame.Color(173, 127, 168)
COLOR_PLUM_1 = pygame.Color(117, 80, 123)
COLOR_PLUM_2 = pygame.Color(92, 53, 102)

COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)
COLOR_SCARLET_RED_2 = pygame.Color(164, 0, 0)

COLOR_ALUMINIUM_0 = pygame.Color(238, 238, 236)
COLOR_ALUMINIUM_1 = pygame.Color(211, 215, 207)
COLOR_ALUMINIUM_2 = pygame.Color(186, 189, 182)
COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)
COLOR_ALUMINIUM_4 = pygame.Color(85, 87, 83)
COLOR_ALUMINIUM_4_5 = pygame.Color(66, 62, 64)
COLOR_ALUMINIUM_5 = pygame.Color(46, 52, 54)

COLOR_CYAN = pygame.Color(0,255,255)

COLOR_RED   = pygame.Color(255, 0, 0)
COLOR_GREEN = pygame.Color(0, 255, 0)
COLOR_BLUE  = pygame.Color(0, 0, 255)
COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)

# Macro Defines
PIXELS_PER_METER = 12
PIXELS_AHEAD_VEHICLE = 100#150


# ==============================================================================
# -- Util -----------------------------------------------------------
# ==============================================================================


class Util(object):

    @staticmethod
    def blits(destination_surface, source_surfaces, rect=None, blend_mode=0):
        """Function that renders the all the source surfaces in a destination source"""
        for surface in source_surfaces:
            destination_surface.blit(surface[0], surface[1], rect, blend_mode)

    @staticmethod
    def length(v):
        """Returns the length of a vector"""
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)

    @staticmethod
    def get_bounding_box(actor):
        """Gets the bounding box corners of an actor in world space"""
        bb = actor.trigger_volume.extent
        corners = [carla.Location(x=-bb.x, y=-bb.y),
                   carla.Location(x=bb.x, y=-bb.y),
                   carla.Location(x=bb.x, y=bb.y),
                   carla.Location(x=-bb.x, y=bb.y),
                   carla.Location(x=-bb.x, y=-bb.y)]
        corners = [x + actor.trigger_volume.location for x in corners]
        t = actor.get_transform()
        t.transform(corners)
        return corners


# ==============================================================================
# -- TrafficLightSurfaces ------------------------------------------------------
# ==============================================================================


class TrafficLightSurfaces(object):
    """Holds the surfaces (scaled and rotated) for painting traffic lights"""

    def __init__(self):
        def make_surface(tl):
            """Draws a traffic light, which is composed of a dark background surface with 3 circles that indicate its color depending on the state"""
            w = 40
            surface = pygame.Surface((w, 3 * w), pygame.SRCALPHA)
            surface.fill(COLOR_ALUMINIUM_5 if tl != 'h' else COLOR_ORANGE_2)
            if tl != 'h':
                hw = int(w / 2)
                off = COLOR_ALUMINIUM_4
                red = COLOR_SCARLET_RED_0
                yellow = COLOR_BUTTER_0
                green = COLOR_CHAMELEON_0

                # Draws the corresponding color if is on, otherwise it will be gray if its off
                pygame.draw.circle(surface, red if tl == tls.Red else off, (hw, hw), int(0.4 * w))
                pygame.draw.circle(surface, yellow if tl == tls.Yellow else off, (hw, w + hw), int(0.4 * w))
                pygame.draw.circle(surface, green if tl == tls.Green else off, (hw, 2 * w + hw), int(0.4 * w))

            return pygame.transform.smoothscale(surface, (15, 45) if tl != 'h' else (19, 49))

        self._original_surfaces = {
            'h': make_surface('h'),
            tls.Red: make_surface(tls.Red),
            tls.Yellow: make_surface(tls.Yellow),
            tls.Green: make_surface(tls.Green),
            tls.Off: make_surface(tls.Off),
            tls.Unknown: make_surface(tls.Unknown)
        }
        self.surfaces = dict(self._original_surfaces)

    def rotozoom(self, angle, scale):
        """Rotates and scales the traffic light surface"""
        for key, surface in self._original_surfaces.items():
            self.surfaces[key] = pygame.transform.rotozoom(surface, angle, scale)
            
            
# ==============================================================================
# -- MapImage ---------------------------------------------------------------------
# ==============================================================================


class MapImage(object):
    """Class encharged of rendering a 2D image from top view of a carla world. Please note that a cache system is used, so if the OpenDrive content
    of a Carla town has not changed, it will read and use the stored image if it was rendered in a previous execution"""

    def __init__(self, carla_world, carla_map, pixels_per_meter, \
    show_triggers=False, show_connections=False, show_spawn_points=False):
        """ Renders the map image generated based on the world, its map and additional flags that provide extra information about the road network"""
        self._pixels_per_meter = pixels_per_meter
        self.scale = 1.0
        self.show_triggers = show_triggers
        self.show_connections = show_connections
        self.show_spawn_points = show_spawn_points

        waypoints = carla_map.generate_waypoints(2)
        margin = 10
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        self.width = max(max_x - min_x, max_y - min_y)
        self.world_offset = (min_x, min_y-20)

        # Maximum size of a Pygame surface
        width_in_pixels = (1 << 14) - 1

        # Adapt Pixels per meter to make world fit in surface
        surface_pixel_per_meter = int(width_in_pixels / self.width)
        if surface_pixel_per_meter > PIXELS_PER_METER:
            surface_pixel_per_meter = PIXELS_PER_METER

        self._pixels_per_meter = surface_pixel_per_meter
        width_in_pixels = int(self._pixels_per_meter * self.width)

        self.big_map_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()

        # Load OpenDrive content
        opendrive_content = carla_map.to_opendrive()

        # Get hash based on content
        hash_func = hashlib.sha1()
        hash_func.update(opendrive_content.encode("UTF-8"))
        opendrive_hash = str(hash_func.hexdigest())

        # Build path for saving or loading the cached rendered map
        filename = carla_map.name.split('/')[-1] + "_" + opendrive_hash + ".tga"
        dirname = os.path.join("cache", "no_rendering_mode")
        full_path = str(os.path.join(dirname, filename))

        if os.path.isfile(full_path):
            # Load Image
            self.big_map_surface = pygame.image.load(full_path)
        else:
            # Render map
            self.draw_road_map(
                self.big_map_surface,
                carla_world,
                carla_map,
                self.world_to_pixel,
                self.world_to_pixel_width)

            # If folders path does not exist, create it
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            # Remove files if selected town had a previous version saved
            list_filenames = glob.glob(os.path.join(dirname, carla_map.name) + "*")
            for town_filename in list_filenames:
                os.remove(town_filename)

            # Save rendered map for next executions of same map
            pygame.image.save(self.big_map_surface, full_path)

        self.surface = self.big_map_surface

    def draw_road_map(self, map_surface, carla_world, carla_map, world_to_pixel, world_to_pixel_width):
        """Draws all the roads, including lane markings, arrows and traffic signs"""
        map_surface.fill(COLOR_ALUMINIUM_4)#COLOR_ALUMINIUM_4
        precision = 0.05

        def lane_marking_color_to_tango(lane_marking_color):
            """Maps the lane marking color enum specified in PythonAPI to a Tango Color"""
            tango_color = COLOR_BLACK

            if lane_marking_color == carla.LaneMarkingColor.White:
                tango_color = COLOR_ALUMINIUM_2

            elif lane_marking_color == carla.LaneMarkingColor.Blue:
                tango_color = COLOR_SKY_BLUE_0

            elif lane_marking_color == carla.LaneMarkingColor.Green:
                tango_color = COLOR_CHAMELEON_0

            elif lane_marking_color == carla.LaneMarkingColor.Red:
                tango_color = COLOR_SCARLET_RED_0

            elif lane_marking_color == carla.LaneMarkingColor.Yellow:
                tango_color = COLOR_ORANGE_0

            return tango_color

        def draw_solid_line(surface, color, closed, points, width):
            """Draws solid lines in a surface given a set of points, width and color"""
            if len(points) >= 2:
                pygame.draw.lines(surface, color, closed, points, width)

        def draw_broken_line(surface, color, closed, points, width):
            """Draws broken lines in a surface given a set of points, width and color"""
            # Select which lines are going to be rendered from the set of lines
            broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]

            # Draw selected lines
            for line in broken_lines:
                pygame.draw.lines(surface, color, closed, line, width)

        def get_lane_markings(lane_marking_type, lane_marking_color, waypoints, sign):
            """For multiple lane marking types (SolidSolid, BrokenSolid, SolidBroken and BrokenBroken), it converts them
             as a combination of Broken and Solid lines"""
            margin = 0.0
            marking_1 = [world_to_pixel(lateral_shift(w.transform, sign * w.lane_width * 0.5)) for w in waypoints]
            if lane_marking_type == carla.LaneMarkingType.Broken or (lane_marking_type == carla.LaneMarkingType.Solid):
                return [(lane_marking_type, lane_marking_color, marking_1)]
            else:
                marking_2 = [world_to_pixel(lateral_shift(w.transform,
                                                          sign * (w.lane_width * 0.5 + margin * 2))) for w in waypoints]
                if lane_marking_type == carla.LaneMarkingType.SolidBroken:
                    return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]
                elif lane_marking_type == carla.LaneMarkingType.BrokenSolid:
                    return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
                elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
                    return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
                elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
                    return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]

            return [(carla.LaneMarkingType.NONE, carla.LaneMarkingColor.Other, [])]

        def draw_lane(surface, lane, color):
            """Renders a single lane in a surface and with a specified color"""
            for side in lane:
                lane_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in side]
                lane_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in side]

                polygon = lane_left_side + [x for x in reversed(lane_right_side)]
                polygon = [world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    pygame.draw.polygon(surface, color, polygon, 5)
                    pygame.draw.polygon(surface, color, polygon)

        def draw_lane_marking(surface, waypoints):
            """Draws the left and right side of lane markings"""
            # Left Side
            draw_lane_marking_single_side(surface, waypoints[0], -1)

            # Right Side
            draw_lane_marking_single_side(surface, waypoints[1], 1)

        def draw_lane_marking_single_side(surface, waypoints, sign):
            """Draws the lane marking given a set of waypoints and decides whether drawing the right or left side of
            the waypoint based on the sign parameter"""
            lane_marking = None

            marking_type = carla.LaneMarkingType.NONE
            previous_marking_type = carla.LaneMarkingType.NONE

            marking_color = carla.LaneMarkingColor.Other
            previous_marking_color = carla.LaneMarkingColor.Other

            markings_list = []
            temp_waypoints = []
            current_lane_marking = carla.LaneMarkingType.NONE
            for sample in waypoints:
                lane_marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking

                if lane_marking is None:
                    continue

                marking_type = lane_marking.type
                marking_color = lane_marking.color

                if current_lane_marking != marking_type:
                    # Get the list of lane markings to draw
                    markings = get_lane_markings(
                        previous_marking_type,
                        lane_marking_color_to_tango(previous_marking_color),
                        temp_waypoints,
                        sign)
                    current_lane_marking = marking_type

                    # Append each lane marking in the list
                    for marking in markings:
                        markings_list.append(marking)

                    temp_waypoints = temp_waypoints[-1:]

                else:
                    temp_waypoints.append((sample))
                    previous_marking_type = marking_type
                    previous_marking_color = marking_color

            # Add last marking
            last_markings = get_lane_markings(
                previous_marking_type,
                lane_marking_color_to_tango(previous_marking_color),
                temp_waypoints,
                sign)
            for marking in last_markings:
                markings_list.append(marking)

            # Once the lane markings have been simplified to Solid or Broken lines, we draw them
            for markings in markings_list:
                if markings[0] == carla.LaneMarkingType.Solid:
                    draw_solid_line(surface, markings[1], False, markings[2], 2)
                elif markings[0] == carla.LaneMarkingType.Broken:
                    draw_broken_line(surface, markings[1], False, markings[2], 2)

        def draw_arrow(surface, transform, color=COLOR_ALUMINIUM_2):
            """ Draws an arrow with a specified color given a transform"""
            transform.rotation.yaw += 180
            forward = transform.get_forward_vector()
            transform.rotation.yaw += 90
            right_dir = transform.get_forward_vector()
            end = transform.location
            start = end - 2.0 * forward
            right = start + 0.8 * forward + 0.4 * right_dir
            left = start + 0.8 * forward - 0.4 * right_dir

            # Draw lines
            pygame.draw.lines(surface, color, False, [world_to_pixel(x) for x in [start, end]], 4)
            pygame.draw.lines(surface, color, False, [world_to_pixel(x) for x in [left, start, right]], 4)

        def draw_traffic_signs(surface, font_surface, actor, color=COLOR_ALUMINIUM_2, trigger_color=COLOR_PLUM_0):
            """Draw stop traffic signs and its bounding box if enabled"""
            transform = actor.get_transform()
            waypoint = carla_map.get_waypoint(transform.location)

            angle = -waypoint.transform.rotation.yaw - 90.0
            font_surface = pygame.transform.rotate(font_surface, angle)
            pixel_pos = world_to_pixel(waypoint.transform.location)
            offset = font_surface.get_rect(center=(pixel_pos[0], pixel_pos[1]))
            surface.blit(font_surface, offset)

            # Draw line in front of stop
            forward_vector = carla.Location(waypoint.transform.get_forward_vector())
            left_vector = carla.Location(-forward_vector.y, forward_vector.x,
                                         forward_vector.z) * waypoint.lane_width / 2 * 0.7

            line = [(waypoint.transform.location + (forward_vector * 1.5) + (left_vector)),
                    (waypoint.transform.location + (forward_vector * 1.5) - (left_vector))]

            line_pixel = [world_to_pixel(p) for p in line]
            pygame.draw.lines(surface, color, True, line_pixel, 2)

            # Draw bounding box of the stop trigger
            if self.show_triggers:
                corners = Util.get_bounding_box(actor)
                corners = [world_to_pixel(p) for p in corners]
                pygame.draw.lines(surface, trigger_color, True, corners, 2)

        # def draw_crosswalk(surface, transform=None, color=COLOR_ALUMINIUM_2):
        #     """Given two points A and B, draw white parallel lines from A to B"""
        #     a = carla.Location(0.0, 0.0, 0.0)
        #     b = carla.Location(10.0, 10.0, 0.0)

        #     ab = b - a
        #     length_ab = math.sqrt(ab.x**2 + ab.y**2)
        #     unit_ab = ab / length_ab
        #     unit_perp_ab = carla.Location(-unit_ab.y, unit_ab.x, 0.0)

        #     # Crosswalk lines params
        #     space_between_lines = 0.5
        #     line_width = 0.7
        #     line_height = 2

        #     current_length = 0
        #     while current_length < length_ab:

        #         center = a + unit_ab * current_length

        #         width_offset = unit_ab * line_width
        #         height_offset = unit_perp_ab * line_height
        #         list_point = [center - width_offset - height_offset,
        #                       center + width_offset - height_offset,
        #                       center + width_offset + height_offset,
        #                       center - width_offset + height_offset]

        #         list_point = [world_to_pixel(p) for p in list_point]
        #         pygame.draw.polygon(surface, color, list_point)
        #         current_length += (line_width + space_between_lines) * 2

        def lateral_shift(transform, shift):
            """Makes a lateral shift of the forward vector of a transform"""
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def draw_topology(carla_topology, index):
            """ Draws traffic signs and the roads network with sidewalks, parking and shoulders by generating waypoints"""
            topology = [x[index] for x in carla_topology]
            topology = sorted(topology, key=lambda w: w.transform.location.z)
            set_waypoints = []
            for waypoint in topology:
                waypoints = [waypoint]

                # Generate waypoints of a road id. Stop when road id differs
                nxt = waypoint.next(precision)
                if len(nxt) > 0:
                    nxt = nxt[0]
                    while nxt.road_id == waypoint.road_id:
                        waypoints.append(nxt)
                        nxt = nxt.next(precision)
                        if len(nxt) > 0:
                            nxt = nxt[0]
                        else:
                            break
                set_waypoints.append(waypoints)

                # Draw Shoulders, Parkings and Sidewalks
                PARKING_COLOR  = pygame.Color(0,0,255)#COLOR_ALUMINIUM_4_5
                SHOULDER_COLOR = COLOR_ALUMINIUM_5
                SIDEWALK_COLOR = COLOR_ALUMINIUM_3

                shoulder = [[], []]
                parking = [[], []]
                sidewalk = [[], []]

                for w in waypoints:
                    # Classify lane types until there are no waypoints by going left
                    l = w.get_left_lane()
                    while l and l.lane_type != carla.LaneType.Driving:

                        if l.lane_type == carla.LaneType.Shoulder:
                            shoulder[0].append(l)

                        if l.lane_type == carla.LaneType.Parking:
                            parking[0].append(l)

                        if l.lane_type == carla.LaneType.Sidewalk:
                            sidewalk[0].append(l)

                        l = l.get_left_lane()

                    # Classify lane types until there are no waypoints by going right
                    r = w.get_right_lane()
                    while r and r.lane_type != carla.LaneType.Driving:

                        if r.lane_type == carla.LaneType.Shoulder:
                            shoulder[1].append(r)

                        if r.lane_type == carla.LaneType.Parking:
                            parking[1].append(r)

                        if r.lane_type == carla.LaneType.Sidewalk:
                            sidewalk[1].append(r)

                        r = r.get_right_lane()

                # Draw classified lane types
                draw_lane(map_surface, shoulder, SHOULDER_COLOR)
                draw_lane(map_surface, parking, PARKING_COLOR)
                draw_lane(map_surface, sidewalk, SIDEWALK_COLOR)

            # Draw Roads
            for waypoints in set_waypoints:
                waypoint = waypoints[0]
                road_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
                road_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

                polygon = road_left_side + [x for x in reversed(road_right_side)]
                polygon = [world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon, 5)
                    pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon)

                # Draw Lane Markings and Arrows
                if not waypoint.is_junction:
                    draw_lane_marking(map_surface, [waypoints, waypoints])
                    #for n, wp in enumerate(waypoints):
                    #    if ((n + 1) % 400) == 0:
                    #        draw_arrow(map_surface, wp.transform)

        topology = carla_map.get_topology()
        draw_topology(topology, 0)

        if self.show_spawn_points:
            for sp in carla_map.get_spawn_points():
                draw_arrow(map_surface, sp, color=COLOR_CHOCOLATE_0)

        if self.show_connections:
            dist = 1.5

            def to_pixel(wp): return world_to_pixel(wp.transform.location)
            for wp in carla_map.generate_waypoints(dist):
                col = (0, 255, 255) if wp.is_junction else (0, 255, 0)
                for nxt in wp.next(dist):
                    pygame.draw.line(map_surface, col, to_pixel(wp), to_pixel(nxt), 2)
                if wp.lane_change & carla.LaneChange.Right:
                    r = wp.get_right_lane()
                    if r and r.lane_type == carla.LaneType.Driving:
                        pygame.draw.line(map_surface, col, to_pixel(wp), to_pixel(r), 2)
                if wp.lane_change & carla.LaneChange.Left:
                    l = wp.get_left_lane()
                    if l and l.lane_type == carla.LaneType.Driving:
                        pygame.draw.line(map_surface, col, to_pixel(wp), to_pixel(l), 2)

        actors = carla_world.get_actors()

        # Find and Draw Traffic Signs: Stops and Yields
        font_size = world_to_pixel_width(1)
        font = pygame.font.SysFont('Arial', font_size, True)

        stops = [actor for actor in actors if 'stop' in actor.type_id]
        yields = [actor for actor in actors if 'yield' in actor.type_id]

        stop_font_surface = font.render("STOP", False, COLOR_ALUMINIUM_2)
        stop_font_surface = pygame.transform.scale(
            stop_font_surface, (stop_font_surface.get_width(), stop_font_surface.get_height() * 2))

        yield_font_surface = font.render("YIELD", False, COLOR_ALUMINIUM_2)
        yield_font_surface = pygame.transform.scale(
            yield_font_surface, (yield_font_surface.get_width(), yield_font_surface.get_height() * 2))

        for ts_stop in stops:
            draw_traffic_signs(map_surface, stop_font_surface, ts_stop, trigger_color=COLOR_SCARLET_RED_1)

        for ts_yield in yields:
            draw_traffic_signs(map_surface, yield_font_surface, ts_yield, trigger_color=COLOR_ORANGE_1)

    def world_to_pixel(self, location, offset=(0, 0)):
        """Converts the world coordinates to pixel coordinates"""
        x = self.scale * self._pixels_per_meter * (location.x - self.world_offset[0])
        y = self.scale * self._pixels_per_meter * (location.y - self.world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]

    def world_to_pixel_width(self, width):
        """Converts the world units to pixel units"""
        return int(self.scale * self._pixels_per_meter * width)

    def scale_map(self, scale):
        """Scales the map surface"""
        if scale != self.scale:
            self.scale = scale
            width = int(self.big_map_surface.get_width() * self.scale)
            self.surface = pygame.transform.smoothscale(self.big_map_surface, (width, width))
            
            
# ==============================================================================
# -- BirdEyeView ----------------------------------------------------------------
# ==============================================================================


class BirdEyeView(object):
    def __init__(self, world, pixels_per_meter, pixels_ahead_vehicle, \
    display_size, display_pos, display_pos_global, hero_actor, \
    target_transform, waypoints = None):
        # resource from outside
        self.world = world
        self.pixels_per_meter = pixels_per_meter
        self.pixels_ahead_vehicle = pixels_ahead_vehicle
        self.display_size = display_size
        self.display_pos = display_pos
        self.display_pos_global = display_pos_global
        self.hero_actor = hero_actor
        self.target_transform = target_transform
        self.waypoints = waypoints
        
        # self created resource
        self.server_clock = pygame.time.Clock()
        self.surface = pygame.Surface(display_size).convert()
        self.surface.set_colorkey(COLOR_BLACK)
        self.surface_global = pygame.Surface(display_size).convert()
        self.surface_global.set_colorkey(COLOR_BLACK)
        self.measure_data = np.zeros((display_size[0],display_size[1],3), dtype=np.uint8)
        
        # World data
        self.town_map = self.world.get_map()
        self.actors_with_transforms = []
        
        # Hero actor
        if self.hero_actor is not None:
            self.hero_id = self.hero_actor.id
            self.hero_transform = self.hero_actor.get_transform()
        else:
            self.hero_id = None
            self.hero_transform = None
            
        # Create Surfaces
        self.map_image = MapImage(self.world, self.town_map, \
        self.pixels_per_meter)
        
        self.original_surface_size = min(self.display_size[0], \
        display_size[1])
        self.surface_size = self.map_image.big_map_surface.get_width()
        
        # Render Actors
        self.actors_surface = pygame.Surface( \
        (self.map_image.surface.get_width(), \
        self.map_image.surface.get_height()))
        self.actors_surface.set_colorkey(COLOR_BLACK)
        
        self.waypoints_surface = pygame.Surface( \
        (self.map_image.surface.get_width(), \
        self.map_image.surface.get_height()))
        self.waypoints_surface.set_colorkey(COLOR_BLACK)
        
        scaled_original_size = self.original_surface_size * (1.0 / 0.7)
        self.hero_surface = pygame.Surface((scaled_original_size, \
        scaled_original_size)).convert()
        
        self.result_surface = pygame.Surface((self.surface_size, \
        self.surface_size)).convert()
        self.result_surface.set_colorkey(COLOR_BLACK)
        
        self.traffic_light_surfaces = TrafficLightSurfaces()
        
        weak_self = weakref.ref(self)
        self.world.on_tick(lambda timestamp: BirdEyeView.on_world_tick(\
        weak_self, timestamp))
    
    def destroy(self):
        del self.server_clock
        del self.surface
        del self.surface_global
        del self.measure_data
        del self.town_map
        del self.actors_with_transforms
        del self.hero_id
        del self.hero_transform
        del self.map_image
        del self.actors_surface
        del self.waypoints_surface
        del self.hero_surface
        del self.result_surface
        del self.traffic_light_surfaces            
           
    @staticmethod
    def on_world_tick(weak_self, timestamp):
        self = weak_self()
        if not self:
            return
        
        self.server_clock.tick()
        
    def set_hero(self, hero_actor, hero_id):
        self.hero_actor = hero_actor
        self.hero_id = hero_id
        
    def tick(self, clock):
        actors = self.world.get_actors()
        # We store the transforms also so that we avoid having transforms of previous tick and current tick when rendering them.
        self.actors_with_transforms = [(actor, actor.get_transform()) \
        for actor in actors]
        if self.hero_actor is not None:
            self.hero_transform = self.hero_actor.get_transform()
    
    def split_actors(self):
        vehicles = []
        traffic_lights = []
        speed_limits = []
        walkers = []
        
        for actor_with_transform in self.actors_with_transforms:
            actor = actor_with_transform[0]
            if 'vehicle' in actor.type_id:
                vehicles.append(actor_with_transform)
            elif 'traffic_light' in actor.type_id:
                traffic_lights.append(actor_with_transform)
            elif 'speed_limit' in actor.type_id:
                speed_limits.append(actor_with_transform)
            elif 'walker.pedestrian' in actor.type_id:
                walkers.append(actor_with_transform)
            
        return (vehicles, traffic_lights, speed_limits, walkers)
    
    def render_traffic_lights(self, surface, list_tl, \
    show_triggers, world_to_pixel):
        self.affected_traffic_light = None
        
        for tl in list_tl:
            world_pos = tl.get_location()
            pos = world_to_pixel(world_pos)
            
            if show_triggers:
                corners = Util.get_bounding_box(tl)
                corners = [world_to_pixel(p) for p in corners]
                pygame.draw.lines(surface, COLOR_BUTTER_1, \
                True, corners, 2)
            
            if self.hero_actor is not None:
                corners = Util.get_bounding_box(tl)
                corners = [world_to_pixel(p) for p in corners]
                tl_t = tl.get_transform()
                
                transformed_tv = \
                tl_t.transform(tl.trigger_volume.location)
                hero_location = self.hero_actor.get_location()
                d = hero_location.distance(transformed_tv)
                s = Util.length(tl.trigger_volume.extent) + \
                Util.length(self.hero_actor.bounding_box.extent)
                if (d <= s):
                    # Highlight traffic light
                    self.affected_traffic_light = tl
                    srf = self.traffic_light_surfaces.surfaces['h']
                    surface.blit(srf, srf.get_rect(center=pos))
                    
            srf = self.traffic_light_surfaces.surfaces[tl.state]
            surface.blit(srf, srf.get_rect(center=pos))
    
    def render_speed_limits(self, surface, list_sl, show_triggers, \
    world_to_pixel, world_to_pixel_width):
        font_size = world_to_pixel_width(2)
        radius = world_to_pixel_width(2)
        font = pygame.font.SysFont('Arial', font_size)
        
        for sl in list_sl:
            x, y = world_to_pixel(sl.get_location())
            
            # Render speed limit concentric circles
            white_circle_radius = int(radius * 0.75)
            
            pygame.draw.circle(surface, COLOR_SCARLET_RED_1, (x, y), \
            radius)
            pygame.draw.circle(surface, COLOR_ALUMINIUM_0, (x, y), \
            white_circle_radius)
            
            limit = sl.type_id.split('.')[2]
            font_surface = font.render(limit, True, COLOR_ALUMINIUM_5)
            
            if show_triggers:
                corners = Util.get_bounding_box(sl)
                corners = [world_to_pixel(p) for p in corners]
                pygame.draw.lines(surface, COLOR_PLUM_2, True, \
                corners, 2)
                
            # Blit
            if self.hero_actor is not None:
                angle = -self.hero_transform.rotation.yaw - 90.0
                font_surface = pygame.transform.rotate(font_surface, \
                angle)
                offset = font_surface.get_rect(center=(x, y))
                surface.blit(font_surface, offset)
    
    def render_walkers(self, surface, list_w, world_to_pixel):
        for w in list_w:
            color = COLOR_RED#COLOR_PLUM_0
            
            # Compute bounding box points
            bb = w[0].bounding_box.extent
            corners = [carla.Location(x=-bb.x, y=-bb.y), \
            carla.Location(x=bb.x, y=-bb.y), \
            carla.Location(x=bb.x, y=bb.y), \
            carla.Location(x=-bb.x, y=bb.y)]
                
            w[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.polygon(surface, color, corners)
            
    def render_vehicles(self, surface, list_v, world_to_pixel):
        for v in list_v:
            color = COLOR_SCARLET_RED_1#COLOR_SKY_BLUE_0
            if int(v[0].attributes['number_of_wheels']) == 2:
                color = COLOR_SCARLET_RED_2#COLOR_CHOCOLATE_1
                bb = v[0].bounding_box.extent
                corners = [carla.Location(x=-bb.x, y=-bb.y-0.1), \
                carla.Location(x=bb.x, y=-bb.y-0.1), \
                carla.Location(x=bb.x + 0.3, y=0), \
                carla.Location(x=bb.x, y=bb.y+0.1), \
                carla.Location(x=-bb.x, y=bb.y+0.1), \
                carla.Location(x=-bb.x, y=-bb.y-0.1)]
            else:
                if v[0].attributes['role_name'] == 'hero':
                    color = COLOR_BLUE#COLOR_CHAMELEON_0
                # Compute bounding box points
                bb = v[0].bounding_box.extent
                corners = [carla.Location(x=-bb.x, y=-bb.y), \
                carla.Location(x=bb.x, y=-bb.y), \
                carla.Location(x=bb.x + 0.3, y=0), \
                carla.Location(x=bb.x, y=bb.y), \
                carla.Location(x=-bb.x, y=bb.y), \
                carla.Location(x=-bb.x, y=-bb.y)]
            
            v[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            #pygame.draw.lines(surface, color, False, corners, \
            #int(math.ceil(4.0 * self.map_image.scale)))
            pygame.draw.polygon(surface, color, corners)
            
    def render_points(self, surface, colors, transform, radius_in_pix, world_to_pixel):
        location = world_to_pixel(transform.location)
        pygame.draw.circle(surface, colors[0], location, radius_in_pix, 0)
        location = world_to_pixel(self.hero_actor.get_location())
        pygame.draw.circle(surface, colors[1], location, radius_in_pix, 0)
            
    def render_actors(self, surface, vehicles, traffic_lights, speed_limits, walkers):
        # Static actors
        #self.render_traffic_lights(surface, \
        #[tl[0] for tl in traffic_lights], \
        #self.map_image.show_triggers, \
        #self.map_image.world_to_pixel)
        
        #self.render_speed_limits(surface, \
        #[sl[0] for sl in speed_limits], \
        #self.map_image.show_triggers, \
        #self.map_image.world_to_pixel, \
        #self.map_image.world_to_pixel_width)
        
        # Dynamic actors
        self.render_vehicles(surface, vehicles, \
        self.map_image.world_to_pixel)
        self.render_walkers(surface, walkers, \
        self.map_image.world_to_pixel)
        
        self.render_points(surface, (COLOR_GREEN,COLOR_BLUE), \
        self.target_transform, 10, \
        self.map_image.world_to_pixel)
    
    def render_waypoints(self, surface, waypoints, world_to_pixel):
        #print("rendering waypoints")
        color = COLOR_CYAN
        corners = []
        for p in waypoints:
            corners.append(carla.Location(x=p[0].transform.location.x, \
            y=p[0].transform.location.y))
        corners = [world_to_pixel(p) for p in corners]
        for c in corners:
            pygame.draw.circle(surface, color, c, \
            self.pixels_per_meter * 1.5)
        
    def clip_surfaces(self, clipping_rect):
        self.actors_surface.set_clip(clipping_rect)
        self.result_surface.set_clip(clipping_rect)
        
    def update_bird_eye_view(self):
        # clock tick
        self.tick(self.server_clock)
        
        if self.actors_with_transforms is None:
            return
        self.result_surface.fill(COLOR_BLACK)
        
        # Split the actors by vehicle type id
        vehicles, traffic_lights, speed_limits, walkers = \
        self.split_actors()
        
        # No zoom in and out
        scale_factor = 1.0
        
        # Render Waypoints
        self.waypoints_surface.fill(COLOR_BLACK)
        if self.waypoints is not None:
            self.render_waypoints(self.waypoints_surface, \
            self.waypoints, self.map_image.world_to_pixel)
        
        # Render Actors
        self.actors_surface.fill(COLOR_BLACK)
        self.render_actors(self.actors_surface, vehicles, \
        traffic_lights, speed_limits, walkers)

        # Blit surfaces
        surfaces = ((self.map_image.surface, (0, 0)), \
        (self.waypoints_surface, (0, 0)), \
        (self.actors_surface, (0, 0)))
        
        angle = 0.0 if self.hero_actor is None else \
        self.hero_transform.rotation.yaw + 90.0
        
        center_offset = (0, 0)
        
        hero_location_screen = \
        self.map_image.world_to_pixel(self.hero_transform.location)
        hero_front = self.hero_transform.get_forward_vector()
        translation_offset = (hero_location_screen[0] - \
        self.hero_surface.get_width() / 2 + \
        hero_front.x * self.pixels_ahead_vehicle, \
        hero_location_screen[1] - \
        self.hero_surface.get_height() / 2 + \
        hero_front.y * self.pixels_ahead_vehicle)
        
        # Apply clipping rect
        clipping_rect = pygame.Rect(translation_offset[0], \
        translation_offset[1], \
        self.hero_surface.get_width(), \
        self.hero_surface.get_height())
        #self.clip_surfaces(clipping_rect)
        
        Util.blits(self.result_surface, surfaces)
        
        # Set background black
        self.hero_surface.fill(COLOR_ALUMINIUM_4)
        self.hero_surface.blit(self.result_surface, \
        (-translation_offset[0], -translation_offset[1]))
        
        rotated_result_surface = \
        pygame.transform.rotate(self.hero_surface, angle).convert()
        
        center = (self.surface.get_width() / 2, \
        self.surface.get_height() / 2)
        rotation_pivot = rotated_result_surface.get_rect(center=center)
        self.surface.blit(rotated_result_surface, rotation_pivot)
        
        self.render_points(self.result_surface, (COLOR_GREEN,COLOR_BLUE), \
        self.target_transform, 100, self.map_image.world_to_pixel)
        self.surface_global.blit(pygame.transform.smoothscale(self.result_surface,self.display_size),(0,0))

        self.measure_data = \
        np.array(pygame.surfarray.array3d(self.surface)).swapaxes(0,1)
