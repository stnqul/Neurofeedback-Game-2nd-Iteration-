import pygame
import pygame.gfxdraw
import math
import numpy as np
import os
import atexit

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
# from scipy.fft import rfft, rfftfreq

from button import Button
from sensor import Sensor
from threading import Thread


# Initializations of some static variables:
SESSION_LENGTH = 900
BLINK_THRESHOLD = 5 * (10 ** -5) # 0.00005
ATTENTION_THRESHOLD = 1 * (10 ** -5)

WIN_WIDTH, WIN_HEIGHT = 800, 700 # !Original height: 800, Original width: 700
PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_VEL = 100, 20, 100
BALL_RADIUS, BALL_INITIAL_VEL = 10, 4 # !Original velocity: 4
MAX_DIFFICULTY, MAX_LIVES = 5, 5
FPS = 60
SAMPLE_FREQ = 250

def reading_task(mySensor: Sensor):
    """
    Calls the sensor reading method from the given sensor object.
    """
    print('Start reading the sensor: ')
    mySensor.read_sensor_Ts(SESSION_LENGTH)

class Game:
    def __init__(self):
        pygame.init()
        
        # Set the working directory to directory of the file:
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

        self.EEGSensor = Sensor()
        self.button_background_img = pygame.image.load('../images/UI_Flat_Frame_02_Horizontal.png')
        self.WIDTH, self.HEIGHT = WIN_WIDTH, WIN_HEIGHT
        self.win = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.screen = pygame.display.get_surface()
        pygame.display.set_caption("Brick Breaker")
        self.main_menu = True
        self.flicker_test = False
        self.connection_flag = False
        self.difficulty = 1

        self.FPS = FPS
        self.PADDLE_WIDTH = PADDLE_WIDTH
        self.PADDLE_HEIGHT = PADDLE_HEIGHT
        self.BALL_RADIUS = BALL_RADIUS

        self.LIVES_FONT = pygame.font.SysFont("comicsans", 40)
        self.MENU_TEXT_FONT = pygame.font.SysFont("comicsans", 20)
        self.GAUGE_FONT = pygame.font.SysFont("comicsans", 20)
        
        # Alternative combo: bg. azure & fl. aliceblue
        self.GAME_COLOR_BG = "ghostwhite"
        self.GAME_COLOR_FLICKER = "floralwhite"

        self.steps_needed = 0
        self.steps_taken = 0
        self.steps = 0
        self.direction = 0
        
        self.x, self.y = [], []
        fig_size = (8, 6)
        self.fig, self.ax = plt.subplots(1, 1, figsize = fig_size, dpi=80)
        self.current_graph_times = 0
        self.flicker_count = 0
        self.cross_width = 20
        self.cross_height = 120
        self.cross_x = WIN_WIDTH / 2 - self.cross_width / 2 - 1
        self.cross_y = WIN_HEIGHT / 2 - self.cross_height / 2 - 1
        self.cross_color = "gray"

        # Plotting (for debug purposes)
        game_occ_1_plot_file_name = "../plots/game_occ_1.txt"
        game_occ_2_plot_file_name = "../plots/game_occ_2.txt"
        
        os.makedirs(os.path.dirname(game_occ_1_plot_file_name), exist_ok=True)
        os.makedirs(os.path.dirname(game_occ_2_plot_file_name), exist_ok=True)
        self.game_occ_1_plot_file = open(game_occ_1_plot_file_name, 'w')
        self.game_occ_2_plot_file = open(game_occ_2_plot_file_name, 'w')
        
        self.game_occ_plot_time_counter = 0

        # Cleanup
        atexit.register(self.cleanup)
    
    def cleanup(self):
        self.game_occ_1_plot_file.close()

    def log_current_data_buffer(self, ys, hemisphere):
        for y in ys:
            if hemisphere == 'occ_1':
                self.game_occ_1_plot_file.write(f"{y} {self.game_occ_plot_time_counter}\n")
            elif hemisphere == 'occ_2':
                self.game_occ_2_plot_file.write(f"{y} {self.game_occ_plot_time_counter}\n")
            self.game_occ_plot_time_counter += 1

    def list_overlap_len(self, l1, l2):
        def list_overlap_len_rec(l1, l2, rel_last_idx):
            
            if rel_last_idx <= len(l2) - 1:
                l1_slice = l1[-(rel_last_idx + 1):]
                l2_slice = l2[:rel_last_idx + 1]
            elif rel_last_idx <= len(l1) - 1:
                l1_slice = l1[-(rel_last_idx + 1) : -(rel_last_idx + 1) + len(l2)]
                l2_slice = l2
            elif rel_last_idx <= len(l1) + len(l2) - 1:
                left_spill = rel_last_idx - len(l1) + 1
                slice_len = len(l2) - left_spill
                l1_slice = l1[:slice_len]
                l2_slice = l2[left_spill:]
            else:
                return 0
            
            if l1_slice == l2_slice:
                return len(l1_slice) 
            else:
                list_overlap_len_rec(l1, l2, rel_last_idx + 1)

        if len(l1) < len(l2):
            aux = l1
            l1 = l2
            l2 = aux
            
        return list_overlap_len_rec(l1,l2,0)

    def calculate_steps_needed(self, paddle, x_pred):
        """
        Calculates the amount of steps needed (integer) for the paddle to reach to the location that was predicted for the ball to get to at the bottom of the game window.
        Then assigns this value to the game object's field named steps_needed.
        This calculation is executed as soon as the ball touches either a brick or the top of the game window.
        """
        self.steps_needed = np.abs(np.floor(x_pred / PADDLE_VEL) - np.floor(paddle.x_initial / PADDLE_VEL))

    def calculate_steps_and_direction(self, paddle, x_pred):
            """
            Calculates the amount of steps "currently" needed (integer) for the paddle to reach to the location that was predicted for the ball to get to at the bottom of the game window.
            Also, finds the necessary horizontal direction for the paddle to go. Then assigns these values to the game object's fields named steps and direction.
            """
            steps = np.floor(x_pred / PADDLE_VEL) - np.floor(paddle.x / PADDLE_VEL)
            
            if steps > 0:
                self.direction = 1
            elif steps < 0:
                self.direction = -1
            else:
                self.direction = 0
            self.steps = np.abs(steps)

    class Paddle:
        def __init__(self, x, y, width, height, color):
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.color = color
            self.VEL = PADDLE_VEL
            self.has_moved = False

        def get_x(self):
            return self.x
        
        def get_y(self):
            return self.y
        
        def get_width(self):
            return self.width

        def get_height(self):
            return self.height

        def draw(self, win):
            """
            Draw method for paddle. Draws the object at each iteration of frame update.
            """
            pygame.draw.rect(
                win, self.color, (self.x, self.y, self.width, self.height))

        def move(self, direction=1):
            """
            Move method for paddle. Horizontally moves the paddle by self.VEL in the direction given by direction.
            """
            self.x = self.x + self.VEL * direction

        def move_to_final_location(self, x_pred, steps_needed, direction=1):
            """
            Move method for paddle. Horizontally moves the paddle in the direction given by direction to the final location
            in one step.
            """
            # if self.x != (x_pred - self.width / 2):
            #     self.x = x_pred - self.width / 2
            if not(self.has_moved):
                self.x = self.x + steps_needed * self.VEL * direction
                self.has_moved = True
        
        def reset_move_flag(self):
            self.has_moved = False


    class Ball:
        def __init__(self, x, y, radius, color, period):
            self.x = x
            self.y = y
            self.radius = radius
            self.color = color
            self.x_vel = 4
            self.y_vel = -BALL_INITIAL_VEL
            self.vel_scale = 1
            self.x_predict = -np.inf

            self.flicker_count = 0
            self.period = period

        def draw(self, win):
            """
            Draw method for ball. Draws the object at each iteration of frame update.
            """
            pygame.draw.circle(win, self.color, (self.x, self.y), self.radius)

        def move(self):
            """
            Move method for ball. Moves the ball by self.x_vel in x direction and self.y_vel in y direction. Additionally moves the ball in both directions w.r.t. current difficulty.
            """
            self.x += self.x_vel * ((self.vel_scale + MAX_DIFFICULTY) / MAX_DIFFICULTY)
            self.y += self.y_vel * ((self.vel_scale + MAX_DIFFICULTY) / MAX_DIFFICULTY)

        def set_vel_scale(self, new_scale):
            """
            Set method for vel_scale.
            """
            self.vel_scale = new_scale

        def set_vel(self, x_vel, y_vel):
            """
            Set method for x_vel and y_vel of ball.
            """
            self.x_vel = x_vel
            self.y_vel = y_vel

        def set_prediction(self, x_pred):
            """
            Set method for x_predict.
            """
            self.x_predict = x_pred

        def predict(self):
            """
            Predicts location of the ball in x-axis when it would reach the bottom. Then assigns the x_pred to the predicted value.
            """
            steps = ((WIN_HEIGHT - self.y - PADDLE_HEIGHT - BALL_RADIUS) / -self.y_vel) - 1
            x_pred = self.x + self.x_vel * steps
            if x_pred > WIN_WIDTH - BALL_RADIUS:
                x_pred = WIN_WIDTH - (x_pred - WIN_WIDTH) - 2 * BALL_RADIUS
            if x_pred < BALL_RADIUS:
                x_pred = -x_pred + 2 * BALL_RADIUS
            self.set_prediction(x_pred=x_pred)
            #print("Prediction: " + str(x_pred))


    class Brick:
        def __init__(self, x, y, width, height, health, colors):
            self.x = x
            self.y = y
            self.width = width
            self.height = height
            self.health = health
            self.max_health = health
            self.colors = colors
            self.color = colors[0]

        def draw(self, win):
            """
            Draw method for brick. Draws the object at each iteration of frame update.
            """
            pygame.draw.rect(
                win, self.color, (self.x, self.y, self.width, self.height))

        def collide(self, ball):
            """
            Checks whether the brick is collided with the ball. If collided returns true, otherwise returns false.

            :return: boolean
            """
            if not (ball.x <= self.x + self.width and ball.x >= self.x):
                return False
            if not (ball.y - ball.radius <= self.y + self.height):
                return False

            ball.predict()
            self.hit()
            ball.set_vel(ball.x_vel, ball.y_vel * -1)
            return True

        def hit(self):
            """
            Decreases the health by one if the ball hits the brick. Changes the color of the brick.
            """
            self.health -= 1
            self.color = self.interpolate(
                *self.colors, self.health/self.max_health)

        @staticmethod
        def interpolate(color_a, color_b, t):
            """
            Naive color interpolation method for the brick.
            """
            # t is contained in [0.0, 1.0]
            return tuple(int(a + (b - a) * t) for a, b in zip(color_a, color_b))


    class Gauge:
        def __init__(self, font, x, y, thickness, radius, arc_color):
            self.font = font
            self.x = x
            self.y = y
            self.thickness = thickness
            self.radius = radius
            self.arc_color = arc_color
            self.percentage = 0

        def get_x(self):
            return self.x

        def get_y(self):
            return self.y
        
        def get_thickness(self):
            return self.thickness

        def draw(self, win):
            """
            Draw method for gauge. Draws the object at each iteration of frame update.
            """
            fill_angle = int(self.percentage * 270 / 100)
            per = self.percentage
            if per <= 40:
                per = 0
            if per > 100:
                per = 100
            score = [int(255 - per * 255 / 100), int(per * 255 / 100), int(0), 255]
            for i in range(len(score)):
                if score[i] < 0:
                    score[i] = 0
                if score[i] > 255:
                    score[i] = 255
            per_text = self.font.render(str(self.percentage) + "%", True, score)
            per_text_rect = per_text.get_rect(center=(int(self.x), int(self.y)))
            win.blit(per_text, per_text_rect)
            for i in range(0, self.thickness):
                pygame.gfxdraw.arc(win, int(self.x), int(self.y), self.radius - i, -225, 270 - 225, self.arc_color)
                if self.percentage > 0:
                    pygame.gfxdraw.arc(win, int(self.x), int(self.y), self.radius - i, -225, fill_angle - 225, score)

        def change_percentage(self, value):
            """
            Adjusts the percentage of the performance for gauge.
            """
            if value != None:
                self.percentage = value
            else:
                self.percentage = 100
            if self.percentage > 100:
                self.percentage = 100
            if self.percentage < 0:
                self.percentage = 0


    class FlickerWindow:
        def __init__(self, flicker_on=True,
                           period=2,          flicker_on_period=None,
                           left_period=None,  left_flicker_on_period=None,
                           right_period=None, right_flicker_on_period=None):
            # Focus cross
            self.cross_width = 20
            self.cross_height = 120
            self.cross_x = WIN_WIDTH / 2 - self.cross_width / 2 - 1
            self.cross_y = WIN_HEIGHT / 2 - self.cross_height / 2 - 1
            self.horizontal_x = self.cross_x - (self.cross_height / 2 - self.cross_width / 2)
            self.horizontal_y = self.cross_y + (self.cross_height / 2 - self.cross_width / 2)
            self.horizontal_width = self.cross_height
            self.horizontal_height = self.cross_width
            self.CROSS_COLOR = "grey"

            # Flickering patches
            self.cross_flicker_distance = 360 # Original: 120
            self.flicker_width = 180
            self.flicker_height = 120
            self.flicker_left_x = self.horizontal_x - self.flicker_width - self.cross_flicker_distance
            self.flicker_right_x = self.horizontal_x + self.horizontal_width + self.cross_flicker_distance
            self.flicker_y = self.cross_y

            self.FLICKER_TEST_COLOR_BG = "white" # "ghostwhite"
            self.FLICKER_TEST_COLOR_PULSE = "black" # "floralwhite"

            # Instruction arrows
            self.arrow_body_width = 120
            self.arrow_body_height = 90
            self.arrow_tip_width = 30
            self.arrow_tip_height = 150
            self.ARROW_COLOR = "black"

            # Right-pointing arrow
            self.arrow_body_x_right = WIN_WIDTH / 2 - self.arrow_body_width / 2 - self.arrow_tip_width / 2
            self.arrow_body_y_right = WIN_HEIGHT / 2 - self.arrow_body_height / 2
            self.arrow_tip_x1_right = self.arrow_body_x_right + self.arrow_body_width
            self.arrow_tip_y1_right = WIN_HEIGHT / 2 - self.arrow_tip_height / 2
            self.arrow_tip_x2_right = self.arrow_body_x_right + self.arrow_body_width
            self.arrow_tip_y2_right = WIN_HEIGHT / 2 + self.arrow_tip_height / 2
            self.arrow_tip_x3_right = self.arrow_body_x_right + self.arrow_body_width + self.arrow_tip_width
            self.arrow_tip_y3_right = WIN_HEIGHT / 2

            # Left-pointing arrow
            self.arrow_body_x_left = WIN_WIDTH / 2 - self.arrow_body_width / 2 + self.arrow_tip_width / 2
            self.arrow_body_y_left = WIN_HEIGHT / 2 - self.arrow_body_height / 2
            self.arrow_tip_x1_left = self.arrow_body_x_left
            self.arrow_tip_y1_left = WIN_HEIGHT / 2 - self.arrow_tip_height / 2
            self.arrow_tip_x2_left = self.arrow_body_x_left
            self.arrow_tip_y2_left = WIN_HEIGHT / 2 + self.arrow_tip_height / 2
            self.arrow_tip_x3_left = self.arrow_body_x_left - self.arrow_tip_width
            self.arrow_tip_y3_left = WIN_HEIGHT / 2

            # General test timing
            self.test_timer_fps = 0

            # i. Basic test
            self.basic_countdown_secs = 5
            self.basic_countdown_fps = FPS * self.basic_countdown_secs
            self.basic_test_period_secs = 60 * 5 # 5 mins
            self.basic_test_period_fps = FPS * self.basic_test_period_secs

            self.basic_flicker = flicker_on
            
            self.central_flicker = False
            self.left_flicker = False
            self.right_flicker = False

            # if self.basic_flicker:
            if left_period or right_period:
                if left_period:
                    self.left_flicker = True
                    self.left_sided_flicker = True
                    self.left_period = left_period
                    self.left_flicker_on_period = left_flicker_on_period
                    self.left_flicker_count = 1

                if right_period:
                    self.right_flicker = True
                    self.right_sided_flicker = True
                    self.right_period = right_period
                    self.right_flicker_on_period = right_flicker_on_period
                    self.right_flicker_count = 1
            else:
                self.central_flicker = True
                self.period = period
                if flicker_on_period:
                    self.flicker_on_period = flicker_on_period
                else:
                    self.flicker_on_period = self.period // 2
                self.flicker_count = 1

            self.basic_patch_width = 120
            self.basic_patch_height = 120
            self.basic_patch_x = WIN_WIDTH / 2 - self.basic_patch_width / 2 - 1
            self.basic_patch_y = WIN_HEIGHT / 2 - self.basic_patch_height / 2 - 1

            # ii. Complete test
            self.countdown_secs = 1
            self.countdown_fps = FPS * self.countdown_secs
            self.single_test_period_secs = 2
            self.single_test_period_fps = FPS * self.single_test_period_secs

            # Instruction text fonts
            self.FLICKER_FONT = pygame.font.SysFont("calibri", 30)
            self.TEST_END_FONT = pygame.font.SysFont("calibri", 50)

            # Flicker activation and location
            self.is_flickering = flicker_on
            self.flicker_location = 0 # 0 - right, 1 - left
            self.is_testing = False

            # Statistics logging
            flicker_log_file_name = "../logs/flicker.log"
            os.makedirs(os.path.dirname(flicker_log_file_name), exist_ok=True)
            self.flicker_log_file = open(flicker_log_file_name, 'w')

            self.test_no = 0
            self.no_of_tests = 15
            self.has_signaled_flicker_start = False
            self.test_log = dict()
            for t in range(self.no_of_tests):
                self.test_log[t] = {'test_side': None, 'left': 0, 'right': 0, 'indet': 0}
            self.average_correct_activations_percentage = 0
            self.average_incorrect_activations_percentage = 0
            self.average_indet_activations_percentage = 0

            # Logging
            if self.central_flicker or not(self.basic_flicker):
                if self.central_flicker:
                    occ_1_plot_file_name = "../plots/occ_1_basic_flicker.txt"
                    occ_2_plot_file_name = "../plots/occ_2_basic_flicker.txt"
                    occ_1_data_file_name = "../plots/occ_1_basic_flicker_data.txt"
                    occ_2_data_file_name = "../plots/occ_2_basic_flicker_data.txt"
                    
                    tmp_1_plot_file_name = "../plots/tmp_1_basic_flicker.txt"
                    tmp_2_plot_file_name = "../plots/tmp_2_basic_flicker.txt"
                    tmp_1_data_file_name = "../plots/tmp_1_basic_flicker_data.txt"
                    tmp_2_data_file_name = "../plots/tmp_2_basic_flicker_data.txt"
                else:
                    occ_1_plot_file_name = "../plots/occ_1_basic_no_flicker.txt"
                    occ_2_plot_file_name = "../plots/occ_2_basic_no_flicker.txt"
                    occ_1_data_file_name = "../plots/occ_1_basic_no_flicker_data.txt"
                    occ_2_data_file_name = "../plots/occ_2_basic_no_flicker_data.txt"
                    
                    tmp_1_plot_file_name = "../plots/tmp_1_basic_no_flicker.txt"
                    tmp_2_plot_file_name = "../plots/tmp_2_basic_no_flicker.txt"
                    tmp_1_data_file_name = "../plots/tmp_1_basic_no_flicker_data.txt"
                    tmp_2_data_file_name = "../plots/tmp_2_basic_no_flicker_data.txt"

                os.makedirs(os.path.dirname(occ_1_plot_file_name), exist_ok=True)
                os.makedirs(os.path.dirname(occ_2_plot_file_name), exist_ok=True)
                os.makedirs(os.path.dirname(occ_1_data_file_name), exist_ok=True)
                os.makedirs(os.path.dirname(occ_2_data_file_name), exist_ok=True)

                os.makedirs(os.path.dirname(tmp_1_plot_file_name), exist_ok=True)
                os.makedirs(os.path.dirname(tmp_2_plot_file_name), exist_ok=True)
                os.makedirs(os.path.dirname(tmp_1_data_file_name), exist_ok=True)
                os.makedirs(os.path.dirname(tmp_2_data_file_name), exist_ok=True)

                self.occ_1_plot_file = open(occ_1_plot_file_name, 'w')
                self.occ_2_plot_file = open(occ_2_plot_file_name, 'w')
                self.occ_1_data_file = open(occ_1_data_file_name, 'w')
                self.occ_2_data_file = open(occ_2_data_file_name, 'w')
                
                self.tmp_1_plot_file = open(tmp_1_plot_file_name, 'w')
                self.tmp_2_plot_file = open(tmp_2_plot_file_name, 'w')
                self.tmp_1_data_file = open(tmp_1_data_file_name, 'w')
                self.tmp_2_data_file = open(tmp_2_data_file_name, 'w')

                self.occ_1_plot_time_point = 0
                self.occ_2_plot_time_point = 0
                self.tmp_1_plot_time_point = 0
                self.tmp_2_plot_time_point = 0
            
            else:
                if self.right_flicker:
                    occ_1_left_plot_file_name = "../plots/occ_1_left_basic_flicker.txt"
                    occ_2_left_plot_file_name = "../plots/occ_2_left_basic_flicker.txt"
                    occ_1_left_data_file_name = "../plots/occ_1_left_basic_flicker_data.txt"
                    occ_2_left_data_file_name = "../plots/occ_2_left_basic_flicker_data.txt"
                    
                    tmp_1_left_plot_file_name = "../plots/tmp_1_left_basic_flicker.txt"
                    tmp_2_left_plot_file_name = "../plots/tmp_2_left_basic_flicker.txt"
                    tmp_1_left_data_file_name = "../plots/tmp_1_left_basic_flicker_data.txt"
                    tmp_2_left_data_file_name = "../plots/tmp_2_left_basic_flicker_data.txt"

                    os.makedirs(os.path.dirname(occ_1_left_plot_file_name), exist_ok=True)
                    os.makedirs(os.path.dirname(occ_2_left_plot_file_name), exist_ok=True)
                    os.makedirs(os.path.dirname(occ_1_left_data_file_name), exist_ok=True)
                    os.makedirs(os.path.dirname(occ_2_left_data_file_name), exist_ok=True)
                    
                    os.makedirs(os.path.dirname(tmp_1_left_plot_file_name), exist_ok=True)
                    os.makedirs(os.path.dirname(tmp_2_left_plot_file_name), exist_ok=True)
                    os.makedirs(os.path.dirname(tmp_1_left_data_file_name), exist_ok=True)
                    os.makedirs(os.path.dirname(tmp_2_left_data_file_name), exist_ok=True)

                    self.occ_1_left_plot_file = open(occ_1_left_plot_file_name, 'w')
                    self.occ_2_left_plot_file = open(occ_2_left_plot_file_name, 'w')
                    self.occ_1_left_data_file = open(occ_1_left_data_file_name, 'w')
                    self.occ_2_left_data_file = open(occ_2_left_data_file_name, 'w')
                    
                    self.tmp_1_left_plot_file = open(tmp_1_left_plot_file_name, 'w')
                    self.tmp_2_left_plot_file = open(tmp_2_left_plot_file_name, 'w')
                    self.tmp_1_left_data_file = open(tmp_1_left_data_file_name, 'w')
                    self.tmp_2_left_data_file = open(tmp_2_left_data_file_name, 'w')

                    self.occ_1_left_plot_time_point = 0
                    self.occ_2_left_plot_time_point = 0
                    
                    self.tmp_1_left_plot_time_point = 0
                    self.tmp_2_left_plot_time_point = 0
                
                if self.left_flicker:
                    occ_2_right_plot_file_name = "../plots/occ_2_right_basic_flicker.txt"
                    occ_1_right_plot_file_name = "../plots/occ_1_right_basic_flicker.txt"
                    occ_2_right_data_file_name = "../plots/occ_2_right_basic_flicker_data.txt"
                    occ_1_right_data_file_name = "../plots/occ_1_right_basic_flicker_data.txt"

                    tmp_2_right_plot_file_name = "../plots/tmp_2_right_basic_flicker.txt"
                    tmp_1_right_plot_file_name = "../plots/tmp_1_right_basic_flicker.txt"
                    tmp_2_right_data_file_name = "../plots/tmp_2_right_basic_flicker_data.txt"
                    tmp_1_right_data_file_name = "../plots/tmp_1_right_basic_flicker_data.txt"

                    os.makedirs(os.path.dirname(occ_2_right_plot_file_name), exist_ok=True)
                    os.makedirs(os.path.dirname(occ_1_right_plot_file_name), exist_ok=True)
                    os.makedirs(os.path.dirname(occ_2_right_data_file_name), exist_ok=True)
                    os.makedirs(os.path.dirname(occ_1_right_data_file_name), exist_ok=True)

                    os.makedirs(os.path.dirname(tmp_2_right_plot_file_name), exist_ok=True)
                    os.makedirs(os.path.dirname(tmp_1_right_plot_file_name), exist_ok=True)
                    os.makedirs(os.path.dirname(tmp_2_right_data_file_name), exist_ok=True)
                    os.makedirs(os.path.dirname(tmp_1_right_data_file_name), exist_ok=True)

                    self.occ_2_right_plot_file = open(occ_2_right_plot_file_name, 'w')
                    self.occ_1_right_plot_file = open(occ_1_right_plot_file_name, 'w')
                    self.occ_2_right_data_file = open(occ_2_right_data_file_name, 'w')
                    self.occ_1_right_data_file = open(occ_1_right_data_file_name, 'w')

                    self.tmp_2_right_plot_file = open(tmp_2_right_plot_file_name, 'w')
                    self.tmp_1_right_plot_file = open(tmp_1_right_plot_file_name, 'w')
                    self.tmp_2_right_data_file = open(tmp_2_right_data_file_name, 'w')
                    self.tmp_1_right_data_file = open(tmp_1_right_data_file_name, 'w')

                    self.occ_2_right_plot_time_point = 0
                    self.occ_1_right_plot_time_point = 0
                    
                    self.tmp_2_right_plot_time_point = 0
                    self.tmp_1_right_plot_time_point = 0

            self.write_plot_data_at_end = False
            if self.write_plot_data_at_end:
                # TODO: Temporal lobes
                self.occ_1_plot_points = []
                self.occ_2_plot_points = []
                
                self.occ_1_left_plot_points = []
                self.occ_2_left_plot_points = []
                self.occ_2_right_plot_points = []
                self.occ_1_right_plot_points = []

            # File cleanup
            atexit.register(self.cleanup)

        def cleanup(self):
            if self.write_plot_data_at_end:
                self.empty_logging_queue()
            self.flicker_log_file.close()

            if self.central_flicker or not(self.basic_flicker):
                self.occ_1_plot_file.close()
                self.occ_2_plot_file.close()
                self.tmp_1_plot_file.close()
                self.tmp_2_plot_file.close()
            else:
                if self.right_flicker:
                    self.occ_1_left_plot_file.close()
                    self.occ_2_left_plot_file.close()
                    self.tmp_1_left_plot_file.close()
                    self.tmp_2_left_plot_file.close()
                
                if self.left_flicker:
                    self.occ_1_right_plot_file.close()
                    self.occ_2_right_plot_file.close()
                    self.tmp_1_right_plot_file.close()
                    self.tmp_2_right_plot_file.close()

            # Logging the sample rate for the FFT in the data log file
            if self.central_flicker or not(self.basic_flicker):
                self.occ_1_data_file.write(f"{SAMPLE_FREQ}")
                self.occ_2_data_file.write(f"{SAMPLE_FREQ}")
                self.occ_1_data_file.close()
                self.occ_2_data_file.close()

                self.tmp_1_data_file.write(f"{SAMPLE_FREQ}")
                self.tmp_2_data_file.write(f"{SAMPLE_FREQ}")
                self.tmp_1_data_file.close()
                self.tmp_2_data_file.close()
            else:
                if self.right_flicker:
                    self.occ_1_left_data_file.write(f"{SAMPLE_FREQ}")
                    self.occ_2_left_data_file.write(f"{SAMPLE_FREQ}")
                    self.occ_1_left_data_file.close()
                    self.occ_2_left_data_file.close()

                    self.tmp_1_left_data_file.write(f"{SAMPLE_FREQ}")
                    self.tmp_2_left_data_file.write(f"{SAMPLE_FREQ}")
                    self.tmp_1_left_data_file.close()
                    self.tmp_2_left_data_file.close()

                if self.left_flicker:
                    self.occ_1_right_data_file.write(f"{SAMPLE_FREQ}")
                    self.occ_2_right_data_file.write(f"{SAMPLE_FREQ}")
                    self.occ_1_right_data_file.close()
                    self.occ_2_right_data_file.close()
                    
                    self.tmp_1_right_data_file.write(f"{SAMPLE_FREQ}")
                    self.tmp_2_right_data_file.write(f"{SAMPLE_FREQ}")
                    self.tmp_1_right_data_file.close()
                    self.tmp_2_right_data_file.close()

        def get_testing_state(self):
            return self.is_testing
        
        def get_flickering_state(self):
            return self.is_flickering

        def get_flicker_location_string(self):
            if self.flicker_location == 0:
                return "right"
            else:
                return "left"
        
        def is_flicker_central(self):
            return self.central_flicker
        
        def get_flicker_location(self):
            if self.central_flicker:
                return 'center'
            if self.left_flicker and self.right_flicker:
                return 'left-right'
            if self.left_flicker:
                return 'left'
            if self.right_flicker:
                return 'right'

        def get_flicker_frequency(self):
            # It's whole division in order to keep the reduction factor divisible by new_window_size, a.k.a. 4
            return FPS // self.period
        
        def get_left_flicker_frequency(self):
            # It's whole division in order to keep the reduction factor divisible by new_window_size, a.k.a. 4
            if self.left_flicker:
                return FPS // self.left_period
            elif self.right_flicker:
                return FPS // self.right_period
            else:
                return FPS // self.period
        
        def get_right_flicker_frequency(self):
            # It's whole division in order to keep the reduction factor divisible by new_window_size, a.k.a. 4
            if self.right_flicker:
                return FPS // self.right_period
            elif self.left_flicker:
                return FPS // self.left_period
            else:
                return FPS // self.period

        def log_threshold_crossing(self, delta, side):
            # self.flicker_log_file.write(f"{self.test_no}: Threshold crossed by {delta} in the {side} hs\n")
            self.test_log[self.test_no][side] += 1 # just counting the # of activations per side for now
        
        def log_indeterminate_diff(self):
            self.test_log[self.test_no]['indet'] += 1

        # def log_error(self):
        #     self.flicker_log_file.write(f"Avg difference was >= in the wrong hs\n")
            
        def log_plot_data(self, amp, file):
            if self.write_plot_data_at_end:
                if file == 'occ_1':
                    self.occ_1_plot_points.append((amp, self.occ_1_plot_time_point))
                    self.occ_1_plot_time_point += 1
                elif file == 'occ_2':
                    self.occ_2_plot_points.append((amp, self.occ_2_plot_time_point))
                    self.occ_2_plot_time_point += 1

                elif file == 'occ_1_left':
                    self.occ_1_left_plot_points.append((amp, self.occ_1_left_plot_time_point))
                    self.occ_1_left_plot_time_point += 1
                elif file == 'occ_2_left':
                    self.occ_2_left_plot_points.append((amp, self.occ_2_left_plot_time_point))
                    self.occ_2_left_plot_time_point += 1
                    
                elif file == 'occ_1_right':
                    self.occ_1_right_plot_points.append((amp, self.occ_1_right_plot_time_point))
                    self.occ_1_right_plot_time_point += 1
                elif file == 'occ_2_right':
                    self.occ_2_right_plot_points.append((amp, self.occ_2_right_plot_time_point))
                    self.occ_2_right_plot_time_point += 1
                
                # TODO: Temporals

            else:
                if file == 'occ_1':
                    self.occ_1_plot_file.write(f"{amp} {self.occ_1_plot_time_point}\n")
                    self.occ_1_plot_time_point += 1
                elif file == 'occ_2':
                    self.occ_2_plot_file.write(f"{amp} {self.occ_2_plot_time_point}\n")
                    self.occ_2_plot_time_point += 1
                
                elif file == 'occ_1_left':
                    self.occ_1_left_plot_file.write(f"{amp} {self.occ_1_left_plot_time_point}\n")
                    self.occ_1_left_plot_time_point += 1
                elif file == 'occ_2_left':
                    self.occ_2_left_plot_file.write(f"{amp} {self.occ_2_left_plot_time_point}\n")
                    self.occ_2_left_plot_time_point += 1

                elif file == 'occ_1_right':
                    self.occ_1_right_plot_file.write(f"{amp} {self.occ_1_right_plot_time_point}\n")
                    self.occ_1_right_plot_time_point += 1
                elif file == 'occ_2_right':
                    self.occ_2_right_plot_file.write(f"{amp} {self.occ_2_right_plot_time_point}\n")
                    self.occ_2_right_plot_time_point += 1

                # Temporals
                elif file == 'tmp_1':
                    self.tmp_1_plot_file.write(f"{amp} {self.tmp_1_plot_time_point}\n")
                    self.tmp_1_plot_time_point += 1
                elif file == 'tmp_2':
                    self.tmp_2_plot_file.write(f"{amp} {self.tmp_2_plot_time_point}\n")
                    self.tmp_2_plot_time_point += 1

                elif file == 'tmp_1_right':
                    self.tmp_1_right_plot_file.write(f"{amp} {self.tmp_1_right_plot_time_point}\n")
                    self.tmp_1_right_plot_time_point += 1
                elif file == 'tmp_2_right':
                    self.tmp_2_right_plot_file.write(f"{amp} {self.tmp_2_right_plot_time_point}\n")
                    self.tmp_2_right_plot_time_point += 1
                    
                elif file == 'tmp_1_left':
                    self.tmp_1_left_plot_file.write(f"{amp} {self.tmp_1_left_plot_time_point}\n")
                    self.tmp_1_left_plot_time_point += 1
                elif file == 'tmp_2_left':
                    self.tmp_2_left_plot_file.write(f"{amp} {self.tmp_2_left_plot_time_point}\n")
                    self.tmp_2_left_plot_time_point += 1

        def log_data(self, buff, hs):
            if hs == 'occ_1':
                for dp in buff:
                    self.occ_1_data_file.write(str(dp) + " ")
                self.occ_1_data_file.write("\n")
            elif hs == 'occ_2':
                for dp in buff:
                    self.occ_2_data_file.write(str(dp) + " ")
                self.occ_2_data_file.write("\n")
            
            elif hs == 'occ_1_left':
                for dp in buff:
                    self.occ_1_left_data_file.write(str(dp) + " ")
                self.occ_1_left_data_file.write("\n")
            elif hs == 'occ_2_left':
                for dp in buff:
                    self.occ_2_left_data_file.write(str(dp) + " ")
                self.occ_2_left_data_file.write("\n")
                
            elif hs == 'occ_1_right':
                for dp in buff:
                    self.occ_1_right_data_file.write(str(dp) + " ")
                self.occ_1_right_data_file.write("\n")
            elif hs == 'occ_2_right':
                for dp in buff:
                    self.occ_2_right_data_file.write(str(dp) + " ")
                self.occ_2_right_data_file.write("\n")
                
            # Temporals
            elif hs == 'tmp_1_right':
                for dp in buff:
                    self.tmp_1_right_data_file.write(str(dp) + " ")
                self.tmp_1_right_data_file.write("\n")
            elif hs == 'tmp_2_right':
                for dp in buff:
                    self.tmp_2_right_data_file.write(str(dp) + " ")
                self.tmp_2_right_data_file.write("\n")
            
            elif hs == 'tmp_1_left':
                for dp in buff:
                    self.tmp_1_left_data_file.write(str(dp) + " ")
                self.tmp_1_left_data_file.write("\n")
            elif hs == 'tmp_2_left':
                for dp in buff:
                    self.tmp_2_left_data_file.write(str(dp) + " ")
                self.tmp_2_left_data_file.write("\n")
                
            elif hs == 'tmp_1':
                for dp in buff:
                    self.tmp_1_data_file.write(str(dp) + " ")
                self.tmp_1_data_file.write("\n")
            elif hs == 'tmp_2':
                for dp in buff:
                    self.tmp_2_data_file.write(str(dp) + " ")
                self.tmp_2_data_file.write("\n")

        def empty_logging_queue(self):
            # self.linearly_regress_plot_data()
            for (amp,t) in self.occ_1_plot_points:
                self.occ_1_plot_file.write(f"{amp} {t}\n")
            for (amp,t) in self.occ_2_plot_points:
                self.occ_2_plot_file.write(f"{amp} {t}\n")

        def linearly_regress_plot_data(self):
            ys_occ_1_amps = list(map(lambda t: t[0], self.occ_1_plot_points))
            ys_occ_2_amps = list(map(lambda t: t[0], self.occ_2_plot_points))
            no_of_data_points = len(ys_occ_1_amps)
            xs_flicker = np.arange(0, no_of_data_points).tolist()
            
            reg1_flicker = LinearRegression(fit_intercept=True)
            reg2_flicker = LinearRegression(fit_intercept=True)
            reg1_flicker.fit(np.array(xs_flicker).reshape(-1,1), np.array(ys_occ_1_amps))
            reg2_flicker.fit(np.array(xs_flicker).reshape(-1,1), np.array(ys_occ_2_amps))

            for t in range(no_of_data_points):
                ys_occ_1_amps_curr_drift = reg1_flicker.coef_[0] * (no_of_data_points - t)
                ys_occ_2_amps_curr_drift = reg2_flicker.coef_[0] * (no_of_data_points - t)

                ys_occ_1_amps[t] = ys_occ_1_amps[t] + ys_occ_1_amps_curr_drift - reg1_flicker.intercept_
                ys_occ_2_amps[t] = ys_occ_2_amps[t] + ys_occ_2_amps_curr_drift - reg2_flicker.intercept_
                
            self.occ_1_plot_points = list(zip(ys_occ_1_amps, xs_flicker))
            self.occ_2_plot_points = list(zip(ys_occ_2_amps, xs_flicker))

        def analyze_test_run(self, test):
            side = self.test_log[test]['side']
            left_activations = self.test_log[test]['left']
            right_activations = self.test_log[test]['right']
            indet_activations = self.test_log[test]['indet']

            total_activations = left_activations + right_activations + indet_activations
            left_activations_percentage = left_activations / total_activations
            right_activations_percentage = right_activations / total_activations
            indet_activations_percentage = indet_activations / total_activations
            
            if side == 'left':
                correct_activations_percentage = left_activations_percentage
                incorrect_activations_percentage = right_activations_percentage
            else:
                correct_activations_percentage = right_activations_percentage
                incorrect_activations_percentage = left_activations_percentage
            
            self.average_correct_activations_percentage += correct_activations_percentage
            self.average_incorrect_activations_percentage += incorrect_activations_percentage
            self.average_indet_activations_percentage += indet_activations_percentage

            self.flicker_log_file.write(f"Test #{test+1} ({side}):\n"
                                        + f"\tcorrect: {correct_activations_percentage:.2f}\n"
                                        + f"\tincorrect: {incorrect_activations_percentage:.2f}\n"
                                        + f"\tindeterminate: {indet_activations_percentage:.2f}\n"
                                        + f"\ttotal # of activations: {total_activations:.2f}\n\n")

        def analyze_tests(self):
            """
            TODO:
            - record data per test
                - % of correct lobe activation - done
                - % of incorrect lobe activation - done
                - % of indeterminate lobe activation - done
                - average difference between the two activation amplitudes
            - aggregate all test data
                - average % of correct lobe activation - done
                - highest % of correct lobe activation (+ associated test)
                - average % of incorrect lobe activation - done
                - highest % of incorrect lobe activation (+ associated test)
                - average % of indeterminate lobe activation - done
                - overall average difference between the two activation amplitudes
                - greatest difference (+ associated test)
            """
            for test in range(self.no_of_tests):
                self.analyze_test_run(test)
            
            self.average_correct_activations_percentage = self.average_correct_activations_percentage / self.no_of_tests
            self.average_incorrect_activations_percentage = self.average_incorrect_activations_percentage / self.no_of_tests
            self.average_indet_activations_percentage = self.average_indet_activations_percentage / self.no_of_tests
            self.flicker_log_file.write(f"Total % of correct hs activation: {self.average_correct_activations_percentage:.2f}\n"
                                        + f"Total % of incorrect hs activation: {self.average_incorrect_activations_percentage:.2f}\n"
                                        + f"Total % of indeterminate activation: {self.average_indet_activations_percentage:.2f}\n")

        def draw_arrow(self, win):
            if self.flicker_location == 0:
                pygame.draw.rect(win, self.ARROW_COLOR,
                                (self.arrow_body_x_right, self.arrow_body_y_right,
                                self.arrow_body_width, self.arrow_body_height))
                pygame.draw.polygon(win, self.ARROW_COLOR,
                                    [(self.arrow_tip_x1_right, self.arrow_tip_y1_right),
                                    (self.arrow_tip_x2_right, self.arrow_tip_y2_right),
                                    (self.arrow_tip_x3_right, self.arrow_tip_y3_right)])
            else:
                pygame.draw.rect(win, self.ARROW_COLOR,
                                (self.arrow_body_x_left, self.arrow_body_y_left,
                                self.arrow_body_width, self.arrow_body_height))
                pygame.draw.polygon(win, self.ARROW_COLOR,
                                    [(self.arrow_tip_x1_left, self.arrow_tip_y1_left),
                                    (self.arrow_tip_x2_left, self.arrow_tip_y2_left),
                                    (self.arrow_tip_x3_left, self.arrow_tip_y3_left)])

        def draw_basic_test_window(self, win):
            """
            Draw method for the basic (central flicker/blank screen) 60 second test
            """
            win.fill(self.FLICKER_TEST_COLOR_BG)

            # Stage 1: Countdown and instruction
            if self.test_timer_fps < self.basic_countdown_fps:
                if self.central_flicker:
                    instr_text = self.FLICKER_FONT.render("Focus on the center of the screen for " +
                                                        f"{self.basic_test_period_secs} seconds",
                                                        1, "black")
                elif self.left_flicker and self.right_flicker:
                    instr_text = self.FLICKER_FONT.render("Focus covertly on the left and right sides of the screen for " +
                                                        f"{self.basic_test_period_secs} seconds",
                                                        1, "black")
                elif self.left_flicker:
                    instr_text = self.FLICKER_FONT.render("Focus covertly on the left side of the screen for " +
                                                        f"{self.basic_test_period_secs} seconds",
                                                        1, "black")
                elif self.right_flicker:
                    instr_text = self.FLICKER_FONT.render("Focus covertly on the right side of the screen for " +
                                                        f"{self.basic_test_period_secs} seconds",
                                                        1, "black")

                # TODO: make the text always fit the window                    

                win.blit(instr_text, (WIN_WIDTH / 2 - instr_text.get_width() / 2,
                                      WIN_HEIGHT / 4))
                self.test_timer_fps += 1

            # Stage 2: Actual test
            elif self.test_timer_fps < self.basic_test_period_fps:
                if not(self.has_signaled_flicker_start):
                    self.has_signaled_flicker_start = True
                
                if self.basic_flicker == True:
                    if self.left_flicker or self.right_flicker:
                        if self.left_flicker:
                            if self.left_flicker_count <= self.left_flicker_on_period:
                                left_color = self.FLICKER_TEST_COLOR_BG
                            else:
                                left_color = self.FLICKER_TEST_COLOR_PULSE
                                
                            if self.left_flicker_count == self.left_period:
                                self.left_flicker_count = 1
                            else:
                                self.left_flicker_count += 1
                        
                        if self.right_flicker:
                            if self.right_flicker_count <= self.right_flicker_on_period:
                                right_color = self.FLICKER_TEST_COLOR_BG
                            else:
                                right_color = self.FLICKER_TEST_COLOR_PULSE
                                
                            if self.right_flicker_count == self.right_period:
                                self.right_flicker_count = 1
                            else:
                                self.right_flicker_count += 1
                    else:
                        if self.flicker_count <= self.flicker_on_period:
                            color = self.FLICKER_TEST_COLOR_BG
                        else:
                            color = self.FLICKER_TEST_COLOR_PULSE

                        if self.flicker_count == self.period:
                            self.flicker_count = 1
                        else:
                            self.flicker_count += 1

                    self.is_flickering = True
                    self.is_testing = True
                else:
                    color = self.CROSS_COLOR
                    self.is_testing = True
                
                if self.left_flicker or self.right_flicker:
                    # Focus cross is added when the flicker is to the side
                    pygame.draw.rect(win, self.CROSS_COLOR,
                                    (self.cross_x, self.cross_y,
                                    self.cross_width, self.cross_height))
                    pygame.draw.rect(win, self.CROSS_COLOR,
                                    (self.horizontal_x, self.horizontal_y,
                                    self.horizontal_width, self.horizontal_height))
                    
                    if self.left_flicker:
                        pygame.draw.rect(win, left_color,
                                        (self.flicker_left_x, self.flicker_y,
                                        self.flicker_width, self.flicker_height))
                        
                    if self.right_flicker:
                        pygame.draw.rect(win, right_color,
                                        (self.flicker_right_x, self.flicker_y,
                                        self.flicker_width, self.flicker_height))                                        
                else:
                    pygame.draw.rect(win, color,
                                    (self.basic_patch_x, self.basic_patch_y,
                                    self.basic_patch_width, self.basic_patch_height))
                self.test_timer_fps += 1
            
            # Stage 3: Test end
            else:
                instr_text = self.FLICKER_FONT.render(f"The test is over", 1, "black")
                win.blit(instr_text, (WIN_WIDTH / 2 - instr_text.get_width() / 2,
                                      WIN_HEIGHT / 4))
                self.is_flickering = False
                self.is_testing = False

                pygame.display.update()
                return 0
            
            pygame.display.update()
            return 1

        def draw_flicker_test_window(self, win):
            """
            Draw method for the flicker test
            """
            win.fill(self.FLICKER_TEST_COLOR_BG)

            if self.test_no == self.no_of_tests:
                end_text = self.FLICKER_FONT.render("Testing has ended", 1, "black")
                win.blit(end_text, (WIN_WIDTH / 2 - end_text.get_width() / 2,
                                    WIN_HEIGHT / 4))
                pygame.display.update()
                return 0

            # Stage 1: Instructing the user in-between tests
            if self.test_timer_fps < self.countdown_secs:
                if self.flicker_location == 0:
                    dir_text = "right"
                    self.test_log[self.test_no]['side'] = "right"
                else:
                    dir_text = "left"
                    self.test_log[self.test_no]['side'] = "left"

                instr_text = self.FLICKER_FONT.render(f"({self.test_no + 1})" +
                                                      f"Shift your focus to the {dir_text} side of the screen", 1, "black")
                win.blit(instr_text, (WIN_WIDTH / 2 - instr_text.get_width() / 2,
                                      WIN_HEIGHT / 4))
                self.draw_arrow(win)
                self.test_timer_fps += 1

            # Stage 2: The flicker test itself
            elif self.test_timer_fps < self.countdown_secs + self.single_test_period_fps:
                # Logging
                if not(self.has_signaled_flicker_start):
                    # self.flicker_log_file.write(f"{self.test_no}: Flickering begins ({self.get_flicker_location_string()})\n")
                    self.has_signaled_flicker_start = True

                # Focus cross
                pygame.draw.rect(win, self.CROSS_COLOR,
                                (self.cross_x, self.cross_y,
                                 self.cross_width, self.cross_height))
                pygame.draw.rect(win, self.CROSS_COLOR,
                                (self.horizontal_x, self.horizontal_y,
                                 self.horizontal_width, self.horizontal_height))

                # Flickering patches
                if self.flicker_count <= self.flicker_on_period:
                    color = self.FLICKER_TEST_COLOR_BG
                else:
                    color = self.FLICKER_TEST_COLOR_PULSE

                if self.flicker_count == self.period:
                    self.flicker_count = 1
                else:
                    self.flicker_count += 1

                pygame.draw.rect(win, color,
                                (self.flicker_left_x, self.flicker_y,
                                self.flicker_width, self.flicker_height))
                pygame.draw.rect(win, color,
                                (self.flicker_right_x, self.flicker_y,
                                self.flicker_width, self.flicker_height))
                
                self.is_flickering = True
                self.is_testing = True
                self.test_timer_fps += 1

            # Setting back the timer
            else:
                # Logging
                # self.flicker_log_file.write(f"{self.test_no}: Flickering stops ({self.get_flicker_location_string()})\n\n")
                self.has_signaled_flicker_start = False

                self.is_flickering = False
                self.is_testing = False
                self.test_timer_fps = 0
                self.flicker_location = 1 - self.flicker_location
                self.test_no += 1
                if self.test_no == self.no_of_tests:
                    self.analyze_tests()
                    # self.flicker_log_file.write("Testing done")

            pygame.display.update()
            return 1

        # def draw_flicker_test_window_complete(self, win):
        #     """
        #     Draw method for the complete flicker test
        #     """
        #     win.fill(self.FLICKER_TEST_COLOR_BG)

        #     # There are 4 stages to the test

        #     # Stage 1: Instructing the user
        #     if self.test_timer_fps < self.instruction_period_fps:
        #         if self.direction == 0:
        #             dir_text = "right"
        #         else:
        #             dir_text = "left"
        #         instr_text = self.FLICKER_FONT.render(f"Next up, focus on the {dir_text} side of the screen", 1, "black")
        #         win.blit(instr_text, (WIN_WIDTH / 2 - instr_text.get_width() / 2,
        #                               WIN_HEIGHT / 4))
        #         self.draw_arrow(win)
        #         self.test_timer_fps += 1

        #     # Stage 2: Countdown to the flicker test
        #     elif self.test_timer_fps < self.instruction_period_fps + self.countdown_secs:
        #         if self.test_timer_fps < self.instruction_period_fps + self.countdown_secs / 3:
        #             countdown_text = self.FLICKER_FONT.render("3", 1, "black")
        #         elif self.test_timer_fps < self.instruction_period_fps + self.countdown_secs * 2 / 3:
        #             countdown_text = self.FLICKER_FONT.render("2", 1, "black")
        #         else:
        #             countdown_text = self.FLICKER_FONT.render("1", 1, "black")

        #         win.blit(countdown_text, (WIN_WIDTH / 2 - countdown_text.get_width() / 2,
        #                                   WIN_HEIGHT / 4))
        #         self.draw_arrow(win)
        #         self.test_timer_fps += 1

        #     # Stage 3: The flicker test itself
        #     elif self.test_timer_fps < self.instruction_period_fps +    \
        #                                self.countdown_secs +              \
        #                                self.single_test_period_fps:
        #         # Focus cross
        #         pygame.draw.rect(win, self.CROSS_COLOR,
        #                         (self.cross_x, self.cross_y,
        #                          self.cross_width, self.cross_height))
        #         pygame.draw.rect(win, self.CROSS_COLOR,
        #                         (self.horizontal_x, self.horizontal_y,
        #                          self.horizontal_width, self.horizontal_height))

        #         # Flickering patches
        #         if self.flicker_count <= self.period // 2:
        #             color = self.FLICKER_TEST_COLOR_BG
        #         else:
        #             color = self.FLICKER_TEST_COLOR_PULSE
        #         if self.flicker_count == self.period:
        #             self.flicker_count = 0
        #         else:
        #             self.flicker_count += 1

        #         pygame.draw.rect(win, color,
        #                         (self.flicker_left_x, self.flicker_y,
        #                          self.flicker_width, self.flicker_height))
        #         pygame.draw.rect(win, color,
        #                         (self.flicker_right_x, self.flicker_y,
        #                          self.flicker_width, self.flicker_height))
                
        #         self.test_timer_fps += 1
            
        #     # Stage 4: Post-test cooldown period
        #     else:
        #         if self.test_timer_fps == self.instruction_period_fps +    \
        #                                   self.countdown_secs +              \
        #                                   self.single_test_period_fps +    \
        #                                   self.test_halt_fps:
        #             self.test_timer_fps = 0
        #             self.direction = 1 - self.direction
        #         else:
        #             test_end_text = self.TEST_END_FONT.render(f"Test end", 1, "black")
        #             win.blit(test_end_text, (WIN_WIDTH / 2 - test_end_text.get_width() / 2,
        #                                      WIN_HEIGHT / 4))
        #             self.test_timer_fps += 1

        #     pygame.display.update()


    def draw(self, win, paddle, ball, bricks, gauge, lives, direction, period):
        """
        Draw method for game object. Calls the draw method of all objects defined inside the game object.
        """
        win.fill(self.GAME_COLOR_BG)
        lives_text = self.LIVES_FONT.render(f"Lives: {lives}", 1, "black")
        win.blit(lives_text, (gauge.get_x() - lives_text.get_width() / 2,
                              gauge.get_y() + 40))
        #    Originally:     (x: 10, y: self.HEIGHT - lives_text.get_height() - 20)    #

        # Focus cross
        pygame.draw.rect(win, self.cross_color,
                            (self.cross_x, self.cross_y,
                             self.cross_width, self.cross_height))
            
        horizontal_x = self.cross_x - (self.cross_height / 2 - self.cross_width / 2)
        horizontal_y = self.cross_y + (self.cross_height / 2 - self.cross_width / 2)
        horizontal_width = self.cross_height
        horizontal_height = self.cross_width

        pygame.draw.rect(win, self.cross_color,
                            (horizontal_x, horizontal_y,
                             horizontal_width, horizontal_height))

        # Flickering areas
        flicker_width = 200
        flicker_height = 120
        cross_x_mid = self.cross_x + self.cross_width / 2
        cross_flicker_distance = 50

        if self.flicker_count == 1:
            self.flicker_count = 0
            color = self.GAME_COLOR_BG
        else:
            self.flicker_count = 1
            color = self.GAME_COLOR_FLICKER
            
        horizontal_x = self.cross_x - (self.cross_height / 2 - self.cross_width / 2)
        horizontal_y = self.cross_y + (self.cross_height / 2 - self.cross_width / 2)
        horizontal_width = self.cross_height
        horizontal_height = self.cross_width

        if direction != 0:
            if direction == 1:
                flicker_x = cross_x_mid + horizontal_width / 2 + cross_flicker_distance
            else: # direction == -1
                flicker_x = cross_x_mid - horizontal_width / 2 - cross_flicker_distance - flicker_width

            # flicker_left_x = cross_x_mid - horizontal_width / 2 - cross_flicker_distance - flicker_width
            # flicker_right_x = cross_x_mid + horizontal_width / 2 + cross_flicker_distance
            flicker_y = paddle.get_y() - flicker_height - 30

            # print(f"Left x diff: {cross_x_mid - (flicker_left_x + flicker_width)}\n")
            # print(f"Right x diff: {flicker_right_x - cross_x_mid}\n\n")

            pygame.draw.rect(win, color,
                             (flicker_x, flicker_y,
                              flicker_width, flicker_height))

        paddle.draw(win)
        ball.draw(win)
        gauge.draw(win)

        for brick in bricks:
            brick.draw(win)

        pygame.display.update()


    # Ball Wall Collision:
    def ball_collision(self, ball, paddle):
        """
        Collision method for ball versus walls. Allows the ball to be appropriately reflected upon hitting a wall.
        """
        if ball.x - self.BALL_RADIUS <= 0 or ball.x + self.BALL_RADIUS >= self.WIDTH:
            ball.set_vel(ball.x_vel * -1, ball.y_vel)
        if ball.y + self.BALL_RADIUS >= self.HEIGHT:
            ball.set_vel(4, ball.y_vel * -1)
        if ball.y - self.BALL_RADIUS <= 0:
            ball.predict()
            paddle.x_initial = paddle.x
            self.calculate_steps_needed(paddle, ball.x_predict)
            ball.set_vel(ball.x_vel, ball.y_vel * -1)

    # Ball Paddle Collision:
    def ball_paddle_collision(self, ball, paddle, gauge):
        """
        Collision method for ball versus paddle. Allows the ball to be appropriately reflected upon hitting the paddle. 
        This reflection considers the angle of collision as well as the collision location on the paddle.
        """
        if not (ball.x <= paddle.x + paddle.width and ball.x >= paddle.x):
            return
        if not (ball.y + ball.radius >= paddle.y):
            return
        
        if self.steps_needed != 0:
            gauge.change_percentage(int(np.floor(100 - 100 * (np.abs(self.steps_needed - self.steps_taken) / self.steps_needed))))
        else:
            gauge.change_percentage(100)
        self.steps_taken = 0
        self.steps_needed = 0

        paddle_center = paddle.x + paddle.width/2
        distance_to_center = ball.x - paddle_center

        percent_width = distance_to_center / paddle.width
        angle = percent_width * 90
        angle_radians = math.radians(angle)

        x_vel = math.sin(angle_radians) * BALL_INITIAL_VEL
        y_vel = math.cos(angle_radians) * BALL_INITIAL_VEL * -1
        
        y_vel = -ball.y_vel
        ball.set_vel(x_vel, y_vel)

        # For the case when we program the paddle to move directly to its final spot
        paddle.reset_move_flag()

    def generate_bricks(self, rows, cols):
        """
        Method for generating all necessary bricks.
        """
        gap = 2
        brick_width = self.WIDTH // cols - gap
        brick_height = 20

        bricks = []
        for row in range(rows): # Currently rows = 1 within the later calls
            for col in range(cols):
                brick = self.Brick(col * brick_width + gap * col, row * brick_height +
                            gap * row, brick_width, brick_height, 2, [(0, 0, 255), (255, 0, 0)]) # [(0, 255, 255), (255, 0, 0)])
                bricks.append(brick)

        return bricks

    def fix_ball_conditions(self, ball):
            """
            Fixes the ball conditions. If the ball surpasses the walls, fixes its position.
            """
            # Fix if stuck to right or left wall:
            if ball.x <= BALL_RADIUS:
                ball.x = BALL_RADIUS
            elif ball.x >= WIN_WIDTH - BALL_RADIUS:
                ball.x = WIN_WIDTH - BALL_RADIUS
            # Fix if stuck to upper wall:
            if ball.y <= BALL_RADIUS:
                ball.y = BALL_RADIUS


    def main(self):
        clock = pygame.time.Clock()
        mySensor = self.EEGSensor
        
        paddle_x = self.WIDTH / 2 - self.PADDLE_WIDTH / 2
        paddle_y = self.HEIGHT - self.PADDLE_HEIGHT - 5
        paddle = self.Paddle(paddle_x, paddle_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT, "red")
        ball = self.Ball(self.WIDTH / 2, paddle_y - self.BALL_RADIUS, self.BALL_RADIUS, "black", 8)
        gauge = self.Gauge(font=self.GAUGE_FONT, x=90, y=150, thickness=20, radius=60, arc_color=(0, 0, 0))
        #                            Originally: x:90  y:WIN_HEIGHT - 120                           

        # TODO: Change the FlickerWindow object instantiation so that there is only one period and flicker_on option
        flicker_window = self.FlickerWindow(flicker_on=True, right_period=3, right_flicker_on_period=1)
        #                                   period=2 - 1 flicker color change every frame,
        #                                              2 frame flicker period,
        #                                              30 Hz
        #                                   period=4 - 1 flicker color change every 2 frames,
        #                                              4 frame flicker period,
        #                                              15 Hz
        
        if flicker_window.get_flicker_location() == 'center':
            reduction_factor_full = SAMPLE_FREQ / flicker_window.get_flicker_frequency()

            # reduction_factor_int is a multiple of new_data_window_size, since the flicker frequency is a submultiple of FPS
            reduction_factor_int = SAMPLE_FREQ // flicker_window.get_flicker_frequency()

            reduction_factor_error = reduction_factor_full - reduction_factor_int
            error_recovery_threshold = np.rint(1 / reduction_factor_error)

            # TODO: Handle the case when error_recovery_threshold has a fractional part
            # error_recovery_threshold_frac = np.modf(error_recovery_threshold)[0]
            # if error_recovery_threshold_frac:
            #     mult_factor = 1 / error_recovery_threshold_frac
            #     error_recovery_threshold = np.rint(mult_factor * error_recovery_threshold)

            error_recovery_counter = 0

            # Debugging - TODO: remove
            # print(f"flicker period: {flicker_window.period}")
            # print(f"flicker frequency: {flicker_window.get_flicker_frequency()}")
            # print(f"reduction_factor_full: {reduction_factor_full}")
            # print(f"reduction_factor_int: {reduction_factor_int}")
            # print(f"reduction_factor_error: {reduction_factor_error}")
            # print(f"error_recovery_threshold: {error_recovery_threshold}")

        else:
            # We need the contralateral lobe for each flicker, so:
            #   Occ. lobe 1 (left) for the right-sided flicker
            #   Occ. lobe 2 (right) for the left-sided flicker
            
            if 'right' in flicker_window.get_flicker_location():
                reduction_factor_full_left = SAMPLE_FREQ / flicker_window.get_right_flicker_frequency()

                # reduction_factor_int is a multiple of new_data_window_size, since the flicker frequency is a submultiple of FPS
                reduction_factor_int_left = SAMPLE_FREQ // flicker_window.get_right_flicker_frequency()
                
                reduction_factor_error_left = reduction_factor_full_left - reduction_factor_int_left
                error_recovery_threshold_left = np.rint(1 / reduction_factor_error_left)
                
                error_recovery_counter_left = 0

            if 'left' in flicker_window.get_flicker_location():
                reduction_factor_full_right = SAMPLE_FREQ / flicker_window.get_left_flicker_frequency()

                # reduction_factor_int is a multiple of new_data_window_size, since the flicker frequency is a submultiple of FPS
                reduction_factor_int_right = SAMPLE_FREQ // flicker_window.get_left_flicker_frequency()
                
                reduction_factor_error_right = reduction_factor_full_right - reduction_factor_int_right
                error_recovery_threshold_right = np.rint(1 / reduction_factor_error_right)
                
                error_recovery_counter_right = 0


        bricks = self.generate_bricks(1, 10)
        lives = MAX_LIVES

        def reset():
            paddle.x = paddle_x
            paddle.y = paddle_y
            ball.x = self.WIDTH/2
            ball.y = paddle_y - self.BALL_RADIUS


        def display_text(text):
            text_render = self.LIVES_FONT.render(text, 1, "red")
            self.win.blit(text_render, (self.WIDTH/2 - text_render.get_width() /
                                2, self.HEIGHT/2 - text_render.get_height()/2))
            pygame.display.update()
            pygame.time.delay(3000)

        # Initializating the EEG variables for blinking detection:
        xs = np.arange(0,100).tolist()
        ys = [list(), list(), list(), list()]
        ys1_last = [0] * 100
        ys2_last = [0] * 100
        ys3_last = [0] * 100
        ys4_last = [0] * 100
        reg1 = LinearRegression(fit_intercept=True)
        reg2 = LinearRegression(fit_intercept=True)
        reg3 = LinearRegression(fit_intercept=True)
        reg4 = LinearRegression(fit_intercept=True)
        blinked_bool = 0
        blinked_count = 0

        # Initializating the EEG variables for flicker-induced SSVEP detection:
        new_data_window_size = SAMPLE_FREQ // FPS
        # roughly 250 Hz // 60 Hz = 4 new data points are added each refresh
        # print(f"new_data_window_size: {new_data_window_size}")

        window_size = 100 
        # initial_window_size = reduction_factor * window_size
        # window_catchup_interval = 6 # to catch up the 0.1(6) elemetns lost each 4 we coallesce
        small_window_prop = 10 # proportion of the total window that the small window occupies
        small_window_begin = window_size - 2 * window_size // small_window_prop
        small_window_end = window_size - window_size // small_window_prop
        # ys_flicker = [list(), list(), list(), list()]
        # ys1_flicker = [0] * window_size
        # ys2_flicker = [0] * window_size
        # ys3_flicker = [0] * window_size
        # ys4_flicker = [0] * window_size
        # ys_occ_1 = []
        # ys_occ_2 = []
        ys_occ_1_new = []
        ys_occ_2_new = []
        ys_occ_1_left_new = []
        ys_occ_1_right_new = []
        ys_occ_2_left_new = []
        ys_occ_2_right_new = []

        # Temporals
        ys_tmp_1_new = []
        ys_tmp_2_new = []
        ys_tmp_1_left_new = []
        ys_tmp_1_right_new = []
        ys_tmp_2_left_new = []
        ys_tmp_2_right_new = []

        # ys_occ_1_old = None
        # ys_occ_2_old = None
        # reg1_flicker = LinearRegression(fit_intercept=True)
        # reg2_flicker = LinearRegression(fit_intercept=True)

        # Initializing the buttons:
        start_button = Button(250, 100, "Start", self.button_background_img, 2)
        difficulty_button = Button(250, 200, "Difficulty: " + str(self.difficulty), self.button_background_img, 2)
        connect_button = Button(250, 300, "Connect", self.button_background_img, 2)
        start_reading_button = Button(250, 400, "Read Sensor", self.button_background_img, 2)
        flicker_test_button = Button(250, 500, "Flicker Test", self.button_background_img, 2)
        quit_button = Button(250, 600, "Quit", self.button_background_img, 2)

        readThread = Thread(target=reading_task, args=[self.EEGSensor])

        connect_text = self.MENU_TEXT_FONT.render('Connection:', True, (0, 0, 0))
        connect_textRect = connect_text.get_rect()
        connect_textRect.center = (WIN_WIDTH - 90, WIN_HEIGHT - 52)

        reading_text = self.MENU_TEXT_FONT.render('Reading:', True, (0, 0, 0))
        reading_textRect = connect_text.get_rect()
        reading_textRect.center = (WIN_WIDTH - 62, WIN_HEIGHT - 22)

        # MainLoop:
        run = True
        while run:
            clock.tick(self.FPS)

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.main_menu = True
                        self.x, self.y = [], []
                        self.current_graph_times = 0
                if event.type == pygame.QUIT:
                    run = False
                    break

            
            if self.main_menu == True:
                self.win.fill((200, 230, 240))

                self.win.blit(connect_text, connect_textRect)
                self.win.blit(reading_text, reading_textRect)
                if self.connection_flag == True:
                    pygame.draw.rect(self.win, (0, 255, 0), pygame.Rect(WIN_WIDTH - 30, WIN_HEIGHT - 60, 20, 20))
                else:
                    pygame.draw.rect(self.win, (255, 0, 0), pygame.Rect(WIN_WIDTH - 30, WIN_HEIGHT - 60, 20, 20))

                if readThread.is_alive():
                    pygame.draw.rect(self.win, (0, 255, 0), pygame.Rect(WIN_WIDTH - 30, WIN_HEIGHT - 30, 20, 20))
                else:
                    pygame.draw.rect(self.win, (255, 0, 0), pygame.Rect(WIN_WIDTH - 30, WIN_HEIGHT - 30, 20, 20))
                
                # Defining button triggers for the menu
                if start_button.draw(self.win):
                    self.main_menu = False
                    self.flicker_test = False

                if difficulty_button.draw(self.win):
                    if self.difficulty < MAX_DIFFICULTY:
                        self.difficulty += 1
                    else:
                        self.difficulty = 1
                    ball.set_vel_scale(self.difficulty)
                    difficulty_button.text = "Difficulty: " + str(self.difficulty)

                if connect_button.draw(self.win):
                    self.EEGSensor.activate_sensor()
    
                    try:
                        if self.EEGSensor.sensor.name == "BrainBit":
                            self.connection_flag = True
                    except Exception as err:
                        print(err)

                if start_reading_button.draw(self.win):
                    if readThread.is_alive():
                        readThread.start()
                    else:
                        readThread = Thread(target=reading_task, args=[self.EEGSensor])
                        readThread.start()

                if flicker_test_button.draw(self.win):
                    self.main_menu = False
                    self.flicker_test = True  

                if quit_button.draw(self.win):
                    run = False
                pygame.display.update()
            
            elif self.flicker_test == True:
                # ret = flicker_window.draw_flicker_test_window(self.win)
                ret = flicker_window.draw_basic_test_window(self.win)
                if ret == 0:
                    continue

                if flicker_window.get_testing_state():

                    # new_batch_size_left = min(new_data_window_size, reduction_factor_int_left - len(ys_occ_1_new))
                    # new_batch_size_right = min(new_data_window_size, reduction_factor_int_right - len(ys_occ_2_new))

                    # The buffers where each lobe (1 and 2) is updated by its corresponding batch size (left and right),
                    #       as determined using the flicker frequency on each side
                    # ys_occ_1_left_new += mySensor.get_data()[0][-new_batch_size_left:]
                    # ys_occ_2_right_new += mySensor.get_data()[1][-new_batch_size_right:]

                    # The buffers where each lobe (1 and 2) is updated by the oher's batch size (right and left),
                    #       as determined using the flicker frequency on each side
                    # ys_occ_1_right_new += mySensor.get_data()[0][-new_batch_size_right:]
                    # ys_occ_2_left_new += mySensor.get_data()[1][-new_batch_size_left:]
                    
                    if flicker_window.get_flicker_location() == 'center':
                        new_batch_size = min(new_data_window_size, reduction_factor_int - len(ys_occ_1_new))
                        ys_occ_1_new += mySensor.get_data()[0][-new_batch_size:] # Occ. electrode 1, left hemisphere
                        ys_occ_2_new += mySensor.get_data()[1][-new_batch_size:] # Occ. electrode 2, right hemisphere

                        ys_tmp_1_new += mySensor.get_data()[0][-new_batch_size:] # Tmp. electrode 1, left hemisphere
                        ys_tmp_2_new += mySensor.get_data()[1][-new_batch_size:] # Tmp. electrode 2, right hemisphere

                        if len(ys_occ_1_new) == reduction_factor_int:
                            # A new batch corresp. to 1 flicker period is complete
                            error_recovery_counter += 1
                            if error_recovery_counter == error_recovery_threshold:
                                ys_occ_1_new.append(mySensor.get_data()[0][-(new_data_window_size + 1)])
                                ys_occ_2_new.append(mySensor.get_data()[0][-(new_data_window_size + 1)])
                                ys_tmp_1_new.append(mySensor.get_data()[0][-(new_data_window_size + 1)])
                                ys_tmp_2_new.append(mySensor.get_data()[0][-(new_data_window_size + 1)])
                                error_recovery_counter = 0

                            ys_occ_1_min_max_diff = max(ys_occ_1_new) - min(ys_occ_1_new)
                            ys_occ_2_min_max_diff = max(ys_occ_2_new) - min(ys_occ_2_new)
                            
                            ys_tmp_1_min_max_diff = max(ys_tmp_1_new) - min(ys_tmp_1_new)
                            ys_tmp_2_min_max_diff = max(ys_tmp_2_new) - min(ys_tmp_2_new)
                            
                            flicker_window.log_plot_data(ys_occ_1_min_max_diff, 'occ_1')
                            flicker_window.log_plot_data(ys_occ_2_min_max_diff, 'occ_2')
                            flicker_window.log_data(ys_occ_1_new, 'occ_1')
                            flicker_window.log_data(ys_occ_2_new, 'occ_2')

                            flicker_window.log_plot_data(ys_tmp_1_min_max_diff, 'tmp_1')
                            flicker_window.log_plot_data(ys_tmp_2_min_max_diff, 'tmp_2')
                            flicker_window.log_data(ys_tmp_1_new, 'tmp_1')
                            flicker_window.log_data(ys_tmp_2_new, 'tmp_2')
                            
                            ys_occ_1_new = []
                            ys_occ_2_new = []
                            ys_tmp_1_new = []
                            ys_tmp_2_new = []
                    else:
                        if 'right' in flicker_window.get_flicker_location():
                            # Processing the data registered by the left (1) occipital lobe at the right eye flicker frequency,
                            #   and logging the right (2) occipital lobe data at the same (contralatral) frequency for reference
                            
                            new_batch_size_left = min(new_data_window_size, reduction_factor_int_left - len(ys_occ_1_left_new))

                            # Occ. lobe 1 (left) buffer is updated with its corresponding left batch size,
                            #   while the occ. lobe 2 (right) buffer is updated with the same (so contralateral)
                            #   left batch size
                            ys_occ_1_left_new += mySensor.get_data()[0][-new_batch_size_left:]
                            ys_occ_2_left_new += mySensor.get_data()[1][-new_batch_size_left:]
                            
                            ys_tmp_1_left_new += mySensor.get_data()[0][-new_batch_size_left:]
                            ys_tmp_2_left_new += mySensor.get_data()[1][-new_batch_size_left:]

                            if len(ys_occ_1_left_new) == reduction_factor_int_left:
                                # A new batch corresp. to 1 flicker period is complete
                                error_recovery_counter_left += 1
                                if error_recovery_counter_left == error_recovery_threshold_left:
                                    ys_occ_1_left_new.append(mySensor.get_data()[0][-(new_data_window_size + 1)])
                                    ys_occ_2_left_new.append(mySensor.get_data()[1][-(new_data_window_size + 1)])
                                    ys_tmp_1_left_new.append(mySensor.get_data()[0][-(new_data_window_size + 1)])
                                    ys_tmp_2_left_new.append(mySensor.get_data()[1][-(new_data_window_size + 1)])
                                    error_recovery_counter_left = 0

                                ys_occ_1_left_min_max_diff = max(ys_occ_1_left_new) - min(ys_occ_1_left_new)
                                flicker_window.log_plot_data(ys_occ_1_left_min_max_diff, 'occ_1_left')
                                flicker_window.log_data(ys_occ_1_left_new, 'occ_1_left')
                                ys_occ_1_left_new = []

                                # The second lobe, for comparison
                                ys_occ_2_left_min_max_diff = max(ys_occ_2_left_new) - min(ys_occ_2_left_new)
                                flicker_window.log_plot_data(ys_occ_2_left_min_max_diff, 'occ_2_left')
                                flicker_window.log_data(ys_occ_2_left_new, 'occ_2_left')
                                ys_occ_2_left_new = []

                                # Temporals
                                ys_tmp_1_left_min_max_diff = max(ys_tmp_1_left_new) - min(ys_tmp_1_left_new)
                                flicker_window.log_plot_data(ys_tmp_1_left_min_max_diff, 'tmp_1_left')
                                flicker_window.log_data(ys_tmp_1_left_new, 'tmp_1_left')
                                ys_tmp_1_left_new = []

                                ys_tmp_2_left_min_max_diff = max(ys_tmp_2_left_new) - min(ys_tmp_2_left_new)
                                flicker_window.log_plot_data(ys_tmp_2_left_min_max_diff, 'tmp_2_left')
                                flicker_window.log_data(ys_tmp_2_left_new, 'tmp_2_left')
                                ys_tmp_2_left_new = []

                        if 'left' in flicker_window.get_flicker_location():
                            # Processing the data registered by the right (2) occipital lobe at the left eye flicker frequency,
                            #   and logging the left (1) occipital lobe data at the same (contralatral) frequency for reference
                        
                            new_batch_size_right = min(new_data_window_size, reduction_factor_int_right - len(ys_occ_2_right_new))
                        
                            # Occ. lobe 2 (right) buffer is updated with its corresponding right batch size,
                            #   while the occ. lobe 1 (left) buffer is updated with the same (so contralateral)
                            #   right batch size
                            ys_occ_2_right_new += mySensor.get_data()[1][-new_batch_size_right:]
                            ys_occ_1_right_new += mySensor.get_data()[0][-new_batch_size_right:]
                            
                            ys_tmp_2_right_new += mySensor.get_data()[1][-new_batch_size_right:]
                            ys_tmp_1_right_new += mySensor.get_data()[0][-new_batch_size_right:]

                            if len(ys_occ_2_right_new) == reduction_factor_int_right:
                                # A new batch corresp. to 1 flicker period is complete
                                error_recovery_counter_right += 1
                                if error_recovery_counter_right == error_recovery_threshold_right:
                                    ys_occ_2_right_new.append(mySensor.get_data()[1][-(new_data_window_size + 1)])
                                    ys_occ_1_right_new.append(mySensor.get_data()[0][-(new_data_window_size + 1)])
                                    ys_tmp_2_right_new.append(mySensor.get_data()[1][-(new_data_window_size + 1)])
                                    ys_tmp_1_right_new.append(mySensor.get_data()[0][-(new_data_window_size + 1)])
                                    error_recovery_counter_left = 0

                                ys_occ_2_right_min_max_diff = max(ys_occ_2_right_new) - min(ys_occ_2_right_new)
                                flicker_window.log_plot_data(ys_occ_2_right_min_max_diff, 'occ_2_right')
                                flicker_window.log_data(ys_occ_2_right_new, 'occ_2_right')
                                ys_occ_2_right_new = []

                                # The first lobe, for comparison
                                ys_occ_1_right_min_max_diff = max(ys_occ_1_right_new) - min(ys_occ_1_right_new)
                                flicker_window.log_plot_data(ys_occ_1_right_min_max_diff, 'occ_1_right')
                                flicker_window.log_data(ys_occ_1_right_new, 'occ_1_right')
                                ys_occ_1_right_new = []

                                # Temporal
                                ys_tmp_2_right_min_max_diff = max(ys_tmp_2_right_new) - min(ys_tmp_2_right_new)
                                flicker_window.log_plot_data(ys_tmp_2_right_min_max_diff, 'tmp_2_right')
                                flicker_window.log_data(ys_tmp_2_right_new, 'tmp_2_right')
                                ys_tmp_2_right_new = []

                                ys_tmp_1_right_min_max_diff = max(ys_tmp_1_right_new) - min(ys_tmp_1_right_new)
                                flicker_window.log_plot_data(ys_tmp_1_right_min_max_diff, 'tmp_1_right')
                                flicker_window.log_data(ys_tmp_1_right_new, 'tmp_1_right')
                                ys_tmp_1_right_new = []

            else:
                # Stage 1: Preparing the EEG data (absolute amplitude correction) for the blinking detection
                for j in range(4):
                    ys[j] = mySensor.get_data()[j][-100:]

                # Simply uncomment the respective lines if you want to use the other channels as well
                if len(ys[0]) == 100:
                    reg1.fit(np.array(xs).reshape(-1,1), np.array(ys[0]))
                    reg2.fit(np.array(xs).reshape(-1,1), np.array(ys[1]))
                    #reg3.fit(np.array(xs).reshape(-1,1), np.array(ys[2]))
                    # reg4.fit(np.array(xs).reshape(-1,1), np.array(ys[3]))

                    for t in range(100):
                        ys1_curr_drift = reg1.coef_ * (100 - t)
                        ys2_curr_drift = reg2.coef_ * (100 - t)
                        #ys3_curr_drift = reg3.coef_ * (100 - t)
                        #ys4_curr_drift = reg4.coef_ * (100 - t)
                        
                        ys1_last[t] = ys[0][t] + ys1_curr_drift - reg1.intercept_
                        ys2_last[t] = ys[1][t] + ys2_curr_drift - reg2.intercept_
                        #ys3_last[t] = ys[2][t] + ys3_curr_drift - reg3.intercept_
                        # ys4_last[t] = ys[3][t] + ys4_curr_drift - reg4.intercept_

                # TODO: Correct the amplitude drift without affecting the reltative amplitude differences
                # self.log_current_data_buffer(ys1_last, 'occ_1')
                # self.log_current_data_buffer(ys2_last, 'occ_2')

                # Moving the ball based on player controls (currently: blinking)
                if self.direction != 0 and ball.y_vel > 0:
                    if self.direction == 1:
                        if blinked_bool == 0:
                            # TODO: Set a maximum threshold to prevent detecting the pulse as a blink
                            if (np.average(ys1_last[0:99]) - np.average(ys1_last[80:90]) > BLINK_THRESHOLD) and paddle.x + paddle.width + paddle.VEL <= self.WIDTH:
                                # paddle.move(self.direction)
                                paddle.move_to_final_location(ball.x_predict, self.steps_needed, self.direction)
                                blinked_bool = 1
                                self.steps_taken += 1
                        else:
                            blinked_count += 1
                            if blinked_count >= 10:
                                blinked_bool = 0
                                blinked_count = 0
                    elif self.direction == -1:
                        if blinked_bool == 0:
                            if (np.average(ys1_last[0:99]) - np.average(ys1_last[80:90]) > BLINK_THRESHOLD) and paddle.x - paddle.VEL >= 0:
                                # paddle.move(self.direction)
                                paddle.move_to_final_location(ball.x_predict, self.steps_needed, self.direction)
                                blinked_bool = 1
                                self.steps_taken += 1
                        else:
                            blinked_count += 1
                            if blinked_count >= 10:
                                blinked_bool = 0
                                blinked_count = 0

                self.fix_ball_conditions(ball)
                ball.move()
                self.ball_collision(ball, paddle)
                self.ball_paddle_collision(ball, paddle, gauge)

                for brick in bricks[:]:
                    if brick.collide(ball):
                        paddle.x_initial = paddle.x
                        self.calculate_steps_needed(paddle, ball.x_predict)

                    if brick.health <= 0:
                        bricks.remove(brick)
                if ball.x_predict != -np.inf:
                    self.calculate_steps_and_direction(paddle, ball.x_predict)

                # Lives check:
                # Reset the ball if there are live left, else reset the game
                if ball.y + ball.radius >= self.HEIGHT:
                    lives -= 1

                    if self.steps_needed != 0:
                        gauge.change_percentage(int(np.floor(100 - 100 * (np.abs(self.steps_needed - self.steps_taken) / self.steps_needed))))
                    else:
                        gauge.change_percentage(100)
                    self.steps_taken = 0
                    self.steps_needed = 0
                    ball.x = paddle.x + paddle.width/2
                    ball.y = paddle.y - self.BALL_RADIUS
                    ball.set_vel(ball.x_vel, BALL_INITIAL_VEL * -1)

                if lives <= 0:
                    bricks = self.generate_bricks(1, 10)
                    lives = MAX_LIVES
                    reset()
                    display_text("You Lost!")

                    self.steps_needed = 0
                    self.steps_taken = 0
                    self.steps = 0
                    self.direction = 0

                if len(bricks) == 0:
                    bricks = self.generate_bricks(1, 10)
                    lives = MAX_LIVES
                    reset()
                    display_text("You Won!")

                    self.steps_needed = 0
                    self.steps_taken = 0
                    self.steps = 0
                    self.direction = 0

                self.draw(self.win, paddle, ball, bricks, gauge, lives, self.direction, 1)

        
        if readThread.is_alive():
            self.EEGSensor.threading_event.set()
            readThread.join()
        self.EEGSensor.deactivate_sensor()
        print("\nDeactivation of sensor completed!\n")
        pygame.quit()

if __name__ == "__main__":
    brickGame = Game()
    brickGame.main()