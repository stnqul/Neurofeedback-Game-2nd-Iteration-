import pygame
import pygame.gfxdraw
import math
import numpy as np
import os

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
# matplotlib.use("agg")
# import matplotlib.backends.backend_agg as agg
# import random
# from itertools import count

from button import Button
from sensor import Sensor
from threading import Thread

from sklearn.linear_model import LinearRegression


# Initializations of some static variables:
SESSION_LENGTH = 900
BLINK_THRESHOLD = 0.00005
WIN_WIDTH, WIN_HEIGHT = 700, 700 # !Original height: 800
PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_VEL = 100, 20, 100
BALL_RADIUS, BALL_INITIAL_VEL = 10, 4 # !Original velocity: 4
MAX_DIFFICULTY, MAX_LIVES = 5, 5
FPS = 60 # !Originally: 60

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
        self.button_background_img = pygame.image.load('images/UI_Flat_Frame_02_Horizontal.png')
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
        
        # *Alt combo: bg azure & fl aliceblue
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
        def __init__(self, period = 1):
            self.cross_width = 20
            self.cross_height = 120
            self.cross_x = WIN_WIDTH / 2 - self.cross_width / 2 - 1
            self.cross_y = WIN_HEIGHT / 2 - self.cross_height / 2 - 1
            
            self.flicker_width = 121
            self.flicker_height = 121
            self.cross_flicker_distance = 120

            self.period = period
            self.flicker_count = 0

        def draw_flicker_test_window(self, win):
            """
            Draw method for the flicker test
            """
            win.fill("white")

            color_increment = 255 / self.period
            color = (0,0,0)

            # Focus cross
            pygame.draw.rect(win, color,
                             (self.cross_x, self.cross_y,
                              self.cross_width, self.cross_height))
            
            horizontal_x = self.cross_x - (self.cross_height / 2 - self.cross_width / 2)
            horizontal_y = self.cross_y + (self.cross_height / 2 - self.cross_width / 2)
            horizontal_width = self.cross_height
            horizontal_height = self.cross_width

            pygame.draw.rect(win, color,
                             (horizontal_x, horizontal_y,
                              horizontal_width, horizontal_height))

            # Flickering areas
            if self.flicker_count == self.period:
                self.flicker_count = 0
                color = (0,0,0)
            else:
                self.flicker_count += 1
                color = tuple(map((lambda c : c + color_increment), color))

            flicker_left_x = horizontal_x - self.flicker_width - self.cross_flicker_distance
            flicker_right_x = horizontal_x + horizontal_width + self.cross_flicker_distance
            flicker_y = self.cross_y

            pygame.draw.rect(win, color,
                             (flicker_left_x, flicker_y,
                              self.flicker_width, self.flicker_height))
            pygame.draw.rect(win, color,
                             (flicker_right_x, flicker_y,
                              self.flicker_width, self.flicker_height))

            pygame.display.update()


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

        # if direction != 0:
        #     if direction == 1:
        #         flicker_x = cross_x_mid + horizontal_width / 2 + cross_flicker_distance
        #     else: # direction == -1
        #         flicker_x = cross_x_mid - horizontal_width / 2 - cross_flicker_distance - flicker_width

        #     # flicker_left_x = cross_x_mid - horizontal_width / 2 - cross_flicker_distance - flicker_width
        #     # flicker_right_x = cross_x_mid + horizontal_width / 2 + cross_flicker_distance
        #     flicker_y = paddle.get_y() - flicker_height - 30

        #     # print(f"Left x diff: {cross_x_mid - (flicker_left_x + flicker_width)}\n")
        #     # print(f"Right x diff: {flicker_right_x - cross_x_mid}\n\n")

        #     pygame.draw.rect(win, color,
        #                         (flicker_x, flicker_y,
        #                         flicker_width, flicker_height))
        
        flicker_x = cross_x_mid + horizontal_width / 2 + cross_flicker_distance
        flicker_y = paddle.get_y() - flicker_height - 30
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
        #                            Originally:   x: 90    y: WIN_HEIGHT - 120                           #
        flicker_window = self.FlickerWindow(period=1)

        bricks = self.generate_bricks(1, 10)
        lives = MAX_LIVES

        graph_step = 100
        current_graph_iterations = 0
        graph_x, graph_y = (0,0)

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

        # Initializating the EEG data variables for blinking detection:
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

        # Initializating the EEG data variables for flicker-induced SSVEP detection:
        window_size = 250
        xs_flicker = np.arange(0, window_size).tolist()
        ys_flicker = [list(), list(), list(), list()]
        ys1_flicker = [0] * window_size # The sample frequency of the BrainBit headband is 250Hz
        ys2_flicker = [0] * window_size
        ys3_flicker = [0] * window_size
        ys4_flicker = [0] * window_size
        reg1_flicker = LinearRegression(fit_intercept=True)
        reg2_flicker = LinearRegression(fit_intercept=True)

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
                            # mySensor.print_sensor_information()
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
            
            # TODO: Under works
            elif self.flicker_test == True:
                flicker_window.draw_flicker_test_window(self.win)
                
                for j in range(4):
                    ys[j] = mySensor.get_data()[j][-100:]
                
                if len(ys[0]) == 100:
                    #reg1.fit(np.array(xs).reshape(-1,1), np.array(ys[0]))
                    #reg2.fit(np.array(xs).reshape(-1,1), np.array(ys[1]))
                    #reg3.fit(np.array(xs).reshape(-1,1), np.array(ys[2]))
                    reg4.fit(np.array(xs).reshape(-1,1), np.array(ys[3]))
                    
                    for t in range(100):
                        #ys_curr_drift = reg1.coef_ * (100 - t)
                        #ys2_curr_drift = reg2.coef_ * (100 - t)
                        #ys3_curr_drift = reg3.coef_ * (100 - t)
                        ys4_curr_drift = reg4.coef_ * (100 - t)
                        
                        #ys_last[t] = ys[0][t] + ys_curr_drift - reg1.intercept_
                        #ys2_last[t] = ys[1][t] + ys2_curr_drift - reg2.intercept_
                        #ys3_last[t] = ys[2][t] + ys3_curr_drift - reg3.intercept_
                        ys4_last[t] = ys[3][t] + ys4_curr_drift - reg4.intercept_

                display_data = list(np.array(ys[3]))

                def animate(i):
                    # if self.x == []:
                    #     self.x = list(range(len(display_data)))
                    # else:
                    #     self.x += list(range(x[-1] + 1, x[-1] + len(display_data) + 1))
                    self.x += list(range(self.current_graph_times,
                                         self.current_graph_times + len(display_data)))
                    self.y += display_data
                    
                    self.current_graph_times += len(self.x)

                    print(display_data)
                    plt.xlim(i-30,i+3)
                    plt.style.use("ggplot")
                    plt.plot(self.x,self.y, scalex=True, scaley=True, color="red")
                    
                anim = FuncAnimation(self.fig, animate, interval=100)
                # plt.show()
                
                # time = np.array(range(graph_step * current_graph_iterations,
                #                       graph_step * (current_graph_iterations + 1)))
                # plt.xlim(time[0] - 30,
                #          time[99] + 3)
                # ax.plot(time, ys4_last)
                # current_graph_iterations += 1

                # fig.canvas.draw()
                # canvas = fig.canvas
                # renderer = canvas.get_renderer()
                # raw_data = renderer.tostring_rgb()
                # plt.close()

                # size = canvas.get_width_height()

                # surf = pygame.image.fromstring(raw_data, size, "RGB")
                # self.screen.blit(surf, (graph_x, graph_y))
                # pygame.display.flip()
            
            else:
                # Stage 1: Preparing the EEG data (absolute amplitude correction) for the blinking detection
                for j in range(4):
                    ys[j] = mySensor.get_data()[j][-100:]

                # Simply uncomment the respective lines if you want to use the other channels as well
                if len(ys[0]) == 100:
                    #reg1.fit(np.array(xs).reshape(-1,1), np.array(ys[0]))
                    #reg2.fit(np.array(xs).reshape(-1,1), np.array(ys[1]))
                    #reg3.fit(np.array(xs).reshape(-1,1), np.array(ys[2]))
                    reg4.fit(np.array(xs).reshape(-1,1), np.array(ys[3]))

                    for t in range(100):
                        #ys1_curr_drift = reg1.coef_ * (100 - t)
                        #ys2_curr_drift = reg2.coef_ * (100 - t)
                        #ys3_curr_drift = reg3.coef_ * (100 - t)
                        ys4_curr_drift = reg4.coef_ * (100 - t)
                        
                        #ys1_last[t] = ys[0][t] + ys1_curr_drift - reg1.intercept_
                        #ys2_last[t] = ys[1][t] + ys2_curr_drift - reg2.intercept_
                        #ys3_last[t] = ys[2][t] + ys3_curr_drift - reg3.intercept_
                        ys4_last[t] = ys[3][t] + ys4_curr_drift - reg4.intercept_

                # Stage 2: Preparing the EEG data for flicker-induced SSVEP detection
                for j in range(2): # We only need the occipital lobe data for now
                    ys_flicker[j] = mySensor.get_data()[j][-window_size:]

                    # Method 1: Reducing the sample size from 250 to 60, i.e. to the flicker frequency
                    # t = 0
                    # it = 1
                    # ys_flicker_reduced = []
                    # while t < sample_len:
                    #     curr_window = ys_flicker[j][t : min(t+4, sample_len)]
                        
                    #     # Correcting every 4 iterations for the 0.1(6) in 250/60 = 4.1(6)
                    #     if it % 4 == 0 and sample_len - t > 4:
                    #         curr_window.append(ys_flicker[j][t+4])
                    #         t += 5
                    #     else:
                    #         t += 4
                    #     ys_flicker_reduced.append(sum(curr_window) / len(curr_window))
                    #     it += 1
                    
                    # ys_flicker[j] = ys_flicker_reduced

                # Method 2: Using a moving window
                # window_average = sum(ys_flicker) / len(ys_flicker)
                # ys_flicker = [ys_flicker[t] - window_average for t in range(250)]

                if len(ys_flicker[0]) == window_size:
                    reg1_flicker.fit(np.array(xs_flicker).reshape(-1,1), np.array(ys_flicker[0]))
                    reg2_flicker.fit(np.array(xs_flicker).reshape(-1,1), np.array(ys_flicker[1]))
                    reg1_flicker_slope = reg1_flicker.coef_
                    reg1_flicker_slope = reg1_flicker.coef_

                    for t in range(window_size):
                        ys1_flicker[0]

                # TODO: Correct the amplitude drift without affecting the reltative amplitude differences

                if self.direction != 0 and ball.y_vel > 0:
                    if self.direction == 1:
                        if blinked_bool == 0:
                            if (np.average(ys4_last[0:99]) - np.average(ys4_last[80:90]) > BLINK_THRESHOLD) and paddle.x + paddle.width + paddle.VEL <= self.WIDTH:
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
                            if (np.average(ys4_last[0:99]) - np.average(ys4_last[80:90]) > BLINK_THRESHOLD) and paddle.x - paddle.VEL >= 0:
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