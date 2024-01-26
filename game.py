from constants import *
import pygame
import snake
import apple






class Game():
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.surface = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
        self.game_speed = 0.5
        self.snake = snake.Snake(self.surface)
        self.apple = apple.Apple(self.surface)
        self.prev_time = pygame.time.get_ticks()


    def set_caption(self, cap):
        pygame.display.set_caption(cap)


    def step(self, move):
        self.clock.tick(TICK_RATE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                self.snake.keyboard_controler(event.key)
                if event.key == pygame.K_q:
                    pygame.quit()



        self.snake.ai_move(move)
        self.snake.update()
        self.apple.update(self.snake)
        self.draw()

        reward = int(self.snake.check_apple(self.apple))

        done = int(self.snake.check_bounds())
        if done:
            reward = -1


        return self.get_state(), reward, done




    def reset(self):
        self.snake.reset()
        return self.get_state()

    def draw(self):
        self.surface.fill((0, 0, 0))#background
        self.snake.draw()
        self.apple.draw()
        pygame.display.flip()



    def update(self):
        self.snake.update()
        self.apple.update(self.snake)



    def get_state(self):
        sn = self.snake.get_coords()
        apl = self.apple.get_coords()
        return [sn[0], sn[1], sn[2], sn[3], apl[0], apl[1]]




    def ai_move(self, move):
        if move == 0:
            self.snake.move_up()
        if move == 1:
            self.snake.move_right()
        if move == 2:
            self.snake.move_down()
        if move == 3:
            self.snake.move_left()
        # call snake.move_up().....
if __name__ == "__main__":
    g = Game()
    g.main()
