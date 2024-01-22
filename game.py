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



    def run(self):



        self.clock.tick(TICK_RATE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                self.snake.keyboard_controler(event.key)
                if event.key == pygame.K_q:
                    pygame.quit()


        delta = pygame.time.get_ticks() - self.prev_time
        if delta > self.game_speed * 1000:
            self.prev_time = pygame.time.get_ticks()
            self.draw()
            self.update()





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
        return [sn[0], sn[1], sn[2][0], sn[2][1], apl[0], apl[1]]

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
