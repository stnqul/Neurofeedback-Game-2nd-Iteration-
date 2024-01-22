import pygame

class Button():
    def __init__(self, x, y, text, image, scale):
        self.width = image.get_width()
        self.height = image.get_height()
        self.scale = scale
        self.image = pygame.transform.scale(image, (int(self.width * scale), int(self.height * scale)))
        self.text = text
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)
        self.BUTTON_FONT = pygame.font.SysFont("comicsans", size=22, bold=True)

        self.clicked = False

    def draw(self, win):
        action = False

        win.blit(self.image, (self.rect.x, self.rect.y))
        text_render = self.BUTTON_FONT.render(self.text, 1, "red")
        text_width, text_height = self.BUTTON_FONT.size(self.text)
        text_pos = tuple(map(lambda i, j: i + j, self.rect.topleft, (self.width - text_width / 2, self.height / 2)))
        win.blit(text_render, text_pos)
        
        mouse_pos = pygame.mouse.get_pos()

        if self.rect.collidepoint(mouse_pos):
            if pygame.mouse.get_pressed()[0] == 1 and self.clicked == False:
                self.clicked = True
                action = True
            
        if pygame.mouse.get_pressed()[0] == 0:
            self.clicked = False

        return action
