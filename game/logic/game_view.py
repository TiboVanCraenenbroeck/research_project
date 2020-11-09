import eel


class GameView:
    def __init__(self):
        eel.init("./game\web")
        eel.start("index.html")
    
a = GameView()