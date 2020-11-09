import eel, socket


class GameView:
    def __init__(self):
        eel.init("./game/web")
        self.start()
        
    
    def start(self):
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        eel.start("index.html", port=80, host=ip)



a = GameView()
