import eel, socket

""" 
BRON
https://pypi.org/project/Eel/
https://nitratine.net/blog/post/python-gui-using-chrome/
 """

class GameView:
    def __init__(self):
        eel.init("./game/web")
        self.start()
        
    
    def start(self):
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        eel.start("index.html", port=80, host=ip, block=False)
    
    def change_game_view(self, grid, score, shape_queue):
        test = "Hallo"
        eel.changeGameGrid(grid, score, shape_queue)
        print("send!")
    
    @eel.expose
    def startBackend(data):
        eel.startFront(data)


""" 
a = GameView()

while True:
    eel.sleep(1) """